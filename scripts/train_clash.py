"""Teach the fly brain basic Clash Royale, then let it practise.

Phase 1, coaching: the brain watches a scripted coach (flybrain/envs/clash/
strategy.py) and is rewarded with dopamine for the coach's move in each
situation. Moves actually played are the coach's at first, then more and more
the brain's own (so it also learns from situations its own play creates).

Phase 2, practice: the brain plays matches on its own against bots of rising
difficulty (random -> basic -> coach) and learns from tower damage, crowns
and wins through the same dopamine rule. It moves up a level once it wins
often enough.

    python scripts/train_clash.py                       # CPU, synthetic brain
    python scripts/train_clash.py --data data/mushroom_body.npz --device cuda --batch 32
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", message="Sparse")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from flybrain.agent import AgentConfig, FlyAgent  # noqa: E402
from flybrain.connectome import load  # noqa: E402
from flybrain.envs.clash.env import HEADS, N_CHANNELS, ClashEnv  # noqa: E402
from flybrain.envs.clash.strategy import BasicBot, Coach, RandomBot  # noqa: E402
from flybrain.plasticity import PlasticityConfig  # noqa: E402

OPPONENTS = {"random": lambda s: RandomBot(s), "basic": lambda s: BasicBot(s), "coach": lambda s: Coach()}
CURRICULUM = ["random", "basic", "coach"]


def make_agent(args, probe=None):
    conn = load(args.data) if args.data != "synthetic" else load(
        "synthetic", n_pn=680, n_kc=2500, n_mbon=96, n_dan=120, n_glomeruli=120)
    return FlyAgent(
        conn, N_CHANNELS, HEADS, device=args.device, probe=probe,
        plast_cfg=PlasticityConfig(mode=args.mode, lr=args.lr, recovery=args.recovery,
                                   trace_decay=args.trace_decay),
        cfg=AgentConfig(action_groups="random", seed=args.seed, decision_steps=args.decision_ms,
                        epsilon=args.epsilon, rpe_rate=args.rpe_rate),
    )


def probe_states(n=64, seed=123):
    """Typical game situations (coach vs basic bot) for calibrating firing rates."""
    env = ClashEnv(8, OPPONENTS["basic"], seed=seed)
    env.reset()
    out = []
    while sum(len(o) for o in out) < n:
        acts, _ = env.teacher()
        f, _, done = env.step(acts)
        if env.sims[0].time % 10 < 1:
            out.append(f)
        if done:
            env.reset()
    return np.concatenate(out)[:n]


def evaluate(agent, opponent, matches, seed):
    """Win/draw/loss vs ``opponent`` with no exploration and no learning."""
    eps, agent.cfg.epsilon = agent.cfg.epsilon, 0.0
    env = ClashEnv(matches, OPPONENTS[opponent], seed=seed)
    f, done = env.reset(), False
    while not done:
        f, _, done = env.step(agent.act(f, env.masks(), learn=False))
    agent.cfg.epsilon = eps
    res = env.results()
    return dict(win=float((res == 1).mean()), draw=float((res == 0).mean()), loss=float((res == -1).mean()))


def coach_phase(agent, args, log):
    env = ClashEnv(args.batch, OPPONENTS[args.coach_opponent], seed=args.seed)
    n_rounds = args.coach_matches // args.batch
    t0 = time.time()
    for rnd in range(n_rounds):
        mix = args.self_play_max * rnd / max(1, n_rounds - 1)  # chance the brain's own move is played
        f, done = env.reset(), False
        agree = dict(decisions=0, same_card=0, coach_plays=0, same_play_card=0, same_lane=0)
        while not done:
            masks = env.masks()
            brain = agent.act(f, masks, learn=False)
            coach, active = env.teacher()
            agent.teach(coach, active, strength=args.teach_strength)
            live = ~env.done
            agree["decisions"] += live.sum()
            agree["same_card"] += (live & (brain[:, 0] == coach[:, 0])).sum()
            plays = live & (coach[:, 0] > 0)
            agree["coach_plays"] += plays.sum()
            agree["same_play_card"] += (plays & (brain[:, 0] == coach[:, 0])).sum()
            agree["same_lane"] += (plays & (brain[:, 0] == coach[:, 0]) & (brain[:, 1] == coach[:, 1])).sum()
            use_brain = agent.rng.random(args.batch) < mix
            f, _, done = env.step(np.where(use_brain[:, None], brain, coach))
        row = dict(phase="coach", round=rnd + 1, matches=(rnd + 1) * args.batch, self_play=round(mix, 2),
                   agree_any=agree["same_card"] / agree["decisions"],
                   agree_when_coach_plays=agree["same_play_card"] / max(1, agree["coach_plays"]),
                   agree_card_and_lane=agree["same_lane"] / max(1, agree["coach_plays"]),
                   seconds=round(time.time() - t0))
        if (rnd + 1) % args.eval_every == 0 or rnd + 1 == n_rounds:
            row.update({f"vs_{o}": evaluate(agent, o, args.eval_matches, seed=10_000 + rnd) for o in ("random", "basic")})
        log(row)


def practice_phase(agent, args, log):
    level = 0
    n_rounds = args.practice_matches // args.batch
    recent = []
    t0 = time.time()
    for rnd in range(n_rounds):
        opp = CURRICULUM[level]
        env = ClashEnv(args.batch, OPPONENTS[opp], seed=args.seed + 7 * (rnd + 1))
        f, done = env.reset(), False
        agent.begin_episode(args.batch)
        while not done:
            a = agent.act(f, env.masks(), learn=True)
            if args.keep_teaching:
                coach, active = env.teacher()
                saved = (agent.rule.chosen.clone(), agent.rule.unchosen.clone())
                agent.teach(coach, active, strength=args.keep_teaching)
                agent.rule.chosen, agent.rule.unchosen = saved
            f, r, done = env.step(a)
            agent.reward(r)
        res = env.results()
        recent = (recent + list(res))[-args.promote_window:]
        row = dict(phase="practice", round=rnd + 1, matches=(rnd + 1) * args.batch, opponent=opp,
                   win=float((res == 1).mean()), draw=float((res == 0).mean()), loss=float((res == -1).mean()),
                   seconds=round(time.time() - t0))
        if (len(recent) >= args.promote_window and np.mean(np.array(recent) == 1) >= args.promote_at
                and level + 1 < len(CURRICULUM)):
            level += 1
            recent = []
            row["promoted_to"] = CURRICULUM[level]
        if (rnd + 1) % args.eval_every == 0 or rnd + 1 == n_rounds:
            row.update({f"vs_{o}": evaluate(agent, o, args.eval_matches, seed=20_000 + rnd) for o in CURRICULUM})
        log(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="synthetic")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=16, help="matches played in parallel")
    ap.add_argument("--coach-matches", type=int, default=160)
    ap.add_argument("--coach-opponent", default="basic", choices=list(OPPONENTS))
    ap.add_argument("--self-play-max", type=float, default=0.5,
                    help="by the end of coaching, chance each move is the brain's own")
    ap.add_argument("--teach-strength", type=float, default=1.0)
    ap.add_argument("--practice-matches", type=int, default=160)
    ap.add_argument("--keep-teaching", type=float, default=0.0,
                    help="during practice, also nudge toward the coach with this strength")
    ap.add_argument("--promote-at", type=float, default=0.6, help="win rate needed to face the next bot")
    ap.add_argument("--promote-window", type=int, default=32)
    ap.add_argument("--mode", default="depression", choices=["depression", "bidirectional"])
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--recovery", type=float, default=0.01)
    ap.add_argument("--trace-decay", type=float, default=0.7)
    ap.add_argument("--rpe-rate", type=float, default=0.0)
    ap.add_argument("--decision-ms", type=int, default=40)
    ap.add_argument("--epsilon", type=float, default=0.03)
    ap.add_argument("--eval-every", type=int, default=5, help="rounds between evaluations")
    ap.add_argument("--eval-matches", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    ap.add_argument("--out", default="runs/clash")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.jsonl", "w")

    def log(row):
        logf.write(json.dumps(row) + "\n")
        logf.flush()
        keys = [k for k in row if not k.startswith("vs_")]
        print("  ".join(f"{k}={row[k]:.2f}" if isinstance(row[k], float) else f"{k}={row[k]}" for k in keys),
              flush=True)
        for k in row:
            if k.startswith("vs_"):
                v = row[k]
                print(f"      eval {k}: win {v['win']:.2f} draw {v['draw']:.2f} loss {v['loss']:.2f}", flush=True)

    agent = make_agent(args, probe=probe_states())
    print("calibration:", agent.calibration)
    log(dict(phase="start", **{f"vs_{o}": evaluate(agent, o, args.eval_matches, seed=999) for o in CURRICULUM}))
    if args.coach_matches:
        coach_phase(agent, args, log)
        torch.save(dict(agent=agent.state_dict(), args=vars(args)), out / "brain_coached.pt")
    if args.practice_matches:
        practice_phase(agent, args, log)
    torch.save(dict(agent=agent.state_dict(), args=vars(args)), out / "brain.pt")
    print(f"wrote {out}/")


if __name__ == "__main__":
    main()
