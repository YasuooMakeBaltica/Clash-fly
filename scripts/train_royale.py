"""Teach the fly brain Clash Royale with the full card pool.

1. Record: the coach plays many matches with random decks (and the fly's own
   deck from decks/fly.txt part of the time) against bots; every decision
   is saved as (situation channels, allowed roles, coach's role and lane).
2. Teach: the brain sees each recorded situation and the coach's choice is
   paired with reward dopamine (KC->MBON plasticity with synaptic scaling).
3. Test: the brain plays matches on its own (random decks and the fly deck)
   against the random and basic bots.

    python scripts/train_royale.py                         # CPU, synthetic brain
    python scripts/train_royale.py --data data/mushroom_body.npz --device cuda
    python scripts/train_royale.py --fly-deck decks/fly.txt --record-matches 128 --epochs 2
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore", message="Sparse")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from flybrain.agent import AgentConfig, FlyAgent  # noqa: E402
from flybrain.connectome import load  # noqa: E402
from flybrain.envs.royale.decks import load_deck  # noqa: E402
from flybrain.envs.royale.env import HEADS, N_CHANNELS, SPLIT_HEADS, RoyaleEnv, decode, encode, role_mask  # noqa: E402
from flybrain.envs.royale.strategy import BasicBot, Coach, RandomBot  # noqa: E402
from flybrain.plasticity import PlasticityConfig  # noqa: E402

OPPONENTS = {"random": lambda s: RandomBot(s), "basic": lambda s: BasicBot(s), "coach": lambda s: Coach()}


def make_agent(args, probe=None):
    if args.data == "synthetic":
        conn = load("synthetic", n_pn=680, n_kc=getattr(args, "n_kc", 2500), n_mbon=args.n_mbon, n_dan=120, n_glomeruli=120)
    else:
        conn = load(args.data)
    heads = SPLIT_HEADS if getattr(args, "split", False) else HEADS
    return FlyAgent(
        conn, N_CHANNELS, heads, device=args.device, probe=probe,
        plast_cfg=PlasticityConfig(lr=args.lr, recovery=args.recovery, scaling=args.scaling),
        cfg=AgentConfig(action_groups="random", seed=args.seed, decision_steps=args.decision_ms,
                        mbon_rate_hz=args.mbon_rate, epsilon=0.0),
    )


def record(args, fly_deck):
    X, M, A, ACT = [], [], [], []
    rounds = max(1, args.record_matches // args.batch)
    for r in range(rounds):
        opp = ("basic", "random", "coach")[r % 3]
        env = RoyaleEnv(args.batch, OPPONENTS[opp], fly_deck=fly_deck, fly_deck_share=args.fly_deck_share,
                        seed=args.seed * 1000 + r)
        f, done = env.reset(), False
        while not done:
            m = env.masks()
            a, act = env.teacher()
            live = ~env.done
            X.append(f[live]); M.append(m[0][live]); A.append(a[live]); ACT.append(act[live])
            f, _, done = env.step(a)
        print(f"  recorded round {r + 1}/{rounds} vs {opp}: {sum(len(x) for x in X)} decisions", flush=True)
    return np.concatenate(X), np.concatenate(M), np.concatenate(A), np.concatenate(ACT)


def record_on_policy(agent, args, fly_deck, matches, seed, follow_coach=0.0):
    """DAgger: the brain plays (taking the coach's move with probability ``follow_coach``) and every
    situation it ends up in is labelled with what the coach would do there."""
    X, M, A, ACT = [], [], [], []
    rng = np.random.default_rng(seed)
    for r in range(max(1, matches // args.batch)):
        opp = ("basic", "random", "coach")[r % 3]
        env = RoyaleEnv(args.batch, OPPONENTS[opp], fly_deck=fly_deck, fly_deck_share=args.fly_deck_share,
                        seed=seed * 1000 + r, heads=agent.heads, guard=getattr(args, "dagger_guard", False))
        f, done = env.reset(), False
        while not done:
            m = env.masks()
            raw, active = env.teacher_raw()
            live = ~env.done
            role_masks = np.stack([role_mask(v) for v in env.views()])
            X.append(f[live]); M.append(role_masks[live]); A.append(raw[live]); ACT.append(active[live])
            a = agent.act(f, m, learn=False)
            follow = rng.random(args.batch) < follow_coach
            if follow.any():
                a[follow] = env.teacher()[0][follow]
            f, _, done = env.step(a)
        print(f"  on-policy round {r + 1}: {sum(len(x) for x in X)} decisions vs {opp}", flush=True)
    return np.concatenate(X), np.concatenate(M), np.concatenate(A), np.concatenate(ACT)


def teach_epoch(agent, X, M, A, ACT, idx, B, rng):
    perm = rng.permutation(idx)
    for s in range(0, len(perm) - B + 1, B):
        b = perm[s:s + B]
        agent.act(X[b], brain_masks_batch(agent, M[b]), learn=False)
        agent.teach(*encode(A[b], ACT[b], agent.heads))


def brain_masks_batch(agent, M):
    """Recorded role masks (wait + 15 roles) -> masks in the agent's head layout."""
    n = len(M)
    if tuple(agent.heads) == SPLIT_HEADS:
        roles = M[:, 1:]
        return [np.stack([M[:, 0], roles.any(1)], 1), np.where(roles.any(1)[:, None], roles, True),
                np.ones((n, 2), bool)]
    return [M, np.ones((n, 2), bool)]


def agreement(agent, X, M, A, n=2000):
    """How often the brain does what the coach did: waits when it waited, and the same role / lane when it played."""
    idx = np.arange(min(n, len(X)))
    out = []
    for s in range(0, len(idx), 250):
        b = idx[s:s + 250]
        out.append(agent.act(X[b], brain_masks_batch(agent, M[b]), learn=False))
    a = np.array([decode(x, agent.heads) for x in np.concatenate(out)])
    t = A[idx]
    plays = t[:, 0] > 0
    return dict(wait=float((a[~plays, 0] == 0).mean()), play=float((a[plays, 0] > 0).mean()),
                role=float((a[plays, 0] == t[plays, 0]).mean()), lane=float((a[plays, 1] == t[plays, 1]).mean()))


def evaluate(agent, opponent, matches, fly_deck, share, seed, guard=False):
    env = RoyaleEnv(matches, OPPONENTS[opponent], fly_deck=fly_deck, fly_deck_share=share, seed=seed, guard=guard,
                    heads=agent.heads)
    f, done = env.reset(), False
    while not done:
        f, _, done = env.step(agent.act(f, env.masks(), learn=False))
    r = env.results()
    return dict(win=float((r == 1).mean()), draw=float((r == 0).mean()), loss=float((r == -1).mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="synthetic")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fly-deck", default=str(ROOT / "decks/fly.txt"))
    ap.add_argument("--fly-deck-share", type=float, default=0.3, help="share of matches played with the fly deck")
    ap.add_argument("--record-matches", type=int, default=96)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--teach-batch", type=int, default=32)
    ap.add_argument("--eval-matches", type=int, default=32)
    ap.add_argument("--n-mbon", type=int, default=192)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--recovery", type=float, default=0.0)
    ap.add_argument("--no-scaling", dest="scaling", action="store_false")
    ap.add_argument("--decision-ms", type=int, default=100)
    ap.add_argument("--mbon-rate", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "models/fly_royale.pt"))
    ap.add_argument("--init", help="start from this trained brain instead of a fresh one")
    ap.add_argument("--dagger-rounds", type=int, default=0, help="rounds of learning from the brain's own games")
    ap.add_argument("--dagger-matches", type=int, default=64, help="matches recorded per DAgger round")
    ap.add_argument("--dagger-guard", action="store_true",
                    help="the brain plays its DAgger games with the coach guard on, like the real bot")
    ap.add_argument("--split", action="store_true", help="separate play?/role/lane heads (see env.SPLIT_HEADS)")
    ap.add_argument("--wait-keep", type=float, default=1.0, help="share of coach 'wait' decisions taught (balances plays)")
    ap.add_argument("--n-kc", type=int, default=2500)
    ap.add_argument("--dataset", help="save recorded decisions here (.npz), or reuse them if the file exists")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    fly_deck = load_deck(args.fly_deck) if args.fly_deck else None
    print("fly deck:", fly_deck)

    t0 = time.time()
    if args.dataset and Path(args.dataset).exists():
        print(f"1. reusing recorded decisions from {args.dataset}")
        d = np.load(args.dataset)
        X, M, A, ACT = d["X"], d["M"], d["A"], d["ACT"]
    else:
        print("1. recording coach decisions")
        X, M, A, ACT = record(args, fly_deck)
        if args.dataset:
            np.savez_compressed(args.dataset, X=X, M=M, A=A, ACT=ACT)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(X))
    test, train = idx[:2000], idx[2000:]
    print(f"   {len(X)} decisions ({(A[:, 0] > 0).mean():.0%} plays) in {time.time() - t0:.0f}s")

    if args.init:
        ckpt = torch.load(args.init, weights_only=False, map_location=args.device)
        for k in ("split", "n_mbon", "n_kc", "decision_ms", "mbon_rate", "data", "scaling", "seed"):
            if k in ckpt["args"]:
                setattr(args, k, ckpt["args"][k])
        agent = make_agent(args)
        agent.load_state_dict(ckpt["agent"])
        agent.cfg.epsilon = 0.0
        agent.rule.cfg.lr = args.lr
        print(f"   starting from {args.init}")
    else:
        agent = make_agent(args, probe=X[train[:64]])
    print("   calibration:", agent.calibration)
    log = [dict(stage="start", agreement=agreement(agent, X[test], M[test], A[test]))]
    print("   agreement before teaching:", log[-1]["agreement"])

    print("2. teaching")
    B = args.teach_batch
    for ep in range(args.epochs):
        keep = (A[train, 0] > 0) | (rng.random(len(train)) < args.wait_keep)
        perm = rng.permutation(train[keep])
        teach_epoch(agent, X, M, A, ACT, train[keep], B, rng)
        log.append(dict(stage=f"epoch {ep + 1}", agreement=agreement(agent, X[test], M[test], A[test])))
        print(f"   epoch {ep + 1}: agreement {log[-1]['agreement']}  ({time.time() - t0:.0f}s)", flush=True)

    for k in range(args.dagger_rounds):
        # The brain's own games reach situations the coach's games never show; label them with the coach.
        follow = max(0.0, 0.5 - 0.2 * k)
        print(f"   DAgger round {k + 1}/{args.dagger_rounds} (coach moves taken {follow:.0%} of the time)")
        Xn, Mn, An, ACTn = record_on_policy(agent, args, fly_deck, args.dagger_matches, args.seed + 100 + k, follow)
        start = len(X)
        X, M, A, ACT = (np.concatenate([X, Xn]), np.concatenate([M, Mn]), np.concatenate([A, An]),
                        np.concatenate([ACT, ACTn]))
        new = np.arange(start, len(X))
        old = rng.choice(train, size=min(len(train), len(new)), replace=False)
        train = np.concatenate([train, new])
        idx = np.concatenate([new, old])
        keep = (A[idx, 0] > 0) | (rng.random(len(idx)) < args.wait_keep)
        teach_epoch(agent, X, M, A, ACT, idx[keep], B, rng)
        log.append(dict(stage=f"dagger {k + 1}", agreement=agreement(agent, X[test], M[test], A[test]),
                        on_policy=agreement(agent, Xn, Mn, An)))
        print(f"   after round {k + 1}: coach-data agreement {log[-1]['agreement']}, "
              f"on its own games {log[-1]['on_policy']}  ({time.time() - t0:.0f}s)", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    saved_args = {k: v for k, v in vars(args).items() if k != "device"}
    torch.save(dict(agent=agent.state_dict(), args=saved_args, fly_deck=fly_deck), args.out)
    print(f"   saved {args.out}")

    print("3. testing")
    for opp in ("random", "basic"):
        for name, share in (("random decks", 0.0), ("fly deck", 1.0)):
            r = evaluate(agent, opp, args.eval_matches, fly_deck, share, seed=777)
            log.append(dict(stage="eval", opponent=opp, decks=name, **r))
            print(f"   vs {opp:6s} with {name:12s}: win {r['win']:.2f} draw {r['draw']:.2f} loss {r['loss']:.2f}",
                  flush=True)
    Path(args.out).with_suffix(".json").write_text(json.dumps(log, indent=1))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
