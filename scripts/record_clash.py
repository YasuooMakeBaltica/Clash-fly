"""Record one match of the fly brain for the replay viewer.

    python scripts/record_clash.py --brain runs/clash/brain.pt --opponent basic --out replay.json

Saves arena snapshots every 0.2 s, every card played, and at each decision
what the brain saw (active situation channels), how its card and lane
groups voted, what it did, and what the coach would have done.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", message="Sparse")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from flybrain.envs.clash.cards import DECK, LANE_X, MATCH_TIME  # noqa: E402
from flybrain.envs.clash.env import CHANNELS, ClashEnv  # noqa: E402
from train_clash import OPPONENTS, make_agent, probe_states  # noqa: E402


def record(agent, opponent: str, seed: int) -> dict:
    eps, agent.cfg.epsilon = agent.cfg.epsilon, 0.0
    env = ClashEnv(1, OPPONENTS[opponent], seed=seed)
    f = env.reset()
    sim = env.sims[0]
    frames = [sim.snapshot()]
    decisions = []

    def on_tick(i, s):
        if round(s.time * 10) % 2 == 0 or s.done:
            frames.append(s.snapshot())

    done = False
    while not done:
        masks = env.masks()
        a = agent.act(f, masks, learn=False)[0]
        coach, _ = env.teacher()
        votes = [[round(float(x), 3) for x in v[0].clamp_min(0).cpu()] for v in agent.last["votes"]]
        decisions.append(dict(
            t=round(sim.time, 2), card=int(a[0]), lane=int(a[1]),
            coach_card=int(coach[0, 0]), coach_lane=int(coach[0, 1]),
            card_votes=votes[0], lane_votes=votes[1],
            allowed=masks[0][0].astype(int).tolist(),
            sees=[CHANNELS[j] for j in np.flatnonzero(f[0])],
            kc_active=round(float((agent.last["kc"][0] > 0).float().mean()), 3),
        ))
        f, _, done = env.step(a[None], on_tick=on_tick)
    agent.cfg.epsilon = eps
    return dict(
        opponent=opponent, winner=sim.winner, kc_total=agent.net.kc.stop - agent.net.kc.start, crowns=[p.crowns for p in sim.players],
        match_time=MATCH_TIME, lanes=list(LANE_X),
        deck=[dict(name=c.name, cost=c.cost, kind=c.kind, role=c.role, count=c.count) for c in DECK],
        events=sim.events, frames=frames, decisions=decisions,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--brain", help="checkpoint from train_clash.py (omit for an untrained brain)")
    ap.add_argument("--data", default="synthetic")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--opponent", default="basic", choices=list(OPPONENTS))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="runs/clash/replay.json")
    args = ap.parse_args()

    ckpt = torch.load(args.brain, weights_only=False) if args.brain else None
    train_args = argparse.Namespace(**(ckpt["args"] if ckpt else {}))
    for k, v in dict(data=args.data, device=args.device, mode="depression", lr=0.02, recovery=0.01,
                     trace_decay=0.7, seed=0, decision_ms=40, epsilon=0.0, rpe_rate=0.0).items():
        if not hasattr(train_args, k):
            setattr(train_args, k, v)
    train_args.device = args.device
    agent = make_agent(train_args, probe=None if ckpt else probe_states())
    if ckpt:
        agent.load_state_dict(ckpt["agent"])
    rec = record(agent, args.opponent, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rec, separators=(",", ":")))
    result = {0: "brain wins", 1: "brain loses", None: "draw"}[rec["winner"]]
    print(f"{result} {rec['crowns'][0]}-{rec['crowns'][1]} vs {args.opponent}; "
          f"{len(rec['frames'])} frames, {sum(e['kind'] == 'play' and e['player'] == 0 for e in rec['events'])} "
          f"cards played by the brain -> {args.out}")


if __name__ == "__main__":
    main()
