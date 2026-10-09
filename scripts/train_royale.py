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
from flybrain.envs.royale.env import HEADS, N_CHANNELS, RoyaleEnv  # noqa: E402
from flybrain.envs.royale.strategy import BasicBot, Coach, RandomBot  # noqa: E402
from flybrain.plasticity import PlasticityConfig  # noqa: E402

OPPONENTS = {"random": lambda s: RandomBot(s), "basic": lambda s: BasicBot(s), "coach": lambda s: Coach()}


def make_agent(args, probe=None):
    if args.data == "synthetic":
        conn = load("synthetic", n_pn=680, n_kc=2500, n_mbon=args.n_mbon, n_dan=120, n_glomeruli=120)
    else:
        conn = load(args.data)
    return FlyAgent(
        conn, N_CHANNELS, HEADS, device=args.device, probe=probe,
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


def agreement(agent, X, M, A, n=2000):
    idx = np.arange(min(n, len(X)))
    out = []
    for s in range(0, len(idx), 250):
        b = idx[s:s + 250]
        out.append(agent.act(X[b], [M[b], np.ones((len(b), 2), bool)], learn=False))
    a = np.concatenate(out)
    t = A[idx]
    plays = t[:, 0] > 0
    return dict(wait=float((a[~plays, 0] == 0).mean()), role=float((a[plays, 0] == t[plays, 0]).mean()),
                lane=float((a[plays, 1] == t[plays, 1]).mean()))


def evaluate(agent, opponent, matches, fly_deck, share, seed):
    env = RoyaleEnv(matches, OPPONENTS[opponent], fly_deck=fly_deck, fly_deck_share=share, seed=seed)
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

    agent = make_agent(args, probe=X[train[:64]])
    print("   calibration:", agent.calibration)
    log = [dict(stage="start", agreement=agreement(agent, X[test], M[test], A[test]))]
    print("   agreement before teaching:", log[-1]["agreement"])

    print("2. teaching")
    B = args.teach_batch
    for ep in range(args.epochs):
        perm = rng.permutation(train)
        for s in range(0, len(perm) - B + 1, B):
            b = perm[s:s + B]
            agent.act(X[b], [M[b], np.ones((B, 2), bool)], learn=False)
            agent.teach(A[b], ACT[b])
        log.append(dict(stage=f"epoch {ep + 1}", agreement=agreement(agent, X[test], M[test], A[test])))
        print(f"   epoch {ep + 1}: agreement {log[-1]['agreement']}  ({time.time() - t0:.0f}s)", flush=True)

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
