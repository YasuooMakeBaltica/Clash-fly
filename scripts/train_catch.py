"""Milestones 3-4: train the fly brain on catch-the-ball with dopamine plasticity.

    python scripts/train_catch.py                                  # synthetic, CPU
    python scripts/train_catch.py --data data/mushroom_body.npz --device cuda --batch 64

Writes a CSV learning curve (and a PNG if matplotlib is installed) to --out.
"""

import argparse
import csv
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
from flybrain.envs import CatchEnv  # noqa: E402
from flybrain.plasticity import PlasticityConfig  # noqa: E402


def run_episodes(env, agent, batch, learn=True):
    f = env.reset(batch)
    agent.begin_episode(batch)
    done = False
    while not done:
        f, r, done = env.step(agent.act(f, learn=learn))
        if learn:
            agent.reward(r)
    return env.caught()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="synthetic")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--episodes", type=int, default=1600, help="total games")
    ap.add_argument("--batch", type=int, default=16, help="games played in parallel")
    ap.add_argument("--mode", default="depression", choices=["depression", "bidirectional"])
    ap.add_argument("--eligibility", default="pre_action", choices=["pre_action", "pre_post"])
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--recovery", type=float, default=0.02)
    ap.add_argument("--trace-decay", type=float, default=0.8)
    ap.add_argument("--encoding", default="absolute", choices=["absolute", "relative"])
    ap.add_argument("--width", type=int, default=8)
    ap.add_argument("--shaping", type=float, default=0.0,
                    help="per-step reward for moving toward the ball (0 = only catch/miss at the end)")
    ap.add_argument("--groups", default="side", choices=["side", "random"])
    ap.add_argument("--decision-ms", type=int, default=40, help="simulated ms per decision")
    ap.add_argument("--mbon-rate", type=float, default=20.0, help="calibration target MBON rate (Hz)")
    ap.add_argument("--epsilon", type=float, default=0.05)
    ap.add_argument("--eval-every", type=int, default=10, help="batches between evaluations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/catch")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    env = CatchEnv(width=args.width, height=args.width + 1, encoding=args.encoding,
                   shaping=args.shaping, seed=args.seed)
    eval_env = CatchEnv(width=args.width, height=args.width + 1, encoding=args.encoding, seed=args.seed + 1)
    agent = FlyAgent(
        load(args.data), env.n_channels, env.n_actions, device=args.device,
        plast_cfg=PlasticityConfig(mode=args.mode, lr=args.lr, recovery=args.recovery,
                                   trace_decay=args.trace_decay, eligibility=args.eligibility),
        cfg=AgentConfig(action_groups=args.groups, seed=args.seed, decision_steps=args.decision_ms,
                        mbon_rate_hz=args.mbon_rate, epsilon=args.epsilon),
    )
    print("calibration:", agent.calibration)

    def evaluate():
        eps = agent.cfg.epsilon
        agent.cfg.epsilon = 0.0
        rate = run_episodes(eval_env, agent, 256, learn=False).mean()
        agent.cfg.epsilon = eps
        return float(rate)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [(0, evaluate(), float("nan"))]
    print(f"episodes {0:6d}  eval catch {rows[0][1]:.3f}  (random policy ~0.34)")
    n_batches = args.episodes // args.batch
    recent, t = [], time.time()
    for i in range(1, n_batches + 1):
        recent.append(run_episodes(env, agent, args.batch).mean())
        if i % args.eval_every == 0 or i == n_batches:
            ev, tr = evaluate(), float(np.mean(recent))
            recent = []
            rows.append((i * args.batch, ev, tr))
            w = agent.net.W_kc_mbon.sum() / agent.net.W_kc_mbon_init.sum()
            print(f"episodes {i * args.batch:6d}  eval catch {ev:.3f}  train catch {tr:.3f}  "
                  f"w/w0 {w:.2f}  {time.time() - t:.0f}s", flush=True)

    with open(out / "learning_curve.csv", "w", newline="") as fh:
        csv.writer(fh).writerows([("episodes", "eval_catch_rate", "train_catch_rate"), *rows])
    torch.save({"W_kc_mbon": agent.net.W_kc_mbon.cpu(), "args": vars(args)}, out / "weights.pt")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x, ev, tr = zip(*rows)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(x, ev, marker="o", label="eval (greedy)")
        ax.plot(x, tr, alpha=0.6, label="train")
        ax.axhline(0.34, color="gray", ls="--", label="random policy")
        ax.set(xlabel="episodes", ylabel="catch rate", ylim=(0, 1), title=f"Fly brain, {args.mode} rule")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "learning_curve.png", dpi=120)
    except ImportError:
        pass
    print(f"wrote {out}/")


if __name__ == "__main__":
    main()
