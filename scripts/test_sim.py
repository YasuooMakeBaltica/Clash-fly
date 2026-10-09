"""Milestone 2: run the spiking mushroom body on test input and print firing rates.

    python scripts/test_sim.py --data synthetic
    python scripts/test_sim.py --data data/mushroom_body.npz --device cuda
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", message="Sparse")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from flybrain.agent import AgentConfig, FlyAgent  # noqa: E402
from flybrain.connectome import POPULATIONS, load  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="synthetic")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--channels", type=int, default=16, help="number of input channels")
    ap.add_argument("--active", type=int, default=2, help="channels switched on per test pattern")
    ap.add_argument("--patterns", type=int, default=8)
    ap.add_argument("--steps", type=int, default=100, help="ms to simulate")
    args = ap.parse_args()

    conn = load(args.data)
    print(conn.summary(), "\n")
    t = time.time()
    agent = FlyAgent(conn, args.channels, device=args.device, cfg=AgentConfig(decision_steps=args.steps))
    print(f"calibration ({time.time() - t:.1f}s): {agent.calibration}\n")

    rng = np.random.default_rng(1)
    feats = np.zeros((args.patterns, args.channels), dtype=np.float32)
    for row in feats:
        row[rng.choice(args.channels, args.active, replace=False)] = 1
    i_ext = agent.encode(feats)
    net = agent.net
    net.reset(args.patterns)
    t = time.time()
    counts = net.run(i_ext, args.steps)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    ms = (time.time() - t) * 1000 / args.steps
    rates = net.rates_hz(counts, args.steps)

    print(f"{args.patterns} patterns x {args.steps} ms on {args.device}: {ms:.2f} ms wall per sim step")
    print(f"{'pop':5s} {'n':>6s} {'mean Hz':>8s} {'max Hz':>7s} {'% active':>9s}")
    for pop in POPULATIONS:
        s = net.sl[pop]
        if s.stop == s.start:
            continue
        r = rates[:, s]
        print(f"{pop:5s} {s.stop - s.start:6d} {r.mean():8.1f} {r.max():7.1f} {100 * (r > 0).float().mean():8.1f}%")

    kc = (counts[:, net.kc] > 0).float()
    overlap = (kc @ kc.T) / kc.sum(1).clamp_min(1)
    off = overlap[~torch.eye(args.patterns, dtype=torch.bool)].mean()
    print(f"\nKC code overlap between different patterns: {off:.2f} (low = good pattern separation)")
    print("MBON group votes per pattern (left, right):")
    for v in (counts[:, net.mbon] @ agent.groups.T / agent.groups.sum(1)).tolist():
        print(f"  {v[0]:5.2f} {v[1]:5.2f}")


if __name__ == "__main__":
    main()
