"""Collect "not a troop" close-ups for the type classifier's verifier class.

Runs the troop detector over the training frames and keeps the close-ups of
detections with no labelled troop within 1.5 tiles: the detector's own false
alarms (spell effects, tower parts, UI ...), the hardest negatives. Random
background patches away from any troop are added.

    python scripts/mine_troop_negatives.py --data <dataset>/images/part2 --weights models/troops.pt
"""

import argparse
import glob
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from flybrain.real.troops import TroopDetector, TypeNet, frame_crops, frame_to_tile, net_input  # noqa: E402
from train_troops import class_names, load_frame_labels  # noqa: E402
from train_troop_types import PAD  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", default=str(ROOT / "models/troops.pt"))
    ap.add_argument("--val", default="WTY_20240309,OYASSU_20230203_episodes")
    ap.add_argument("--every", type=int, default=2, help="use every n-th training frame")
    ap.add_argument("--from-val", action="store_true", help="mine the validation sessions instead (for scoring)")
    ap.add_argument("--random", type=int, default=1, help="random background close-ups per frame")
    ap.add_argument("--threshold", type=float, default=0.25, help="detector threshold (lower = more candidates)")
    ap.add_argument("--out", default="runs/troop_negatives.npz")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    rng = np.random.default_rng(0)
    names = class_names(args.data)
    val = set(args.val.split(","))
    files = sorted(f for f in glob.glob(os.path.join(args.data, "*", "*", "*.jpg"))
                   if os.path.exists(f[:-4] + ".txt") and (Path(f).parts[-3] in val) == args.from_val)[::args.every]
    det = TroopDetector(args.weights, threshold=args.threshold, types="none")
    out, size = [], int(64 * PAD)
    hard = 0
    for i, f in enumerate(files):
        frame = cv2.imread(f)
        gts = [(u, v) for s, u, v, w, h, k in load_frame_labels(f, names)]
        gt_tiles = [frame_to_tile(u, v) for u, v in gts]
        heat, cls, off = det.raw(net_input(frame))
        pts = []
        for side, u, v, score, k, p in det.peaks(heat, cls, off):
            tx, ty = frame_to_tile(u, v)
            if all(math.hypot(tx - gx, ty - gy) > 1.5 for gx, gy in gt_tiles):
                pts.append((u, v))
        hard += len(pts)
        for _ in range(args.random):
            for _try in range(10):
                u, v = rng.uniform(40, 528), rng.uniform(60, 860)
                tx, ty = frame_to_tile(u, v)
                if all(math.hypot(tx - gx, ty - gy) > 2.0 for gx, gy in gt_tiles):
                    pts.append((u, v))
                    break
        if pts:
            out.append(frame_crops(frame, pts, int(TypeNet.CROP * PAD), size))
        if i % 500 == 0:
            print(f"  {i}/{len(files)} frames, {hard} false alarms so far", flush=True)
    x = np.concatenate(out)
    np.savez(args.out, x=x)
    print(f"{len(x)} negative close-ups ({hard} detector false alarms) -> {args.out}")


if __name__ == "__main__":
    main()
