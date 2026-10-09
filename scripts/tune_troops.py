"""Pick the troop detector's confidence threshold on held-out matches and store it with the weights.

    python scripts/tune_troops.py --data <dataset>/images/part2 --weights models/troops.pt

Scores each threshold by F1 (the balance of troops found and detections that
are real) on the validation sessions. With --precision-weight above 1,
precision counts more, since a phantom troop makes the coach defend against
nothing.
"""

import argparse
import glob
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from flybrain.real.troops import CLASSES, TroopNet  # noqa: E402
from train_troops import build_cache, class_names, evaluate, load_frame_labels  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", default=str(ROOT / "models/troops.pt"))
    ap.add_argument("--val", default="WTY_20240309,OYASSU_20230203_episodes")
    ap.add_argument("--cache", default="runs/troops_cache.u8")
    ap.add_argument("--precision-weight", type=float, default=1.0, help="beta < 1 in F-beta terms: >1 favours precision")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    names = class_names(args.data)
    files = sorted(f for f in glob.glob(os.path.join(args.data, "*", "*", "*.jpg")) if os.path.exists(f[:-4] + ".txt"))
    labels = [load_frame_labels(f, names) for f in files]
    val = np.array([i for i, f in enumerate(files) if Path(f).parts[-3] in set(args.val.split(","))])
    cache = build_cache(files, args.cache)
    ckpt = torch.load(args.weights, weights_only=False)
    net = TroopNet(len(CLASSES), ckpt.get("width", 32))
    net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ckpt["model"].items()})
    beta2 = 1.0 / args.precision_weight ** 2
    best = None
    for thr in (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55):
        r = evaluate(net, val, cache, labels, threshold=thr)
        p, rc = r["precision"], r["recall"]
        f = (1 + beta2) * p * rc / max(beta2 * p + rc, 1e-9)
        print(f"threshold {thr:.2f}: recall {rc:.3f} precision {p:.3f} score {f:.3f}", flush=True)
        if best is None or f > best[0]:
            best = (f, thr, r)
    ckpt["threshold"], ckpt["val_at_threshold"] = best[1], best[2]
    torch.save(ckpt, args.weights)
    print(f"stored threshold {best[1]} in {args.weights}")


if __name__ == "__main__":
    main()
