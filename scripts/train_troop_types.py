"""Train the troop-type classifier (troops.TypeNet) on full-resolution close-ups of labelled troops.

Same data as scripts/train_troops.py (KataCR's MIT-licensed dataset). Each
labelled troop gives one close-up around its point; whole recording sessions
are held out for validation. Besides exact type accuracy it reports how
often the properties the coach cares about are right (flying, targets
buildings only, tank, ranged).

    python scripts/train_troop_types.py --data Clash-Royale-Detection-Dataset/images/part2 --out models/troop_types.pt
"""

import argparse
import glob
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from flybrain.envs.royale.db import load  # noqa: E402
from flybrain.real.troops import CLASSES, TypeNet, frame_crops  # noqa: E402
from train_troops import class_names, load_frame_labels  # noqa: E402

PAD = 1.25          # crops are cut 25% larger, then randomly re-cropped (position jitter)


def extract(files, labels, cache):
    if os.path.exists(cache):
        d = np.load(cache)
        return d["x"], d["y"], d["session"]
    out = int(64 * PAD)
    xs, ys, ss = [], [], []
    for i, f in enumerate(files):
        objs = [(u, v, k) for s, u, v, w, h, k in labels[i] if k >= 0]
        if not objs:
            continue
        frame = cv2.imread(f)
        xs.append(frame_crops(frame, [(u, v) for u, v, _ in objs], int(TypeNet.CROP * PAD), out))
        ys += [k for _, _, k in objs]
        ss += [Path(f).parts[-3]] * len(objs)
        if i % 1000 == 0:
            print(f"  crops from {i}/{len(files)} frames", flush=True)
    x, y, s = np.concatenate(xs), np.array(ys), np.array(ss)
    np.savez(cache, x=x, y=y, session=s)
    return x, y, s


def properties():
    db = load()
    props = []
    for c in CLASSES:
        s = db.characters[c]
        props.append((s.flying, s.buildings_only, s.hp >= 2000, s.range >= 4.0))
    return np.array(props)


def batch_tensor(x):
    return torch.from_numpy(np.ascontiguousarray(x[..., ::-1])).permute(0, 3, 1, 2).float() / 255.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", default="WTY_20240309,OYASSU_20230203_episodes")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--cache", default="runs/troop_crops.npz")
    ap.add_argument("--out", default=str(ROOT / "models/troop_types.pt"))
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    names = class_names(args.data)
    files = sorted(f for f in glob.glob(os.path.join(args.data, "*", "*", "*.jpg")) if os.path.exists(f[:-4] + ".txt"))
    labels = [load_frame_labels(f, names) for f in files]
    Path(args.cache).parent.mkdir(parents=True, exist_ok=True)
    x, y, sess = extract(files, labels, args.cache)
    val_sessions = set(args.val.split(","))
    va = np.array([s in val_sessions for s in sess])
    tr_idx, va_idx = np.flatnonzero(~va), np.flatnonzero(va)
    counts = np.bincount(y[tr_idx], minlength=len(CLASSES))
    print(f"{len(y)} troop close-ups: {len(tr_idx)} train, {len(va_idx)} validation; "
          f"{(counts > 0).sum()} of {len(CLASSES)} types seen in training", flush=True)
    weight = torch.tensor(1.0 / np.sqrt(np.maximum(counts, 1)), dtype=torch.float32)
    weight = weight / weight[torch.from_numpy(counts > 0)].mean()
    props = properties()

    net = TypeNet(len(CLASSES))
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=5e-4)
    steps = args.epochs * math.ceil(len(tr_idx) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    out = int(64 * PAD)
    best, t0 = -1.0, time.time()
    for ep in range(args.epochs):
        net.train()
        perm = rng.permutation(tr_idx)
        tot, n = 0.0, 0
        for s in range(0, len(perm), args.batch):
            b = perm[s:s + args.batch]
            ox, oy = rng.integers(0, out - 64 + 1, 2)
            xb = x[b, oy:oy + 64, ox:ox + 64].astype(np.float32)
            if rng.random() < 0.5:
                xb = xb[:, :, ::-1]
            xb = np.clip(xb * rng.uniform(0.75, 1.25) + rng.uniform(-25, 25), 0, 255).astype(np.uint8)
            logits = net(batch_tensor(xb))
            loss = F.cross_entropy(logits, torch.from_numpy(y[b]), weight=weight, label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            n += 1
        net.eval()
        preds = []
        with torch.no_grad():
            c = (out - 64) // 2
            for s in range(0, len(va_idx), 256):
                b = va_idx[s:s + 256]
                preds.append(net(batch_tensor(x[b, c:c + 64, c:c + 64])).argmax(1).numpy())
        p = np.concatenate(preds)
        t = y[va_idx]
        acc = (p == t).mean()
        prop = (props[p] == props[t]).all(1).mean()
        print(f"epoch {ep + 1}: loss {tot / n:.3f}; validation type accuracy {acc:.3f}, coach properties right "
              f"{prop:.3f} ({len(t)} troops) ({time.time() - t0:.0f}s)", flush=True)
        if acc > best:
            best = acc
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(model={k: v.half() if v.is_floating_point() else v for k, v in net.state_dict().items()},
                            classes=CLASSES, crop=TypeNet.CROP, val_acc=float(acc), val_props=float(prop)), args.out)


if __name__ == "__main__":
    main()
