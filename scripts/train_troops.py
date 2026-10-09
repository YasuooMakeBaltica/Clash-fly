"""Train the troop detector (flybrain/real/troops.py) on real labelled battle frames.

Data: KataCR's Clash-Royale-Detection-Dataset (MIT licence), about 7,000
568x896 arena frames from real matches with every troop labelled:

    git clone --filter=blob:none --sparse https://github.com/wty-yy/Clash-Royale-Detection-Dataset.git
    cd Clash-Royale-Detection-Dataset && git sparse-checkout set "images/part2/*"

    python scripts/train_troops.py --data Clash-Royale-Detection-Dataset/images/part2 --out models/troops.pt

Whole recording sessions are held out for validation (--val), so the score
is measured on matches the net never saw.
"""

import argparse
import glob
import math
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from flybrain.real.troops import (CLASSES, FRAME_SIZE, INPUT_SIZE, SCALE, STRIDE, TroopNet, frame_to_tile,  # noqa: E402
                                  kata_class, net_input)

NOT_TROOPS = {
    "king-tower", "queen-tower", "cannoneer-tower", "dagger-duchess-tower", "dagger-duchess-tower-bar", "tower-bar",
    "king-tower-bar", "bar", "bar-level", "skeleton-king-bar", "clock", "emote", "text", "elixir", "selected",
    "evolution-symbol", "ice-spirit-evolution-symbol", "zap", "giant-snowball", "rage", "the-log", "arrows",
    "earthquake", "clone", "tornado", "fireball", "freeze", "poison", "graveyard", "lightning", "rocket",
    "barbarian-barrel", "royal-delivery", "goblin-barrel", "dirt", "axe", "bomb", "skeleton-king-skill",
    "tesla-evolution-shock", "goblin-ball", "zap-evolution", "mirror",
}
GH, GW = INPUT_SIZE[1] // STRIDE, INPUT_SIZE[0] // STRIDE


def class_names(data):
    out = {}
    for line in open(os.path.join(data, "ClashRoyale_detection.yaml")):
        m = re.match(r"\s+(\d+): (\S+)", line)
        if m:
            out[int(m.group(1))] = m.group(2)
    return out


def body_point(x0, y0, x1, y1):
    """Where a troop stands, from its sprite box (lower part of the sprite)."""
    return (x0 + x1) / 2, y0 + 0.7 * (y1 - y0)


def load_frame_labels(jpg, names):
    W, H = FRAME_SIZE
    out = []
    for line in open(jpg[:-4] + ".txt"):
        r = line.split()
        if len(r) < 6:
            continue
        name = names.get(int(r[0]), "")
        if name in NOT_TROOPS or name.startswith("pad_"):
            continue
        cx, cy, w, h = map(float, r[1:5])
        side = int(float(r[5]))
        u, v = body_point((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H)
        char = kata_class(name)
        out.append((side, u, v, w * W, h * H, CLASSES.index(char) if char else -1))
    return out


def build_cache(files, path):
    """All frames at the net's input size in one uint8 memmap (decoding JPEGs every epoch is slow)."""
    shape = (len(files), INPUT_SIZE[1], INPUT_SIZE[0], 3)
    if os.path.exists(path):
        return np.memmap(path, np.uint8, "r", shape=shape)
    mm = np.memmap(path + ".tmp", np.uint8, "w+", shape=shape)
    for i, f in enumerate(files):
        mm[i] = net_input(cv2.imread(f))
        if i % 1000 == 0:
            print(f"  cached {i}/{len(files)}", flush=True)
    mm.flush()
    del mm
    os.replace(path + ".tmp", path)
    return np.memmap(path, np.uint8, "r", shape=shape)


def targets(objs, flip):
    heat = np.zeros((2, GH, GW), np.float32)
    cls = np.full((GH, GW), -1, np.int64)
    off = np.zeros((2, GH, GW), np.float32)
    mask = np.zeros((GH, GW), np.float32)
    ys, xs = np.mgrid[0:GH, 0:GW]
    for side, u, v, w, h, k in objs:
        if flip:
            u = FRAME_SIZE[0] - u
        gx, gy = u * SCALE / STRIDE - 0.5, v * SCALE / STRIDE - 0.5
        ix, iy = int(round(gx)), int(round(gy))
        if not (0 <= ix < GW and 0 <= iy < GH):
            continue
        sigma = max(0.7, 0.12 * min(w, h) * SCALE / STRIDE)
        g = np.exp(-((xs - gx) ** 2 + (ys - gy) ** 2) / (2 * sigma ** 2))
        heat[side] = np.maximum(heat[side], g)
        heat[side, iy, ix] = 1.0
        if k >= 0:
            cls[iy, ix] = k
        off[:, iy, ix] = (gx - ix, gy - iy)
        mask[iy, ix] = 1.0
    return heat, cls, off, mask


def focal_loss(logits, target):
    """CenterNet's penalty-reduced focal loss."""
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = target.eq(1).float()
    neg = 1 - pos
    pos_loss = torch.log(p) * (1 - p) ** 2 * pos
    neg_loss = torch.log(1 - p) * p ** 2 * (1 - target) ** 4 * neg
    n = pos.sum().clamp_min(1)
    return -(pos_loss.sum() + neg_loss.sum()) / n


def augment(img, rng):
    img = img.astype(np.float32)
    img = img * rng.uniform(0.75, 1.25) + rng.uniform(-25, 25)
    grey = img.mean(2, keepdims=True)
    img = grey + (img - grey) * rng.uniform(0.7, 1.3)
    return np.clip(img, 0, 255).astype(np.uint8)


def batches(idx, cache, labels, bs, rng, train):
    order = rng.permutation(idx) if train else idx
    for s in range(0, len(order), bs):
        b = order[s:s + bs]
        imgs, ts = [], []
        for i in b:
            img = np.asarray(cache[i])
            flip = bool(train and rng.random() < 0.5)
            if flip:
                img = img[:, ::-1]
            if train:
                img = augment(img, rng)
            imgs.append(img)
            ts.append(targets(labels[i], flip))
        x = torch.from_numpy(np.ascontiguousarray(np.stack(imgs)[..., ::-1])).permute(0, 3, 1, 2).float() / 255.0
        yield b, x, [torch.from_numpy(np.stack(t)) for t in zip(*ts)]


@torch.no_grad()
def evaluate(net, idx, cache, labels, threshold=0.35, tol=1.0):
    """Side-aware detection recall/precision within ``tol`` tiles, and troop-type accuracy of matched troops."""
    net.eval()
    found = total = correct = good = dets = 0
    rng = np.random.default_rng(0)
    for b, x, _ in batches(idx, cache, labels, 16, rng, False):
        heat, cls, off = net(x)
        heat = torch.sigmoid(heat)
        keep = (heat == F.max_pool2d(heat, 3, 1, 1)) & (heat >= threshold)
        for j, i in enumerate(b):
            peaks = []
            for side, gy, gx in zip(*[t.numpy() for t in torch.nonzero(keep[j], as_tuple=True)]):
                u = (gx + 0.5 + off[j, 0, gy, gx].item()) * STRIDE / SCALE
                v = (gy + 0.5 + off[j, 1, gy, gx].item()) * STRIDE / SCALE
                peaks.append((int(side), *frame_to_tile(u, v), int(cls[j, :, gy, gx].argmax())))
            gts = [(s, *frame_to_tile(u, v), k) for s, u, v, w, h, k in labels[i]]
            dets += len(peaks)
            used = set()
            for s, tx, ty, k in gts:
                total += 1
                best = None
                for pi, (ps, px, py, pk) in enumerate(peaks):
                    d = math.hypot(px - tx, py - ty)
                    if ps == s and pi not in used and d <= tol and (best is None or d < best[0]):
                        best = (d, pi, pk)
                if best:
                    used.add(best[1])
                    found += 1
                    if k >= 0:
                        good += 1
                        correct += best[2] == k
    net.train()
    return dict(recall=found / max(total, 1), precision=found / max(dets, 1), type_acc=correct / max(good, 1),
                troops=total, detections=dets)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="the dataset's images/part2 folder")
    ap.add_argument("--val", default="WTY_20240309,OYASSU_20230203_episodes", help="recording sessions held out")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--warmup", type=float, default=0.1, help="share of steps spent warming up the learning rate")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="use only this many training frames (quick tests)")
    ap.add_argument("--cache", default="runs/troops_cache.u8")
    ap.add_argument("--init", help="continue from these weights")
    ap.add_argument("--out", default=str(ROOT / "models/troops.pt"))
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    names = class_names(args.data)
    files = sorted(f for f in glob.glob(os.path.join(args.data, "*", "*", "*.jpg")) if os.path.exists(f[:-4] + ".txt"))
    labels = [load_frame_labels(f, names) for f in files]
    session = [Path(f).parts[-3] for f in files]
    val_sessions = set(args.val.split(","))
    val = np.array([i for i, s in enumerate(session) if s in val_sessions])
    train = np.array([i for i, s in enumerate(session) if s not in val_sessions])
    if args.limit:
        train = rng.choice(train, min(args.limit, len(train)), replace=False)
        val = val[: max(64, args.limit // 8)]
    print(f"{len(files)} frames: {len(train)} train, {len(val)} validation ({', '.join(sorted(val_sessions))}); "
          f"{sum(len(l) for l in labels)} troops, {len(CLASSES)} troop types", flush=True)
    Path(args.cache).parent.mkdir(parents=True, exist_ok=True)
    cache = build_cache(files, args.cache)

    net = TroopNet(len(CLASSES), args.width)
    if args.init:
        net.load_state_dict({k: v.float() if v.is_floating_point() else v
                             for k, v in torch.load(args.init, weights_only=False)["model"].items()})
    print(f"model: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M parameters", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * math.ceil(len(train) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=args.warmup)
    best, t0 = -1.0, time.time()
    for ep in range(args.epochs):
        tot = np.zeros(3)
        n = 0
        for _, x, (heat, cls, off, mask) in batches(train, cache, labels, args.batch, rng, True):
            ph, pc, po = net(x)
            l_heat = focal_loss(ph, heat)
            l_cls = F.cross_entropy(pc, cls, ignore_index=-1) if (cls >= 0).any() else ph.sum() * 0
            m = mask[:, None]
            l_off = (F.l1_loss(po, off, reduction="none") * m).sum() / m.sum().clamp_min(1)
            loss = l_heat + l_cls + l_off
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += (l_heat.item(), l_cls.item(), l_off.item())
            n += 1
            if n % 100 == 0:
                print(f"  epoch {ep + 1} step {n}: heat {tot[0] / n:.3f} type {tot[1] / n:.3f} offset {tot[2] / n:.3f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        r = evaluate(net, val, cache, labels)
        score = 2 * r["recall"] * r["precision"] / max(r["recall"] + r["precision"], 1e-6)
        print(f"epoch {ep + 1}: loss heat {tot[0] / n:.3f} type {tot[1] / n:.3f}; validation recall {r['recall']:.3f} "
              f"precision {r['precision']:.3f} type accuracy {r['type_acc']:.3f} ({r['troops']} troops) "
              f"({time.time() - t0:.0f}s)", flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        ckpt = dict(model={k: v.half() if v.is_floating_point() else v for k, v in net.state_dict().items()},
                    classes=CLASSES, width=args.width, val=r, epoch=ep + 1, ground=0.0)
        torch.save(ckpt, str(args.out).replace(".pt", "_last.pt"))      # resumable even if the run is cut short
        if score > best:
            best = score
            torch.save(ckpt, args.out)
            print(f"  saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
