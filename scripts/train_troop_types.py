"""Train the troop-type classifier (troops.TypeNet) on full-resolution close-ups of labelled troops.

Same data as scripts/train_troops.py (KataCR's MIT-licensed dataset). Each
labelled troop gives one close-up around its point; whole recording sessions
are held out for validation. With --sprites (the dataset's images/segment
folder: cut-out troop sprites with transparency) half of every training batch
is synthetic: a sprite of an evenly drawn troop type pasted onto a random
patch of a real arena, which covers troop types the real frames rarely show. Besides exact type accuracy it reports how
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


def load_sprites(folder):
    """{class index: [BGRA sprite]} from images/segment/<dataset class>/<class>_<side>_<id>.png."""
    from flybrain.real.troops import kata_class
    out = {}
    for d in sorted(Path(folder).iterdir()):
        char = kata_class(d.name)
        if not d.is_dir() or not char:
            continue
        imgs = [cv2.imread(str(f), cv2.IMREAD_UNCHANGED) for f in sorted(d.glob("*.png"))]
        imgs = [im for im in imgs if im is not None and im.ndim == 3 and im.shape[2] == 4]
        if imgs:
            out.setdefault(CLASSES.index(char), []).extend(imgs)
    return out


def synthetic(sprites, backgrounds, n, rng):
    """n close-ups (out x out) with a pasted sprite at the troop point, and their labels."""
    out = int(64 * PAD)
    win = int(TypeNet.CROP * PAD)
    keys = sorted(sprites)
    xs, ys = np.zeros((n, out, out, 3), np.uint8), np.zeros(n, np.int64)
    for i in range(n):
        k = keys[rng.integers(len(keys))]
        spr = sprites[k][rng.integers(len(sprites[k]))]
        if rng.random() < 0.5:
            spr = spr[:, ::-1]
        sc = rng.uniform(0.9, 1.1)
        spr = cv2.resize(spr, (max(2, int(spr.shape[1] * sc)), max(2, int(spr.shape[0] * sc))))
        bg = backgrounds[rng.integers(len(backgrounds))]
        H, W = bg.shape[:2]
        cx, cy = rng.integers(win // 2, W - win // 2), rng.integers(win // 2, H - win // 2)
        patch = bg[cy - win // 2:cy + win // 2, cx - win // 2:cx + win // 2].astype(np.float32).copy()
        h, w = spr.shape[:2]
        # the troop point (bbox centre x, 70% down) sits at the window centre, like real close-ups
        x0, y0 = win // 2 - w // 2, int(win // 2 - 0.7 * h)
        xa, ya, xb, yb = max(0, x0), max(0, y0), min(win, x0 + w), min(win, y0 + h)
        if xb > xa and yb > ya:
            s_ = spr[ya - y0:yb - y0, xa - x0:xb - x0].astype(np.float32)
            a = s_[:, :, 3:4] / 255.0
            patch[ya:yb, xa:xb] = patch[ya:yb, xa:xb] * (1 - a) + s_[:, :, :3] * a
        xs[i] = cv2.resize(patch.astype(np.uint8), (out, out), interpolation=cv2.INTER_AREA)
        ys[i] = k
    return xs, ys


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
    ap.add_argument("--sprites", help="the dataset's images/segment folder (synthetic close-ups of every troop type)")
    ap.add_argument("--all", action="store_true", help="also train on the validation sessions (final model)")
    ap.add_argument("--init", help="continue from these weights")
    ap.add_argument("--negatives", help="'not a troop' close-ups (scripts/mine_troop_negatives.py): adds a verifier class")
    ap.add_argument("--val-negatives", help="'not a troop' close-ups from the validation sessions, for scoring")
    ap.add_argument("--neg-share", type=float, default=0.25, help="share of each batch that is 'not a troop'")
    ap.add_argument("--warmup", type=float, default=0.1)
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
    if args.all:
        tr_idx = np.arange(len(y))
    sprites, backgrounds = {}, []
    if args.sprites:
        sprites = load_sprites(args.sprites)
        train_files = [f for f in files if Path(f).parts[-3] not in val_sessions or args.all]
        for f in rng.choice(train_files, min(150, len(train_files)), replace=False):
            backgrounds.append(cv2.imread(str(f)))
        print(f"sprites for {len(sprites)} troop types ({sum(len(v) for v in sprites.values())} sprites), "
              f"{len(backgrounds)} backgrounds", flush=True)
    counts = np.bincount(y[tr_idx], minlength=len(CLASSES))
    print(f"{len(y)} troop close-ups: {len(tr_idx)} train, {len(va_idx)} validation; "
          f"{(counts > 0).sum()} of {len(CLASSES)} types seen in training", flush=True)
    weight = torch.tensor(1.0 / np.sqrt(np.maximum(counts, 1)), dtype=torch.float32)
    weight = weight / weight[torch.from_numpy(counts > 0)].mean()
    props = properties()

    neg = np.load(args.negatives)["x"] if args.negatives else None
    vneg = np.load(args.val_negatives)["x"] if args.val_negatives else None
    K = len(CLASSES)
    net = TypeNet(K + (neg is not None))
    if args.init:
        state = {k: v.float() if v.is_floating_point() else v
                 for k, v in torch.load(args.init, weights_only=False)["model"].items()}
        if neg is not None and state["fc.weight"].shape[0] == K:     # add the "not a troop" output
            state["fc.weight"] = torch.cat([state["fc.weight"], torch.zeros(1, state["fc.weight"].shape[1])])
            state["fc.bias"] = torch.cat([state["fc.bias"], torch.zeros(1)])
        net.load_state_dict(state)
    if neg is not None:
        weight = torch.cat([weight, torch.ones(1)])
        print(f"{len(neg)} 'not a troop' close-ups for the verifier class", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=5e-4)
    steps = args.epochs * math.ceil(len(tr_idx) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=args.warmup)
    out = int(64 * PAD)
    best, t0 = -1.0, time.time()
    for ep in range(args.epochs):
        net.train()
        perm = rng.permutation(tr_idx)
        tot, n = 0.0, 0
        for s in range(0, len(perm), args.batch):
            b = perm[s:s + args.batch]
            xr, yr = x[b], y[b]
            if neg is not None:
                nb = int(len(b) * args.neg_share / (1 - args.neg_share))
                xr = np.concatenate([xr, neg[rng.integers(0, len(neg), nb)]])
                yr = np.concatenate([yr, np.full(nb, K)])
            if sprites:
                xs_, ys_ = synthetic(sprites, backgrounds, len(b), rng)
                xr, yr = np.concatenate([xr, xs_]), np.concatenate([yr, ys_])
            ox, oy = rng.integers(0, out - 64 + 1, 2)
            xb = xr[:, oy:oy + 64, ox:ox + 64].astype(np.float32)
            if rng.random() < 0.5:
                xb = xb[:, :, ::-1]
            xb = np.clip(xb * rng.uniform(0.75, 1.25) + rng.uniform(-25, 25), 0, 255).astype(np.uint8)
            logits = net(batch_tensor(xb))
            loss = F.cross_entropy(logits, torch.from_numpy(yr), weight=weight, label_smoothing=0.05)
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
                preds.append(net(batch_tensor(x[b, c:c + 64, c:c + 64])).numpy())
        logits_all = np.concatenate(preds)
        t = y[va_idx]
        p = logits_all[:, :K].argmax(1)
        acc = (p == t).mean()
        prop = (props[p] == props[t]).all(1).mean()
        extra = ""
        if neg is not None:
            soft = torch.softmax(torch.from_numpy(logits_all), 1).numpy()
            kept = (soft[:, K] < 0.5).mean()
            extra = f", real troops kept {kept:.3f}"
            if vneg is not None:
                with torch.no_grad():
                    vs = torch.softmax(net(batch_tensor(vneg[:, c:c + 64, c:c + 64])), 1).numpy()
                extra += f", false alarms rejected {(vs[:, K] >= 0.5).mean():.3f}"
        print(f"epoch {ep + 1}: loss {tot / n:.3f}; validation type accuracy {acc:.3f}, coach properties right "
              f"{prop:.3f}{extra} ({len(t)} troops) ({time.time() - t0:.0f}s)", flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        ckpt = dict(model={k: v.half() if v.is_floating_point() else v for k, v in net.state_dict().items()},
                    classes=CLASSES, crop=TypeNet.CROP, val_acc=float(acc), val_props=float(prop), epoch=ep + 1,
                    none=neg is not None)
        torch.save(ckpt, str(args.out).replace(".pt", "_last.pt"))
        if acc > best:
            best = acc
            torch.save(ckpt, args.out)


if __name__ == "__main__":
    main()
