"""Calibrate the screen reader to your LDPlayer screen. Run these on your PC.

1. Take a screenshot during a battle:
       python -m flybrain.real.calibrate screenshot --adb "C:\\LDPlayer\\LDPlayer9\\adb.exe" --out shot.png
2. Check the estimated layout (writes shot_check.png with boxes and readings):
       python -m flybrain.real.calibrate check --image shot.png
3. If boxes are off, drag them yourself (opens a window per region; Enter to confirm):
       python -m flybrain.real.calibrate pick --image shot.png
4. Usually not needed: cards are recognised from the official card art. Only if
   `check` shows "?" for a card, save its picture from your screen (4 hand cards, left to right):
       python -m flybrain.real.calibrate templates --image shot.png --cards "Knight,Archers,Giant,Musketeer" --next Goblins
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from ..envs.royale.decks import resolve
from ..envs.royale.sim import LANE_X, WIDTH
from .adb import Adb
from .bot import draw_overlay
from .cards import OFFICIAL_DIR, download_official
from .layout import Box, Layout
from .perception import Perception


def cmd_screenshot(a):
    adb = Adb(a.adb, a.serial)
    if a.connect:
        print(adb.connect(a.connect))
    img = adb.screencap()
    cv2.imwrite(a.out, img)
    print(f"saved {a.out} ({img.shape[1]}x{img.shape[0]})")


def _read(path):
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"can't open {path}. Take one first: python -m flybrain.real.calibrate screenshot --out {path}")
    return img


def cmd_check(a):
    img = _read(a.image)
    lay = Layout.load(a.layout)
    if not any(OFFICIAL_DIR.glob("*.png")):
        print("downloading the official card pictures (once)...")
        download_official()
    obs = Perception(lay, a.templates, official_dir=OFFICIAL_DIR).read(img)
    out = Path(a.image).with_name(Path(a.image).stem + "_check.png")
    cv2.imwrite(str(out), draw_overlay(img, lay, obs))
    print(f"elixir {obs.elixir}  hand {obs.hand}  next {obs.next_card}")
    print("towers", {f"{'you' if k[0] == 0 else 'enemy'} {k[1]}{'' if k[2] is None else ' ' + 'LR'[k[2]]}": round(v, 2)
                     for k, v in obs.towers.items()})
    print(f"troops seen: {len(obs.units)}")
    print(f"wrote {out}: check the boxes line up with the arena, elixir bar, cards and tower HP bars")


def _roi(img, title):
    print(f"Drag a box around: {title}  (Enter/Space to confirm, c to cancel)")
    x, y, w, h = cv2.selectROI(title, img, showCrosshair=False)
    cv2.destroyWindow(title)
    H, W = img.shape[:2]
    if w == 0 or h == 0:
        return None
    return Box(x / W, y / H, (x + w) / W, (y + h) / H)


def cmd_pick(a):
    img = _read(a.image)
    lay = Layout.load(a.layout)
    steps = [
        ("arena", "the whole arena: from the top edge of the enemy side to the bottom edge of yours (above the cards)"),
        ("elixir_bar", "the elixir bar (the full bar, empty part included)"),
    ]
    for attr, title in steps:
        b = _roi(img, title)
        if b:
            setattr(lay, attr, b)
    slots = []
    for i in range(4):
        b = _roi(img, f"hand card {i + 1} (left to right)")
        slots.append(b or lay.hand_slots[i])
    lay.hand_slots = slots
    b = _roi(img, "the small 'next card' picture")
    if b:
        lay.next_slot = b
    b = _roi(img, "the champion ability button (only if a champion is on the field; c to skip)")
    if b:
        lay.ability_button = b
    # Tower HP bars, stored relative to the tower's tile position.
    b = _roi(img, "the ENEMY LEFT princess tower HP bar")
    if b:
        tx0, ty0 = lay.frac_to_tile(b.x0, b.y0)
        tx1, ty1 = lay.frac_to_tile(b.x1, b.y1)
        lay.princess_bar_tiles = (tx0 - LANE_X[0], ty0 - (32 - 6.5), tx1 - LANE_X[0], ty1 - (32 - 6.5))
    b = _roi(img, "the ENEMY KING tower HP bar (skip with c if not shown)")
    if b:
        tx0, ty0 = lay.frac_to_tile(b.x0, b.y0)
        tx1, ty1 = lay.frac_to_tile(b.x1, b.y1)
        lay.king_bar_tiles = (tx0 - WIDTH / 2, ty0 - (32 - 3.0), tx1 - WIDTH / 2, ty1 - (32 - 3.0))
    lay.save(a.layout)
    print(f"saved {a.layout}; now run: python -m flybrain.real.calibrate check --image {a.image}")


def cmd_templates(a):
    img = _read(a.image)
    lay = Layout.load(a.layout)
    d = Path(a.templates)
    d.mkdir(parents=True, exist_ok=True)
    names = [resolve(n.strip()) for n in a.cards.split(",")]
    if len(names) != 4:
        raise SystemExit("--cards needs exactly 4 names, left to right")
    pairs = list(zip(names, lay.hand_slots))
    if a.next:
        pairs.append((resolve(a.next.strip()), lay.next_slot))
    for name, box in pairs:
        cv2.imwrite(str(d / f"{name}.png"), box.crop(img))
        print(f"saved {d / (name + '.png')}")
    have = sorted(p.stem for p in d.glob("*.png"))
    print(f"card pictures so far: {have}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("screenshot")
    s.add_argument("--adb", default="adb"); s.add_argument("--serial"); s.add_argument("--connect")
    s.add_argument("--out", default="shot.png")
    for name in ("check", "pick", "templates"):
        p = sub.add_parser(name)
        p.add_argument("--image", required=True)
        p.add_argument("--layout", default="layout.json")
        p.add_argument("--templates", default="templates/cards")
        if name == "templates":
            p.add_argument("--cards", required=True)
            p.add_argument("--next")
    a = ap.parse_args()
    {"screenshot": cmd_screenshot, "check": cmd_check, "pick": cmd_pick, "templates": cmd_templates}[a.cmd](a)


if __name__ == "__main__":
    main()
