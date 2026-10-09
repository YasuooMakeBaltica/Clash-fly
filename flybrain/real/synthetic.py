"""Render fake Clash Royale screenshots from a simulator state.

Only for testing the perception -> state -> brain -> tap pipeline without the
real game: it draws the elements the perception code looks for (pink elixir
bar, card pictures in the hand slots, coloured tower HP bars, troop level
badges) at the positions given by a :class:`Layout`. It says nothing about
whether the default layout matches the real game; calibration does.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..envs.clash.cards import DECK
from ..envs.clash.sim import ClashSim
from .layout import Layout

ELIXIR_BGR = (224, 58, 214)   # magenta
BLUE_BGR = (240, 140, 58)
RED_BGR = (50, 50, 230)
DARK_BGR = (40, 40, 40)


def card_picture(idx: int, size=(90, 110)) -> np.ndarray:
    """A distinct fake card picture per card (random blocks + initials)."""
    rng = np.random.default_rng(1000 + idx)
    w, h = size
    img = np.zeros((h, w, 3), np.uint8)
    for _ in range(14):
        x, y = rng.integers(0, w), rng.integers(0, h)
        cv2.rectangle(img, (int(x), int(y)), (int(x + rng.integers(8, 40)), int(y + rng.integers(8, 40))),
                      tuple(int(c) for c in rng.integers(30, 120, 3)), -1)
    cv2.putText(img, DECK[idx].name[:2], (8, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (230, 230, 230), 2)
    return img


def write_templates(directory: str | Path) -> None:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(DECK):
        cv2.imwrite(str(d / f"{c.name}.png"), card_picture(i))


def render(sim: ClashSim, lay: Layout, size=(540, 960)) -> np.ndarray:
    w, h = size
    img = np.full((h, w, 3), (60, 110, 60), np.uint8)
    # river
    x0, y0 = lay.tile_to_px(0, 17, w, h)
    x1, y1 = lay.tile_to_px(18, 15, w, h)
    cv2.rectangle(img, (x0, y0), (x1, y1), (170, 120, 60), -1)
    # bottom UI panel
    ay1 = lay.arena.px(w, h)[3]
    cv2.rectangle(img, (0, ay1), (w, h), (50, 35, 30), -1)
    # towers and HP bars
    bars = lay.tower_bars()
    for t in sim.towers:
        tx, ty = lay.tile_to_px(t.x - 1.5, t.y + 1.5, w, h)
        bx, by = lay.tile_to_px(t.x + 1.5, t.y - 1.5, w, h)
        cv2.rectangle(img, (tx, ty), (bx, by), (120, 120, 120) if t.hp > 0 else (70, 70, 70), -1)
        if t.hp <= 0 or (t.kind == "king" and not t.active and t.hp >= t.max_hp):
            continue
        X0, Y0, X1, Y1 = bars[(t.owner, t.kind, t.lane)].px(w, h)
        cv2.rectangle(img, (X0, Y0), (X1, Y1), DARK_BGR, -1)
        fill = X0 + int(round((X1 - X0) * t.hp / t.max_hp))
        cv2.rectangle(img, (X0, Y0), (max(X0, fill - 1), Y1), BLUE_BGR if t.owner == 0 else RED_BGR, -1)
    # troops: body + level badge above
    for u in sim.units:
        cx, cy = lay.tile_to_px(u.x, u.y, w, h)
        cv2.circle(img, (cx, cy), 7, (90, 90, 90), -1)
        bx, by = lay.tile_to_px(u.x, u.y + 0.6, w, h)
        cv2.rectangle(img, (bx - 4, by - 3), (bx + 4, by + 3), BLUE_BGR if u.owner == 0 else RED_BGR, -1)
    # hand
    p = sim.players[0]
    for slot, idx in zip(lay.hand_slots, p.hand):
        X0, Y0, X1, Y1 = slot.px(w, h)
        img[Y0:Y1, X0:X1] = cv2.resize(card_picture(idx), (X1 - X0, Y1 - Y0))
    X0, Y0, X1, Y1 = lay.next_slot.px(w, h)
    img[Y0:Y1, X0:X1] = cv2.resize(card_picture(p.queue[0]), (X1 - X0, Y1 - Y0))
    # elixir bar
    X0, Y0, X1, Y1 = lay.elixir_bar.px(w, h)
    cv2.rectangle(img, (X0, Y0), (X1, Y1), DARK_BGR, -1)
    fill = X0 + int(round((X1 - X0) * p.elixir / 10))
    if fill > X0:
        cv2.rectangle(img, (X0, Y0), (fill - 1, Y1), ELIXIR_BGR, -1)
    return img
