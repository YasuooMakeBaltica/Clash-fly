"""Read the game state from a Clash Royale screenshot (BGR numpy image).

* elixir: share of the elixir bar that is pink.
* hand: template matching of each card slot against card pictures saved
  with ``calibrate templates`` (templates/cards/<Card name>.png).
* towers: share of each HP bar that is filled with the owner's colour
  (blue = you, red = enemy), tracked over time so a bar that disappears
  after being nearly empty counts as destroyed.
* troops: red/blue level badges and health bars in the arena, grouped into
  units. Troop types are unknown with this detector; a trained detector
  (e.g. YOLO) can replace :class:`BadgeDetector` later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..envs.clash.cards import DECK
from .layout import Box, Layout

CARD_NAMES = [c.name for c in DECK]


def _mask(hsv: np.ndarray, rng) -> np.ndarray:
    lo, hi = rng
    return cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))


def red_mask(hsv, lay: Layout):
    return _mask(hsv, lay.red_hsv) | _mask(hsv, lay.red2_hsv)


def blue_mask(hsv, lay: Layout):
    return _mask(hsv, lay.blue_hsv)


# ------------------------------------------------------------------ elixir
def read_elixir(img: np.ndarray, lay: Layout) -> float:
    crop = lay.elixir_bar.crop(img)
    if crop.size == 0:
        return 0.0
    m = _mask(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV), lay.elixir_hsv) > 0
    cols = m.mean(0) > 0.4
    if not cols.any():
        return 0.0
    # The bar fills from the left; use the rightmost filled column.
    filled = (np.flatnonzero(cols)[-1] + 1) / len(cols)
    return float(np.clip(round(filled * 10, 1), 0, 10))


def in_battle(img: np.ndarray, lay: Layout) -> bool:
    """A battle is on screen if the elixir bar shows any pink."""
    crop = lay.elixir_bar.crop(img)
    if crop.size == 0:
        return False
    return (_mask(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV), lay.elixir_hsv) > 0).mean() > 0.02


# -------------------------------------------------------------------- hand
class CardMatcher:
    """Matches card slots against saved card pictures."""

    size = (64, 80)  # template size (w, h) after resizing

    def __init__(self, template_dir: str | Path = "templates/cards", threshold: float = 0.6):
        self.threshold = threshold
        self.templates: dict[str, np.ndarray] = {}
        d = Path(template_dir)
        for name in CARD_NAMES:
            p = d / f"{name}.png"
            if p.exists():
                self.templates[name] = self._prep(cv2.imread(str(p)))

    def _prep(self, crop: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(cv2.resize(crop, self.size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        return g.astype(np.float32)

    def match(self, crop: np.ndarray) -> tuple[str | None, float]:
        if crop.size == 0 or not self.templates:
            return None, 0.0
        q = self._prep(crop)
        best, score = None, -1.0
        for name, t in self.templates.items():
            s = float(cv2.matchTemplate(q, t, cv2.TM_CCOEFF_NORMED)[0, 0])
            if s > score:
                best, score = name, s
        return (best, score) if score >= self.threshold else (None, score)


def read_hand(img: np.ndarray, lay: Layout, matcher: CardMatcher) -> tuple[list[str | None], str | None]:
    hand = [matcher.match(b.crop(img))[0] for b in lay.hand_slots]
    nxt = matcher.match(lay.next_slot.crop(img))[0]
    return hand, nxt


# ------------------------------------------------------------------ towers
def bar_fill(img: np.ndarray, box: Box, mask_fn) -> float | None:
    """Filled share of an HP bar, or None if no bar is visible."""
    crop = box.crop(img)
    if crop.size == 0:
        return None
    m = mask_fn(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)) > 0
    cols = m.mean(0) > 0.3
    if cols.sum() < max(2, 0.03 * len(cols)):
        return None
    return float((np.flatnonzero(cols)[-1] + 1) / len(cols))


@dataclass
class TowerTracker:
    """Keeps tower HP fractions over time.

    Towers never heal, so HP only goes down, but a single frame is not
    trusted: a troop standing in front of a bar hides part of it. The new
    value is the median of the last ``window`` readings. A missing bar means
    full HP on an untouched tower, but destroyed once the tower has taken
    damage and the bar has been gone for ``gone_frames`` frames in a row.
    """

    window: int = 3
    gone_frames: int = 2
    hp: dict = field(default_factory=dict)
    recent: dict = field(default_factory=dict)
    missing: dict = field(default_factory=dict)

    def update(self, img: np.ndarray, lay: Layout) -> dict:
        for key, box in lay.tower_bars().items():
            owner = key[0]
            fill = bar_fill(img, box, (lambda h: blue_mask(h, lay)) if owner == 0 else (lambda h: red_mask(h, lay)))
            prev = self.hp.get(key, 1.0)
            self.missing[key] = self.missing.get(key, 0) + 1 if fill is None else 0
            if fill is None:
                damaged = prev < 0.995
                if damaged and self.missing[key] >= self.gone_frames:
                    self.hp[key] = 0.0
                    continue
                fill = prev
            hist = (self.recent.get(key, []) + [fill])[-self.window:]
            self.recent[key] = hist
            self.hp[key] = 0.0 if prev == 0 else min(prev, float(np.median(hist)))
        return dict(self.hp)


# ------------------------------------------------------------------ troops
@dataclass
class SeenUnit:
    owner: int     # 0 you, 1 enemy
    x: float       # tile coordinates
    y: float
    size: int      # number of badges/bars merged (≈ how many bodies)
    card: str | None = None


class BadgeDetector:
    """Finds troops by their coloured level badges / health bars above them."""

    def __init__(self, min_area_frac: float = 2e-5, max_area_frac: float = 6e-4, merge_tiles: float = 1.2):
        self.min_area_frac, self.max_area_frac, self.merge_tiles = min_area_frac, max_area_frac, merge_tiles

    def detect(self, img: np.ndarray, lay: Layout) -> list[SeenUnit]:
        h, w = img.shape[:2]
        x0, y0, x1, y1 = lay.arena.px(w, h)
        hsv = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        # Ignore the tower HP bars
        ignore = np.zeros(hsv.shape[:2], np.uint8)
        for box in lay.tower_bars().values():
            bx0, by0, bx1, by1 = box.px(w, h)
            ignore[max(0, by0 - y0 - 2):max(0, by1 - y0 + 2), max(0, bx0 - x0 - 2):max(0, bx1 - x0 + 2)] = 1
        units: list[SeenUnit] = []
        for owner, m in ((0, blue_mask(hsv, lay)), (1, red_mask(hsv, lay))):
            m = m * (1 - ignore)
            n, _, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
            pts = []
            for i in range(1, n):
                area = stats[i, cv2.CC_STAT_AREA] / (w * h)
                if self.min_area_frac <= area <= self.max_area_frac:
                    cx, cy = cents[i]
                    tx, ty = lay.frac_to_tile((cx + x0) / w, (cy + y0) / h)
                    pts.append((tx, ty - 0.6))  # badge floats above the body
            units += [SeenUnit(owner, x, y, k) for x, y, k in self._merge(pts)]
        return units

    def _merge(self, pts):
        groups: list[list[tuple[float, float]]] = []
        for p in pts:
            for g in groups:
                gx = np.mean([q[0] for q in g]); gy = np.mean([q[1] for q in g])
                if np.hypot(p[0] - gx, p[1] - gy) <= self.merge_tiles:
                    g.append(p)
                    break
            else:
                groups.append([p])
        return [(float(np.mean([q[0] for q in g])), float(np.mean([q[1] for q in g])), len(g)) for g in groups]


# ------------------------------------------------------------------- frame
@dataclass
class Observation:
    elixir: float
    hand: list[str | None]
    next_card: str | None
    towers: dict
    units: list[SeenUnit]


class Perception:
    def __init__(self, layout: Layout, template_dir: str | Path = "templates/cards", detector=None):
        self.layout = layout
        self.matcher = CardMatcher(template_dir)
        self.detector = detector or BadgeDetector()
        self.towers = TowerTracker()

    def reset(self) -> None:
        self.towers = TowerTracker()

    def read(self, img: np.ndarray) -> Observation:
        lay = self.layout
        hand, nxt = read_hand(img, lay, self.matcher)
        return Observation(read_elixir(img, lay), hand, nxt, self.towers.update(img, lay), self.detector.detect(img, lay))
