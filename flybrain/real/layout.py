"""Where things are on the Clash Royale screen.

All boxes are fractions of the game screen (0..1 in x and y), so a layout
works at any resolution with the same aspect ratio. The defaults are
ESTIMATES for portrait 9:16 (LDPlayer at 540x960) and must be checked with
``python -m flybrain.real.calibrate check`` on a real screenshot, then fixed
with ``python -m flybrain.real.calibrate pick``.

Arena mapping: the simulator's 18x32 tile grid is mapped linearly onto the
``arena`` box, with tile y = 0 at the bottom (your side) and 32 at the top.
The game camera is tilted, so tiles are shorter than they are wide on
screen; a linear map is a first approximation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..envs.clash.cards import HEIGHT, LANE_X, WIDTH


@dataclass
class Box:
    x0: float
    y0: float
    x1: float
    y1: float

    def px(self, w: int, h: int) -> tuple[int, int, int, int]:
        return int(round(self.x0 * w)), int(round(self.y0 * h)), int(round(self.x1 * w)), int(round(self.y1 * h))

    def crop(self, img):
        h, w = img.shape[:2]
        x0, y0, x1, y1 = self.px(w, h)
        return img[y0:y1, x0:x1]

    def center(self) -> tuple[float, float]:
        return (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2


def _slots() -> list[Box]:
    w, gap, x = 0.165, 0.012, 0.29
    return [Box(x + i * (w + gap), 0.835, x + i * (w + gap) + w, 0.945) for i in range(4)]


@dataclass
class Layout:
    arena: Box = field(default_factory=lambda: Box(0.0, 0.045, 1.0, 0.805))
    elixir_bar: Box = field(default_factory=lambda: Box(0.28, 0.955, 0.98, 0.985))
    hand_slots: list[Box] = field(default_factory=_slots)
    next_slot: Box = field(default_factory=lambda: Box(0.04, 0.885, 0.17, 0.965))
    # Tower HP bars, as boxes in tile coordinates (x0, y_top, x1, y_bottom).
    # Princess bars sit just above each tower; the king's above the king.
    princess_bar_tiles: tuple[float, float, float, float] = (-1.7, 2.4, 1.7, 1.8)
    king_bar_tiles: tuple[float, float, float, float] = (-2.2, 2.9, 2.2, 2.3)
    # HSV colour ranges (OpenCV: H 0-180, S and V 0-255)
    elixir_hsv: tuple = ((135, 90, 110), (170, 255, 255))
    blue_hsv: tuple = ((95, 110, 110), (120, 255, 255))
    red_hsv: tuple = ((0, 120, 110), (8, 255, 255))
    red2_hsv: tuple = ((172, 120, 110), (180, 255, 255))

    # ------------------------------------------------------------ mapping
    def tile_to_frac(self, tx: float, ty: float) -> tuple[float, float]:
        a = self.arena
        return a.x0 + tx / WIDTH * (a.x1 - a.x0), a.y1 - ty / HEIGHT * (a.y1 - a.y0)

    def frac_to_tile(self, fx: float, fy: float) -> tuple[float, float]:
        a = self.arena
        return (fx - a.x0) / (a.x1 - a.x0) * WIDTH, (a.y1 - fy) / (a.y1 - a.y0) * HEIGHT

    def tile_to_px(self, tx: float, ty: float, w: int, h: int) -> tuple[int, int]:
        fx, fy = self.tile_to_frac(tx, ty)
        return int(round(fx * w)), int(round(fy * h))

    def tile_box(self, tx0: float, ty_top: float, tx1: float, ty_bottom: float) -> Box:
        x0, y0 = self.tile_to_frac(tx0, ty_top)
        x1, y1 = self.tile_to_frac(tx1, ty_bottom)
        return Box(x0, y0, x1, y1)

    def tower_bars(self) -> dict[tuple[int, str, int | None], Box]:
        """HP bar box per tower, keyed (owner, kind, lane); owner 0 = you (bottom)."""
        out = {}
        for owner in (0, 1):
            for lane, lx in enumerate(LANE_X):
                ty = 6.5 if owner == 0 else HEIGHT - 6.5
                dx0, dt, dx1, db = self.princess_bar_tiles
                out[(owner, "princess", lane)] = self.tile_box(lx + dx0, ty + dt, lx + dx1, ty + db)
            ty = 3.0 if owner == 0 else HEIGHT - 3.0
            dx0, dt, dx1, db = self.king_bar_tiles
            out[(owner, "king", None)] = self.tile_box(WIDTH / 2 + dx0, ty + dt, WIDTH / 2 + dx1, ty + db)
        return out

    # ---------------------------------------------------------------- io
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path | None) -> "Layout":
        if path is None or not Path(path).exists():
            return cls()
        d = json.loads(Path(path).read_text())
        lay = cls()
        for k, v in d.items():
            if k in ("arena", "elixir_bar", "next_slot"):
                setattr(lay, k, Box(**v))
            elif k == "hand_slots":
                lay.hand_slots = [Box(**b) for b in v]
            elif k.endswith("_hsv"):
                setattr(lay, k, tuple(tuple(x) for x in v))
            elif k.endswith("_tiles"):
                setattr(lay, k, tuple(v))
        return lay
