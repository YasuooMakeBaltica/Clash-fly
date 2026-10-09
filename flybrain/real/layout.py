"""Where things are on the Clash Royale screen.

All boxes are fractions of the game screen (0..1 in x and y), so a layout
works at any resolution with the same aspect ratio. The defaults were
measured on a real battle screenshot (LDPlayer, portrait 540x960). Check
yours with ``python -m flybrain.real.calibrate check`` and fix boxes with
``python -m flybrain.real.calibrate pick`` if they are off. The king tower
HP bars (only shown once the king is hit) were measured on labelled real
frames instead.

Arena mapping: the simulator's 18x32 tile grid is mapped linearly onto the
``arena`` box, with tile y = 0 at the bottom (your side) and 32 at the top.
On screen a tile is about 25 px wide and 21 px tall at 540x960 (the grass
checkerboard is one tile per square).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..envs.royale.sim import HEIGHT, LANE_X, WIDTH


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


def _px(x0, y0, x1, y1, w=540, h=960) -> Box:
    return Box(x0 / w, y0 / h, x1 / w, y1 / h)


def _slots() -> list[Box]:
    return [_px(x, 797, x + 94, 912) for x in (122, 223, 325, 426)]


@dataclass
class Layout:
    arena: Box = field(default_factory=lambda: _px(41, 79, 499, 741))
    elixir_bar: Box = field(default_factory=lambda: _px(150, 926, 519, 947))
    hand_slots: list[Box] = field(default_factory=_slots)
    next_slot: Box = field(default_factory=lambda: _px(30, 895, 71, 947))
    ability_button: Box = field(default_factory=lambda: Box(0.80, 0.745, 0.96, 0.81))  # champion ability (estimate)
    # Tower HP bars, as boxes in tile coordinates relative to the tower
    # (x0, y_top, x1, y_bottom). Enemy bars sit above their towers, yours
    # below; the crown/level icon left of each bar is not included.
    princess_bar_tiles: tuple[float, float, float, float] = (-0.87, 3.75, 1.42, 3.12)
    own_princess_bar_tiles: tuple[float, float, float, float] = (-0.87, 0.56, 1.42, -0.02)
    # King bars only show once the king is hit; measured on ~1,400 labelled king bars in real frames
    # (KataCR dataset, aligned to this screen like troops.py).
    king_bar_tiles: tuple[float, float, float, float] = (-1.17, 5.3, 2.01, 4.94)
    own_king_bar_tiles: tuple[float, float, float, float] = (-1.13, -2.16, 2.03, -2.61)
    # HSV colour ranges (OpenCV: H 0-180, S and V 0-255)
    elixir_hsv: tuple = ((135, 90, 110), (170, 255, 255))
    blue_hsv: tuple = ((95, 110, 110), (120, 255, 255))
    red_hsv: tuple = ((0, 120, 110), (8, 255, 255))
    red2_hsv: tuple = ((165, 110, 110), (180, 255, 255))

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
                dx0, dt, dx1, db = self.own_princess_bar_tiles if owner == 0 else self.princess_bar_tiles
                out[(owner, "princess", lane)] = self.tile_box(lx + dx0, ty + dt, lx + dx1, ty + db)
            ty = 3.0 if owner == 0 else HEIGHT - 3.0
            dx0, dt, dx1, db = self.own_king_bar_tiles if owner == 0 else self.king_bar_tiles
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
            if k in ("arena", "elixir_bar", "next_slot", "ability_button"):
                setattr(lay, k, Box(**v))
            elif k == "hand_slots":
                lay.hand_slots = [Box(**b) for b in v]
            elif k.endswith("_hsv"):
                setattr(lay, k, tuple(tuple(x) for x in v))
            elif k.endswith("_tiles"):
                setattr(lay, k, tuple(v))
        return lay
