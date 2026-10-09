"""A fake adb backed by the full simulator, for testing the real-game bot end to end."""

from __future__ import annotations

import numpy as np

from ..envs.royale.sim import Sim
from .layout import Layout
from .synthetic import render


class FakeAdb:
    """Screenshots are rendered from ``sim``; taps play cards in ``sim`` as player 0."""

    def __init__(self, sim: Sim, layout: Layout, size=(540, 960)):
        self.sim, self.lay, self.size = sim, layout, size
        self.taps: list[tuple[int, int]] = []
        self.plays: list[tuple[str, float, float, bool]] = []
        self.ability_taps = 0

    def screencap(self) -> np.ndarray:
        return render(self.sim, self.lay, self.size)

    def tap(self, x: int, y: int) -> None:
        w, h = self.size
        b = self.lay.ability_button
        if b.x0 <= x / w <= b.x1 and b.y0 <= y / h <= b.y1:
            self.ability_taps += 1
            self.sim.use_ability(0)

    def play_card(self, slot_xy, target_xy, pause: float = 0.0) -> None:
        w, h = self.size
        self.taps += [slot_xy, target_xy]
        fx, fy = slot_xy[0] / w, slot_xy[1] / h
        slot = next(i for i, b in enumerate(self.lay.hand_slots) if b.x0 <= fx <= b.x1 and b.y0 <= fy <= b.y1)
        card = self.sim.players[0].hand[slot]
        tx, ty = self.lay.frac_to_tile(target_xy[0] / w, target_xy[1] / h)
        ok = self.sim.play(0, card, tx, ty)
        self.plays.append((card, tx, ty, ok))
