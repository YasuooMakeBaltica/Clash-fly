"""A fake adb backed by the simulator, for testing the real-game bot end to end."""

from __future__ import annotations

import numpy as np

from ..envs.clash.sim import ClashSim
from .layout import Layout
from .synthetic import render


class FakeAdb:
    """Screenshots are rendered from ``sim``; taps play cards in ``sim`` as player 0."""

    def __init__(self, sim: ClashSim, layout: Layout, size=(540, 960)):
        self.sim, self.lay, self.size = sim, layout, size
        self.taps: list[tuple[int, int]] = []
        self.plays: list[tuple[str, float, float, bool]] = []

    def screencap(self) -> np.ndarray:
        return render(self.sim, self.lay, self.size)

    def play_card(self, slot_xy, target_xy, pause: float = 0.0) -> None:
        w, h = self.size
        self.taps += [slot_xy, target_xy]
        fx, fy = slot_xy[0] / w, slot_xy[1] / h
        slot = next(i for i, b in enumerate(self.lay.hand_slots) if b.x0 <= fx <= b.x1 and b.y0 <= fy <= b.y1)
        card = self.sim.players[0].hand[slot]
        tx, ty = self.lay.frac_to_tile(target_xy[0] / w, target_xy[1] / h)
        ok = self.sim.play(0, card, tx, ty)
        self.plays.append((self.sim.deck[card].name, tx, ty, ok))
