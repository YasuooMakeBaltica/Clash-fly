"""Count the opponent's elixir from the troops they deploy (needs the troop detector's types).

Each frame, enemy troops that weren't there before (no troop of the same type
close enough to have walked there) are new deployments. New troops of one type
that appear together form one card; the card is the one that summons that many
of them (3 Skeletons -> Skeletons, 15 -> Skeleton Army), and its cost comes off
the estimate. Troops that come out of the opponent's spawner buildings or
from dying troops (Golemites, Lava Pups ...) are free. Elixir comes back at
1 per 2.8 s (twice as fast in the last minute and in overtime).

Spells aren't seen, so the estimate is an upper bound: if it says the
opponent is low, they really are.
"""

from __future__ import annotations

import math
from collections import defaultdict

from ..envs.royale.db import DB, load

REGEN = 1 / 2.8
DOUBLE_AT = 120.0
SPAWNED_BY = {   # troop -> enemy building or troop that makes it for free
    "Skeleton": {"Tombstone", "GiantSkeleton", "SkeletonKing"}, "SpearGoblin": {"GoblinHut"},
    "Barbarian": {"BarbarianHut", "BattleRam"}, "FireSpirits": {"FirespiritHut"}, "Golemite": {"Golem"},
    "LavaPups": {"LavaHound"}, "ElixirGolem2": {"ElixirGolem4"}, "ElixirGolem1": {"ElixirGolem2"},
    "PhoenixEgg": {"Phoenix"}, "PhoenixNoRespawn": {"PhoenixEgg"}, "Goblin": {"GoblinCage", "GoblinGiant", "GoblinDrill"},
    "GoblinBrawler": {"GoblinCage"}, "Bat": {"DarkWitch"}, "VoodooHog": {"WitchMother"},
}
NOT_A_CARD = {"Golemite", "LavaPups", "ElixirGolem1", "ElixirGolem2", "PhoenixEgg", "PhoenixNoRespawn", "VoodooHog"}


class OwnDeploys:
    """Troops the bot just deployed, kept in the game state until the detector sees them.

    A new troop plays a deploy animation for about a second before it can be
    recognised, so the next screenshot often doesn't show it yet, and the coach
    would see an undefended lane and defend twice. For ``keep`` seconds after a
    play, its troops are added at the drop spot until the detector sees one of
    them nearby. A play that didn't happen (the card is still in the hand;
    played cards go to the back of the 8-card cycle) is dropped.
    """

    def __init__(self, db: DB | None = None, keep: float = 3.0, radius: float = 3.0):
        self.db = db or load()
        self.keep, self.radius = keep, radius
        self.pending: list[dict] = []

    def add(self, card: str, x: float, y: float, now: float) -> None:
        troops = [(spec.name, k) for spec, k in self.db.cards[card].summons]
        if troops:
            self.pending.append(dict(t=now, card=card, x=x, y=y, troops=troops))

    def fill(self, units: list, hand: list, now: float) -> list:
        """``units`` plus the pending troops the detector doesn't show yet."""
        from .perception import SeenUnit

        self.pending = [p for p in self.pending if now - p["t"] <= self.keep and p["card"] not in hand]
        out = list(units)
        for p in list(self.pending):
            chars = {c for c, _ in p["troops"]}
            if any(u.owner == 0 and u.char in chars and math.hypot(u.x - p["x"], u.y - p["y"]) <= self.radius
                   for u in units):
                self.pending.remove(p)            # seen: the detector has it from here on
                continue
            out += [SeenUnit(owner=0, x=p["x"], y=p["y"], size=1, char=c, score=0.0)
                    for c, k in p["troops"] for _ in range(k)]
        return out


class EnemyElixir:
    def __init__(self, db: DB | None = None, start: float = 5.0, match_radius: float = 3.0, group_radius: float = 3.5):
        self.db = db or load()
        self.elixir, self.time = start, None
        self.match_radius, self.group_radius = match_radius, group_radius
        self.history: list[list[tuple[str, float, float]]] = []   # enemy troops in the last frames (missed detections)
        self.plays: list[tuple[float, str, float]] = []     # (time, card, cost)
        self._cards = defaultdict(list)                     # character -> [(card, count, cost)]
        for name, c in self.db.cards.items():
            if c.event or not c.role:
                continue
            for spec, k in c.summons:
                self._cards[spec.name].append((name, k, c.elixir))

    def _regen(self, now: float) -> None:
        if self.time is not None:
            dt = max(0.0, now - self.time)
            rate = REGEN * (2 if self.time >= DOUBLE_AT else 1)
            self.elixir = min(10.0, self.elixir + rate * dt)
        self.time = now

    def card_for(self, char: str, n: int) -> tuple[str, float] | None:
        opts = self._cards.get(char)
        if not opts or char in NOT_A_CARD:
            return None
        name, k, cost = min(opts, key=lambda o: (abs(o[1] - n), o[2]))
        return name, cost

    def update(self, units, now: float) -> float:
        """``units``: this frame's SeenUnits (only enemy troops with a type are used). Returns the estimate."""
        self._regen(now)
        enemy = [(u.char, u.x, u.y) for u in units if u.owner == 1 and u.char]
        present = {c for c, _, _ in enemy}
        new = []
        for c, x, y in enemy:
            if any(c == pc and math.hypot(x - px, y - py) <= self.match_radius * (1 + age)
                   for age, frame in enumerate(reversed(self.history)) for pc, px, py in frame):
                continue
            if SPAWNED_BY.get(c, set()) & present:
                continue
            new.append((c, x, y))
        while new:
            c, x, y = new.pop(0)
            group = [(c, x, y)] + [t for t in new if t[0] == c and math.hypot(t[1] - x, t[2] - y) <= self.group_radius]
            new = [t for t in new if t not in group]
            card = self.card_for(c, len(group))
            if card:
                self.elixir = max(0.0, self.elixir - card[1])
                self.plays.append((now, card[0], card[1]))
        self.history = (self.history + [enemy])[-3:]
        return self.elixir
