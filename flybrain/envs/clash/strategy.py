"""Basic Clash Royale strategy: a player-frame view, card placement, a coach and bots.

The coach encodes the basic strategies a new player is taught:

1. Finish a tower with a spell when it's in range of a kill.
2. Defend first. Counter the threat with the right card type:
   tanks -> tank killer, swarms -> area spell or splash, melee -> swarm,
   ranged -> mini tank. Don't over-commit if your defence already holds.
3. Make positive elixir trades with spells (hit more value than the cost).
4. Counter-push: support troops that survived defence with more troops.
5. Punish: when the opponent drops a tank at the back, hit the other lane.
6. Don't leak elixir: at full elixir, start a push (tank at the back, or
   at the bridge in double elixir) in the lane with the weaker enemy tower.

Lanes are 0 (left) and 1 (right) in absolute x, the same for both players.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .cards import DECK, LANE_X, RIVER_LO, WIDTH, Card
from .sim import ClashSim, Unit, abs_y, frame_y

THREAT_LINE = 20.0   # enemy troops closer than this (own frame y) count as threats


@dataclass
class Move:
    card: int     # index into DECK
    lane: int     # 0 left, 1 right
    x: float      # absolute drop position
    y: float


class View:
    """What player ``me`` can see, in their own frame (own side is y < 15)."""

    def __init__(self, sim: ClashSim, me: int):
        self.sim, self.me, self.foe = sim, me, 1 - me
        self.p = sim.players[me]

    def fy(self, y: float) -> float:
        return frame_y(self.me, y)

    @property
    def elixir(self) -> float:
        return self.p.elixir

    def affordable(self) -> list[int]:
        return [c for c in self.p.hand if DECK[c].cost <= self.p.elixir]

    def threats(self, lane: int) -> list[Unit]:
        return [u for u in self.sim.units if u.owner == self.foe and u.lane == lane
                and self.fy(u.y) < THREAT_LINE]

    def threat_hp(self, lane: int) -> float:
        return sum(u.hp for u in self.threats(lane))

    def defenders(self, lane: int) -> list[Unit]:
        return [u for u in self.sim.units if u.owner == self.me and u.lane == lane
                and self.fy(u.y) < RIVER_LO + 1]

    def pushers(self, lane: int) -> list[Unit]:
        """My troops heading for the enemy in this lane (past my princess tower)."""
        return [u for u in self.sim.units if u.owner == self.me and u.lane == lane and self.fy(u.y) >= 9]

    def enemy_back_tank(self) -> int | None:
        """Lane of an enemy tank still near their king (a slow push being built)."""
        for u in self.sim.units:
            if u.owner == self.foe and u.card.role == "tank" and self.fy(u.y) > 24:
                return u.lane
        return None

    def tower_frac(self, mine: bool, lane: int) -> float:
        t = self.sim.tower(self.me if mine else self.foe, "princess", lane)
        return t.hp / t.max_hp

    def weak_lane(self) -> int:
        """Lane to attack: the enemy princess tower with less health."""
        f = [self.tower_frac(False, 0), self.tower_frac(False, 1)]
        if f[0] == f[1]:
            return int(self.sim.rng.integers(2))
        return int(np.argmin(f))


# ----------------------------------------------------------------- spells
def spell_value(view: View, card: Card, x: float, y: float) -> float:
    """Elixir worth of enemy troops a spell at (x, y) would kill or damage."""
    v = 0.0
    for u in view.sim.units:
        if u.owner == view.foe and math.dist((u.x, u.y), (x, y)) <= card.radius:
            v += min(u.hp, card.damage) / u.card.hp * u.card.cost / u.card.count
    return v


def best_spell_spot(view: View, card: Card, lane: int | None = None) -> tuple[float, float, float]:
    best = (0.0, 0.0, 0.0)
    for u in view.sim.units:
        if u.owner == view.foe and (lane is None or u.lane == lane):
            v = spell_value(view, card, u.x, u.y)
            if v > best[0]:
                best = (v, u.x, u.y)
    return best


# -------------------------------------------------------------- placement
def place(view: View, card_idx: int, lane: int) -> tuple[float, float]:
    """Where to drop ``card_idx`` in ``lane``. Returns absolute (x, y)."""
    card = DECK[card_idx]
    me = view.me
    lx = LANE_X[lane]
    toward_centre = 1.0 if lane == 0 else -1.0

    if card.kind == "spell":
        v, x, y = best_spell_spot(view, card, lane)
        if v > 0:
            return x, y
        t = view.sim.lane_target(me, lane)
        return t.x, t.y

    threats = view.threats(lane)
    if threats:
        lead = min(threats, key=lambda u: view.fy(u.y))
        lead_fy = view.fy(lead.y)
        if card.role == "ranged":
            return lx + 1.5 * toward_centre, abs_y(me, 4.5)       # behind the princess tower
        if lead_fy > RIVER_LO:                                     # still across the river: meet it
            return lx + toward_centre, abs_y(me, 11.0)
        if card.role == "swarm":
            return lead.x, abs_y(me, max(0.5, lead_fy - 1.0))     # surround it
        # tanks, mini tanks, tank killers: block in front, a bit toward the centre (pull)
        return float(np.clip(lead.x + toward_centre, 0.5, WIDTH - 0.5)), abs_y(me, max(0.5, lead_fy - 2.5))

    pushers = view.pushers(lane)
    if card.role == "tank":
        if view.elixir >= 8 or view.sim.double_elixir:
            return lx, abs_y(me, 14.0)                             # bridge
        return WIDTH / 2 - toward_centre * 1.5, abs_y(me, 1.0)     # behind the king tower
    if pushers:
        lead = max(pushers, key=lambda u: view.fy(u.y))
        return float(np.clip(lead.x, 0.5, WIDTH - 0.5)), abs_y(me, min(14.0, view.fy(lead.y) - 2.0))
    if card.role == "ranged":
        return lx, abs_y(me, 10.0)
    return lx, abs_y(me, 14.0)                                     # bridge


# ------------------------------------------------------------------ coach
COUNTERS = {
    # primary threat role -> card names to answer with, best first
    "tank": ["Mini P.E.K.K.A", "Knight", "Goblins", "Musketeer", "Archers"],
    "tank_killer": ["Goblins", "Knight", "Archers", "Musketeer", "Mini P.E.K.K.A"],
    "mini_tank": ["Goblins", "Mini P.E.K.K.A", "Knight", "Musketeer", "Archers"],
    "ranged": ["Knight", "Mini P.E.K.K.A", "Goblins", "Fireball"],
    "swarm": ["Arrows", "Knight", "Musketeer", "Archers"],
}


class Coach:
    """Rule-based player using basic strategy. ``suggest`` returns (card or None, lane)."""

    name = "coach"

    def suggest(self, view: View) -> tuple[int | None, int]:
        hand = view.affordable()
        names = {DECK[c].name: c for c in hand}

        # 1. Finish a tower with a spell.
        for spell in ("Fireball", "Arrows"):
            if spell in names:
                for lane in (0, 1):
                    t = view.sim.tower(view.foe, "princess", lane)
                    if t.alive and t.hp <= DECK[names[spell]].tower_damage:
                        return names[spell], lane

        # 2. Defend the more threatened lane.
        lanes = sorted((0, 1), key=lambda ln: -view.threat_hp(ln))
        for lane in lanes:
            threats = view.threats(lane)
            if not threats:
                continue
            th_hp = sum(u.hp for u in threats)
            has_tank = any(u.card.role == "tank" for u in threats)
            def_hp = sum(u.hp for u in view.defenders(lane))
            if def_hp >= 0.8 * th_hp and not has_tank:
                continue  # already handled; don't over-commit
            counter = self._counter(view, threats, names, lane)
            if counter is not None:
                return counter, lane
            if min(view.fy(u.y) for u in threats) < 10:
                return None, lane  # can't afford an answer yet; let the tower tank it

        # 3. Positive spell trade anywhere.
        for spell in ("Fireball", "Arrows"):
            if spell in names:
                card = DECK[names[spell]]
                for lane in (0, 1):
                    v, _, _ = best_spell_spot(view, card, lane)
                    if v >= card.cost + 0.5:
                        return names[spell], lane

        # 4. Counter-push: back up troops that survived defence.
        for lane in (0, 1):
            push_hp = sum(u.hp for u in view.pushers(lane))
            if push_hp >= 600 and view.elixir >= 4:
                for name in ("Musketeer", "Archers", "Goblins", "Mini P.E.K.K.A", "Knight"):
                    if name in names:
                        return names[name], lane

        # 5. Punish a tank dropped at the back with pressure in the other lane.
        back = view.enemy_back_tank()
        if back is not None:
            for name in ("Mini P.E.K.K.A", "Goblins", "Knight"):
                if name in names:
                    return names[name], 1 - back

        # 6. Don't leak elixir: start a push in the weak lane.
        full = 9.0 if not view.sim.double_elixir else 7.0
        if view.elixir >= full:
            lane = view.weak_lane()
            if "Giant" in names:
                return names["Giant"], lane
            troops = sorted((c for c in hand if DECK[c].kind == "troop"), key=lambda c: DECK[c].cost)
            if troops:
                return troops[0], lane
        return None, 0

    def _counter(self, view: View, threats: list[Unit], names: dict[str, int], lane: int) -> int | None:
        roles = [u.card.role for u in threats]
        n_swarm = sum(r in ("swarm", "ranged") and u.card.count > 1 for r, u in zip(roles, threats))
        if "tank" in roles:
            primary = "tank"
        elif n_swarm >= 2:
            primary = "swarm"
        else:
            primary = max(threats, key=lambda u: u.hp).card.role
        for name in COUNTERS.get(primary, []):
            if name in names:
                card = DECK[names[name]]
                if card.kind == "spell" and best_spell_spot(view, card, lane)[0] < card.cost:
                    continue  # not worth the elixir
                return names[name]
        return None

    def decide(self, sim: ClashSim, player: int) -> Move | None:
        view = View(sim, player)
        card, lane = self.suggest(view)
        if card is None:
            return None
        return Move(card, lane, *place(view, card, lane))


class RandomBot:
    """Plays a random affordable card at a random spot about half the time."""

    name = "random"

    def __init__(self, seed: int | None = None, rate: float = 0.5):
        self.rng = np.random.default_rng(seed)
        self.rate = rate

    def decide(self, sim: ClashSim, player: int) -> Move | None:
        view = View(sim, player)
        hand = view.affordable()
        if not hand or self.rng.random() > self.rate:
            return None
        card = int(self.rng.choice(hand))
        lane = int(self.rng.integers(2))
        if DECK[card].kind == "spell":
            x, y = place(view, card, lane)
        else:
            x = float(self.rng.uniform(0.5, WIDTH - 0.5))
            y = abs_y(player, float(self.rng.uniform(1.0, 14.0)))
        return Move(card, lane, x, y)


class BasicBot:
    """Defends with a random troop, otherwise waits for 8 elixir and pushes a random lane."""

    name = "basic"

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def decide(self, sim: ClashSim, player: int) -> Move | None:
        view = View(sim, player)
        hand = view.affordable()
        troops = [c for c in hand if DECK[c].kind == "troop"]
        for lane in (0, 1):
            if troops and any(view.fy(u.y) < 12 for u in view.threats(lane)) and not view.defenders(lane):
                card = int(self.rng.choice(troops))
                return Move(card, lane, *place(view, card, lane))
        if view.elixir >= 8 and hand:
            card = int(self.rng.choice(hand))
            lane = int(self.rng.integers(2))
            return Move(card, lane, *place(view, card, lane))
        return None


def play_match(bot0, bot1, seed: int | None = None, decision_every: float = 1.0) -> ClashSim:
    """Run a full match between two controllers (used for tests and sanity checks).

    Both players decide from the same game state, then both moves are applied,
    so neither side gets to react to the other's card within a decision.
    """
    sim = ClashSim(seed=seed)
    next_decision = 0.0
    while not sim.done:
        if sim.time >= next_decision - 1e-9:
            moves = [bot0.decide(sim, 0), bot1.decide(sim, 1)]
            for player, m in enumerate(moves):
                if m is not None:
                    sim.play(player, m.card, m.x, m.y)
            next_decision += decision_every
        sim.step()
    return sim
