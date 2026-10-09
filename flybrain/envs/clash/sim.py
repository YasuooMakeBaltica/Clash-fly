"""A simplified Clash Royale match simulator.

What's modelled: the 18x32 tile arena with a river and two bridges, princess
and king towers (king wakes when hit or when a princess tower falls), elixir
(1 per 2.8 s, double in the last minute), an 8-card deck cycling through a
4-card hand, deploy time, troops that walk their lane, cross at the bridges,
lock onto the nearest target in sight and attack, delayed area spells with
reduced tower damage, crowns, and a 3-minute match.

What's left out: collisions and pushing, air units, projectiles, overtime and
tiebreakers, card levels and elixir collector-style cards.

Coordinates are absolute: player 0 defends the bottom (y < 15), player 1 the
top (y > 17). ``frame_y`` / ``abs_y`` convert to and from a player's own
frame, where their side is always y < 15.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .cards import (
    DECK, DEPLOY_TIME, DOUBLE_ELIXIR_AT, ELIXIR_RATE, HEIGHT, KING_DAMAGE, KING_HIT_SPEED, KING_HP,
    KING_RANGE, LANE_X, MATCH_TIME, MAX_ELIXIR, PRINCESS_DAMAGE, PRINCESS_HIT_SPEED, PRINCESS_HP,
    PRINCESS_RANGE, RIVER_HI, RIVER_LO, START_ELIXIR, TOWER_SIZE, WIDTH, Card,
)

RIVER_Y = (RIVER_LO + RIVER_HI) / 2


def frame_y(player: int, y: float) -> float:
    return y if player == 0 else HEIGHT - y


abs_y = frame_y  # the mirror is its own inverse


def lane_of(x: float) -> int:
    return 0 if x < WIDTH / 2 else 1


@dataclass(eq=False)
class Tower:
    owner: int
    kind: str          # "princess" or "king"
    lane: int | None   # 0 left, 1 right, None for king
    x: float
    y: float
    hp: float
    max_hp: float
    active: bool = True
    cooldown: float = 0.0
    target: "Unit | None" = None
    is_building = True

    @property
    def alive(self) -> bool:
        return self.hp > 0


@dataclass(eq=False)
class Unit:
    uid: int
    owner: int
    card: Card
    card_idx: int
    x: float
    y: float
    hp: float
    deploy_left: float = DEPLOY_TIME
    cooldown: float = 0.0
    target: "Unit | Tower | None" = None
    retarget_in: float = 0.0
    is_building = False

    @property
    def alive(self) -> bool:
        return self.hp > 0

    @property
    def lane(self) -> int:
        return lane_of(self.x)


@dataclass(eq=False)
class Spell:
    owner: int
    card: Card
    card_idx: int
    x: float
    y: float
    delay: float


@dataclass
class Player:
    elixir: float = START_ELIXIR
    hand: list[int] = field(default_factory=list)    # card indices into DECK, 4 slots
    queue: list[int] = field(default_factory=list)   # next cards, front first
    crowns: int = 0
    leaked: float = 0.0       # elixir wasted while full
    spent: float = 0.0


class ClashSim:
    dt = 0.1

    def __init__(self, seed: int | None = None, deck: tuple[Card, ...] = DECK):
        self.rng = np.random.default_rng(seed)
        self.deck = deck
        self.reset()

    # ------------------------------------------------------------- set-up
    def reset(self) -> None:
        self.time = 0.0
        self.units: list[Unit] = []
        self.spells: list[Spell] = []
        self.events: list[dict] = []   # card plays, tower kills
        self._uid = 0
        self.players = [Player(), Player()]
        for p in self.players:
            order = list(self.rng.permutation(len(self.deck)))
            p.hand, p.queue = [int(i) for i in order[:4]], [int(i) for i in order[4:]]
        self.towers: list[Tower] = []
        for owner in (0, 1):
            for lane, x in enumerate(LANE_X):
                self.towers.append(Tower(owner, "princess", lane, x, abs_y(owner, 6.5),
                                         PRINCESS_HP, PRINCESS_HP))
            self.towers.append(Tower(owner, "king", None, WIDTH / 2, abs_y(owner, 3.0),
                                     KING_HP, KING_HP, active=False))
        self.done = False
        self.winner: int | None = None

    # ------------------------------------------------------------ queries
    def tower(self, owner: int, kind: str, lane: int | None = None) -> Tower:
        for t in self.towers:
            if t.owner == owner and t.kind == kind and (kind == "king" or t.lane == lane):
                return t
        raise KeyError((owner, kind, lane))

    def lane_target(self, owner: int, lane: int) -> Tower:
        """The enemy tower a troop of ``owner`` walks to in ``lane``."""
        princess = self.tower(1 - owner, "princess", lane)
        return princess if princess.alive else self.tower(1 - owner, "king")

    @property
    def double_elixir(self) -> bool:
        return self.time >= DOUBLE_ELIXIR_AT

    def can_play(self, player: int, card_idx: int) -> bool:
        p = self.players[player]
        return not self.done and card_idx in p.hand and p.elixir >= self.deck[card_idx].cost

    def valid_position(self, player: int, card_idx: int, x: float, y: float) -> bool:
        if not (0.5 <= x <= WIDTH - 0.5 and 0.5 <= y <= HEIGHT - 0.5):
            return False
        if self.deck[card_idx].kind == "spell":
            return True
        fy = frame_y(player, y)
        if fy <= RIVER_LO - 0.5:
            return True
        # Pocket: past the river in a lane whose enemy princess tower is down.
        lane = lane_of(x)
        return fy <= 20.0 and not self.tower(1 - player, "princess", lane).alive

    # ------------------------------------------------------------ actions
    def play(self, player: int, card_idx: int, x: float, y: float) -> bool:
        """Play a card at absolute (x, y). Returns False if not allowed."""
        if not self.can_play(player, card_idx) or not self.valid_position(player, card_idx, x, y):
            return False
        card = self.deck[card_idx]
        p = self.players[player]
        p.elixir -= card.cost
        p.spent += card.cost
        slot = p.hand.index(card_idx)
        p.hand[slot] = p.queue.pop(0)
        p.queue.append(card_idx)
        self.events.append(dict(t=round(self.time, 2), kind="play", player=player, card=card_idx,
                                x=round(x, 2), y=round(y, 2)))
        if card.kind == "spell":
            self.spells.append(Spell(player, card, card_idx, x, y, card.delay))
            return True
        offsets = {1: [(0, 0)], 2: [(-0.6, 0), (0.6, 0)],
                   4: [(-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5)]}[card.count]
        for dx, dy in offsets:
            self._uid += 1
            self.units.append(Unit(self._uid, player, card, card_idx,
                                   float(np.clip(x + dx, 0.5, WIDTH - 0.5)), y + dy, card.hp))
        return True

    # ----------------------------------------------------------- dynamics
    def step(self, dt: float | None = None) -> None:
        if self.done:
            return
        dt = dt or self.dt
        self.time += dt
        rate = ELIXIR_RATE * (2 if self.double_elixir else 1)
        for p in self.players:
            gain = rate * dt
            room = MAX_ELIXIR - p.elixir
            p.leaked += max(0.0, gain - room)
            p.elixir = min(MAX_ELIXIR, p.elixir + gain)

        for s in self.spells:
            s.delay -= dt
            if s.delay <= 0:
                self._resolve_spell(s)
        self.spells = [s for s in self.spells if s.delay > 0]

        # Random update order so neither player systematically hits first.
        for i in self.rng.permutation(len(self.units)):
            u = self.units[i]
            if u.alive:
                self._update_unit(u, dt)
        for i in self.rng.permutation(len(self.towers)):
            t = self.towers[i]
            if t.alive and t.active:
                self._update_tower(t, dt)
        self.units = [u for u in self.units if u.alive]

        if self.time >= MATCH_TIME - 1e-9 and not self.done:
            self._finish()

    def _resolve_spell(self, s: Spell) -> None:
        for u in self.units:
            if u.owner != s.owner and u.alive and math.dist((u.x, u.y), (s.x, s.y)) <= s.card.radius:
                u.hp -= s.card.damage
        for t in self.towers:
            if t.owner != s.owner and t.alive and math.dist((t.x, t.y), (s.x, s.y)) <= s.card.radius + TOWER_SIZE:
                self._damage_tower(t, s.card.tower_damage, s.owner)

    def _can_target(self, u: Unit, other) -> bool:
        if not other.alive or other.owner == u.owner:
            return False
        if u.card.targets == "buildings":
            return other.is_building
        return True  # no air units in this deck

    def _gap(self, u: Unit, other) -> float:
        d = math.dist((u.x, u.y), (other.x, other.y))
        return d - (TOWER_SIZE if other.is_building else 0.0)

    def _update_unit(self, u: Unit, dt: float) -> None:
        if u.deploy_left > 0:
            u.deploy_left -= dt
            return
        u.retarget_in -= dt
        t = u.target
        if t is not None and (not t.alive or (not t.is_building and self._gap(u, t) > u.card.sight + 1)):
            t = None
        if t is None or (u.retarget_in <= 0 and not (t is not None and self._gap(u, t) <= u.card.range)):
            u.retarget_in = 0.5
            best, best_d = None, u.card.sight
            for o in self.units:
                if self._can_target(u, o):
                    d = self._gap(u, o)
                    if d <= best_d:
                        best, best_d = o, d
            for o in self.towers:
                if self._can_target(u, o):
                    d = self._gap(u, o)
                    if d <= best_d:
                        best, best_d = o, d
            t = best or self.lane_target(u.owner, u.lane)
        u.target = t

        u.cooldown -= dt
        if self._gap(u, t) <= u.card.range:
            if u.cooldown <= 0:
                u.cooldown = u.card.hit_speed
                if t.is_building:
                    self._damage_tower(t, u.card.damage, u.owner)
                else:
                    t.hp -= u.card.damage
            return
        u.cooldown = max(u.cooldown, u.card.hit_speed * 0.5)  # wind-up after moving
        # Move, via the bridge if the target is across the river.
        wx, wy = t.x, t.y
        side_u = 0 if u.y < RIVER_LO else (1 if u.y > RIVER_HI else None)
        side_t = 0 if t.y < RIVER_LO else 1
        if side_u is not None and side_u != side_t:
            bx = min(LANE_X, key=lambda lx: abs(lx - u.x))
            wx, wy = bx, RIVER_Y
        dx, dy = wx - u.x, wy - u.y
        d = math.hypot(dx, dy)
        if d > 1e-6:
            step = min(d, u.card.speed * dt)
            u.x += dx / d * step
            u.y += dy / d * step

    def _update_tower(self, t: Tower, dt: float) -> None:
        rng = PRINCESS_RANGE if t.kind == "princess" else KING_RANGE
        if t.target is not None and (not t.target.alive or math.dist((t.x, t.y), (t.target.x, t.target.y)) > rng):
            t.target = None
        if t.target is None:
            best, best_d = None, rng
            for u in self.units:
                if u.owner != t.owner and u.alive and u.deploy_left <= 0:
                    d = math.dist((t.x, t.y), (u.x, u.y))
                    if d <= best_d:
                        best, best_d = u, d
            t.target = best
        t.cooldown -= dt
        if t.target is not None and t.cooldown <= 0:
            t.cooldown = PRINCESS_HIT_SPEED if t.kind == "princess" else KING_HIT_SPEED
            t.target.hp -= PRINCESS_DAMAGE if t.kind == "princess" else KING_DAMAGE

    def _damage_tower(self, t: Tower, amount: float, attacker: int) -> None:
        if not t.alive:
            return
        t.hp -= amount
        if t.kind == "king":
            t.active = True
        if t.hp <= 0:
            t.hp = 0
            self.events.append(dict(t=round(self.time, 2), kind="tower", player=attacker,
                                    tower=t.kind, lane=t.lane))
            if t.kind == "king":
                self.players[attacker].crowns = 3
                self._finish()
            else:
                self.players[attacker].crowns += 1
                self.tower(t.owner, "king").active = True

    def _finish(self) -> None:
        self.done = True
        c0, c1 = self.players[0].crowns, self.players[1].crowns
        self.winner = 0 if c0 > c1 else 1 if c1 > c0 else None

    # ----------------------------------------------------------- snapshot
    def snapshot(self) -> dict:
        """Compact state for replays."""
        return dict(
            t=round(self.time, 2),
            elixir=[round(p.elixir, 2) for p in self.players],
            hand=[list(p.hand) for p in self.players],
            next=[p.queue[0] for p in self.players],
            crowns=[p.crowns for p in self.players],
            towers=[[t.owner, t.kind, t.lane, round(t.hp / t.max_hp, 3), int(t.active)] for t in self.towers],
            units=[[u.uid, u.owner, u.card_idx, round(u.x, 2), round(u.y, 2), round(u.hp / u.card.hp, 3),
                    int(u.deploy_left > 0)] for u in self.units],
            spells=[[s.owner, s.card_idx, round(s.x, 2), round(s.y, 2)] for s in self.spells],
        )
