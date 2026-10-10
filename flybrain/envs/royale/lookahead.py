"""A coach that looks ahead before it defends.

When the coach wants to defend (or holds because no card fits), every card in
hand that could help is tried at a few spots, and so is waiting, each in a
copy of the match that runs ``horizon`` seconds with the opponent playing
nothing more. The option that leaves the best position wins. A position is
scored in elixir: tower health (``tower_hp`` hit points per elixir, plus
``crown`` for a destroyed tower), the troops left standing on both sides (each
worth its share of its card's cost, scaled by remaining health) and elixir in
hand. Waiting keeps elixir; a defence that loses its troops or lets the tower
get hit scores lower. A spell the rules want to cast for a good trade is played
out against waiting the same way, and kept if waiting comes out ahead (the
coach then plans as if that spell weren't in hand).

In the real game the copy is the state rebuilt from the screen
(``flybrain.real.state.build_sim``), so this works there too: about 20 short
simulations, well under a second on a CPU.
"""

from __future__ import annotations

import copy
import math

import numpy as np

from .sim import LANE_X, WIDTH, Sim, abs_y, frame_y
from .strategy import DROP_ON_TOWER, SIEGE, Coach, View, place

LOOK_DEFAULTS = dict(
    horizon=8.0,        # seconds simulated per option
    tower_hp=150.0,     # tower hit points worth one elixir
    crown=10.0,         # extra elixir worth of a destroyed tower
    danger=1.0,         # weight of enemy troops already on our side
    own=1.0,            # worth of our troops on the field relative to elixir in hand
    offense=0.25,       # weight of damage to enemy towers: the copy's opponent doesn't defend, so troops
                        # sent over the bridge look free (0.25 vs 1.0: +0.48 vs +0.47 against the rule coach)
    wait_margin=0.0,    # a card must beat waiting by this much elixir to be played
    trades=1.0,         # 1: also play spell trades out against waiting
                        # (+0.12 ± 0.04 per game against the lookahead without it)
)

_MEMO: dict = {}
_CHAR_VALUE: dict[str, float] = {}


def _db_memo(db) -> dict:
    """id -> object for everything reachable from the card database, so copies share it."""
    if _MEMO.get("db") is db:
        return _MEMO["memo"]
    memo, stack = {}, [db]
    while stack:
        o = stack.pop()
        if id(o) in memo or isinstance(o, (int, float, str, bool, type(None))):
            continue
        memo[id(o)] = o
        if isinstance(o, dict):
            stack.extend(o.keys())
            stack.extend(o.values())
        elif isinstance(o, (list, tuple, set, frozenset)):
            stack.extend(o)
        elif hasattr(o, "__dict__"):
            stack.extend(vars(o).values())
    _MEMO.update(db=db, memo=memo)
    return memo


def clone(sim: Sim) -> Sim:
    """A copy of the match that can be stepped without touching the original (shares the card data)."""
    events, sim.events = sim.events, []          # the match log isn't needed in a copy
    try:
        return copy.deepcopy(sim, memo=dict(_db_memo(sim.db)))
    finally:
        sim.events = events


def char_value(spec, db) -> float:
    """Elixir worth of one unit of this character: its card's cost split by hit points over the
    card's units (Goblin Giant about 5.6 of 6, its riders the rest). Characters no card summons
    directly (Golemites, Lava Pups ...) count 1 elixir per 600 hit points."""
    if not _CHAR_VALUE:
        for _, c in sorted(db.cards.items(), key=lambda kv: -kv[1].elixir):
            if c.event or not c.role or not c.summons:
                continue
            parts = []
            for s, k in c.summons:
                parts.append((s, k))
                if s.spawn_character is not None and s.spawn_attach:
                    parts.append((s.spawn_character, max(1, s.spawn_number or 1)))
            total = sum(max(s.hp + s.shield, 1) * k for s, k in parts)
            for s, k in parts:                    # cheaper cards last: Goblin is worth Goblins' share
                _CHAR_VALUE[s.name] = c.elixir * max(s.hp + s.shield, 1) / total
    if spec.name in _CHAR_VALUE:
        return _CHAR_VALUE[spec.name]
    return min(3.0, (spec.hp + spec.shield) / 600.0)


def position_value(sim: Sim, me: int, **weights) -> float:
    """How good the match looks for ``me``, in elixir (``weights``: LOOK_DEFAULTS entries to change)."""
    w = {**LOOK_DEFAULTS, **weights}
    own, danger = w["own"], w["danger"]
    v = sim.players[me].elixir
    for t in sim.towers():
        s = 1 if t.owner == me else -w["offense"]
        v += s * (max(t.hp, 0.0) if t.alive else 0.0) / w["tower_hp"]
        if not t.alive or t.hp <= 0:
            v -= s * w["crown"]
    for u in sim.units:
        if u.tower or not u.alive or u.timed or u.spec is None:
            continue
        frac = max(0.0, u.hp + u.shield) / max(u.max_hp + u.spec.shield, 1.0)
        w = char_value(u.spec, sim.db) * frac
        if u.owner == me:
            v += w * own
        else:
            v -= w * (danger if frame_y(me, u.y) < 15 else 1.0)
    return v


def rollout(sim: Sim, me: int, play: tuple[str, float, float] | None, horizon: float) -> Sim | None:
    """The match ``horizon`` seconds after ``me`` plays ``play`` (card, x, y) or waits (None);
    None if the play isn't possible."""
    c = clone(sim)
    if play is not None and not c.play(me, *play):
        return None
    for _ in range(int(round(horizon / c.dt))):
        if c.done:
            break
        c.step()
    return c


def off_towers(sim: Sim, me: int, x: float, y: float) -> tuple[float, float]:
    """(x, y) moved just outside ``me``'s crown towers: the game doesn't let troops be dropped on them
    (princess towers cover 3x3 tiles, the king tower 4x4)."""
    for t in sim.towers(me):
        if not t.alive:
            continue
        h = 1.5 if t.tower == "princess" else 2.0
        dx, dy = x - t.x, y - t.y
        if abs(dx) < h and abs(dy) < h:
            if abs(dx) > abs(dy):
                x = t.x + math.copysign(h + 0.1, dx)
            else:
                y = t.y + math.copysign(h + 0.1, dy if dy else (1.0 if me == 0 else -1.0))   # towards the river
    return x, y


def spots(view: View, name: str, lane: int) -> list[tuple[float, float]]:
    """Where to try ``name``: the usual spot, plus a few others for troops and defensive buildings."""
    c = view.db.cards[name]
    me = view.me
    out = [place(view, name, lane)]
    if (c.type == "spell" and not c.summons) or name in DROP_ON_TOWER:
        return out
    if name in SIEGE or c.role == "spawner":
        return [off_towers(view.sim, me, *out[0])]
    tc = 1.0 if lane == 0 else -1.0
    if c.type == "building":
        out += [(WIDTH / 2 - tc * 1.5, abs_y(me, 6.0)), (WIDTH / 2 - tc * 1.5, abs_y(me, 11.0)),
                (WIDTH / 2, abs_y(me, 8.0))]
        return [off_towers(view.sim, me, *s) for s in out]
    threats = view.threats(lane)
    if threats:
        lead = min(threats, key=lambda u: view.fy(u.y))
        lfy = min(view.fy(lead.y), 14.0)
        x = float(np.clip(lead.x, 0.5, WIDTH - 0.5))
        out += [(x, abs_y(me, max(0.5, lfy - 1.0))), (x, abs_y(me, max(0.5, lfy - 4.0))),
                (WIDTH / 2 - tc * 1.0, abs_y(me, max(0.5, min(lfy - 3.0, 9.0)))),   # pull towards the middle
                (LANE_X[lane] + tc * 1.5, abs_y(me, 4.5))]                           # beside the tower
    out = [off_towers(view.sim, me, *s) for s in out]
    return [s for i, s in enumerate(out) if all(math.dist(s, o) > 0.25 for o in out[:i])]


class _Without(View):
    """The same view with some cards treated as not playable."""

    def __init__(self, view: View, cards: set[str]):
        self.__dict__.update(view.__dict__)
        self._without = set(cards)

    def playable(self) -> list[str]:
        return [c for c in super().playable() if c not in self._without]


class Lookahead(Coach):
    """:class:`Coach` whose defence (card, spot, or waiting) is chosen by simulating the options."""

    name = "lookahead"

    def __init__(self, placer=None, reasons=("defend", "hold"), **params):
        look = {k: params.pop(k) for k in list(params) if k in LOOK_DEFAULTS}
        super().__init__(placer=placer, **params)
        self.look = {**LOOK_DEFAULTS, **look}
        self.reasons = set(reasons)
        self.rollouts = 0
        self.tried: set[str] = set()       # cards played out by the last plan()

    def plan(self, view: View):
        card, lane, reason = super().plan(view)
        self.spot, self.tried = None, set()
        if reason == "trade" and self.look["trades"]:
            h = self.look["horizon"]
            cast = rollout(view.sim, view.me, (card, *place(view, card, lane)), h)
            wait = rollout(view.sim, view.me, None, h)
            self.rollouts += 2
            if cast is None or position_value(cast, view.me, **self.look) <= position_value(wait, view.me, **self.look):
                return self.plan(_Without(view, getattr(view, "_without", set()) | {card}))
            return card, lane, reason
        if reason not in self.reasons:
            return card, lane, reason
        play = view.playable()
        if reason == "hold":
            cands = [n for n in play if view.db.cards[n].role != "win_condition"]
        else:
            prof = view.profile(view.threats(lane))
            cands = [n for n in play if self.card_fit(view, n, prof)]
        if card is not None and card not in cands:
            cands.append(card)
        self.tried = set(cands)
        options = [(None, None)] + [(n, xy) for n in cands for xy in spots(view, n, lane)]
        best, best_v = None, -math.inf
        for n, xy in options:
            r = rollout(view.sim, view.me, None if n is None else (n, *xy), self.look["horizon"])
            self.rollouts += 1
            if r is None:
                continue
            v = position_value(r, view.me, **self.look) - (0.0 if n is None else self.look["wait_margin"])
            if v > best_v + 1e-9:
                best, best_v = (n, xy), v
        if best is None:
            return card, lane, reason
        if best[0] is None:
            return None, lane, "hold"
        self.spot = best
        return best[0], lane, reason
