"""Batched Clash Royale environment for the fly brain (always player 0).

Observations are binary "situation" channels (each drives a group of PNs),
already digested into the things a player looks at: elixir, cards in hand,
threats per lane and what kind, own pushes, tower health. Raw coordinates
would force the mushroom body to learn geometry, which it is bad at.

Actions have two heads:
  card: 0 = wait, 1..8 = play DECK[card - 1]
  lane: 0 = left, 1 = right
Exact placement within the lane is done by :func:`strategy.place`, the same
helper the coach uses.

Both players decide from the same state each ``decision_every`` seconds.
"""

from __future__ import annotations

import numpy as np

from .cards import DECK, PRINCESS_HP
from .sim import ClashSim
from .strategy import Coach, View, best_spell_spot, place


def _channel_names() -> list[str]:
    names = [f"elixir {lo}-{hi}" for lo, hi in ((0, 2), (2, 3), (3, 4), (4, 5), (5, 7), (7, 9), (9, 10))]
    names += [f"{c.name} ready" for c in DECK]
    names += [f"{c.name} in hand" for c in DECK]
    for side in ("left", "right"):
        names += [f"{side}: no threat", f"{side}: small threat", f"{side}: medium threat", f"{side}: big threat"]
        names += [f"{side}: enemy tank", f"{side}: enemy melee", f"{side}: enemy ranged", f"{side}: enemy swarm"]
        names += [f"{side}: threat at my tower", f"{side}: I'm defending"]
        names += [f"{side}: no push", f"{side}: small push", f"{side}: strong push"]
        names += [f"{side}: my tower low", f"{side}: my tower down",
                  f"{side}: enemy tower low", f"{side}: enemy tower down"]
        names += [f"{side}: enemy tank building at back"]
    names += ["double elixir", "fireball has value", "arrows has value"]
    return names


CHANNELS = _channel_names()
N_CHANNELS = len(CHANNELS)
HEADS = (len(DECK) + 1, 2)
ELIXIR_EDGES = (2, 3, 4, 5, 7, 9)
FIREBALL, ARROWS = DECK[6], DECK[7]


def features(view: View) -> np.ndarray:
    f = np.zeros(N_CHANNELS, dtype=np.float32)
    i = 0
    f[i + int(np.searchsorted(ELIXIR_EDGES, view.elixir, side="right"))] = 1
    i += 7
    for c in view.p.hand:
        f[i + c] = DECK[c].cost <= view.elixir
        f[i + len(DECK) + c] = 1
    i += 2 * len(DECK)
    for lane in (0, 1):
        threats = view.threats(lane)
        hp = sum(u.hp for u in threats)
        f[i + int(np.searchsorted((1, 700, 2000), hp, side="right"))] = 1
        roles = {u.card.role for u in threats}
        f[i + 4] = "tank" in roles
        f[i + 5] = bool(roles & {"tank_killer", "mini_tank"})
        f[i + 6] = "ranged" in roles
        f[i + 7] = "swarm" in roles
        f[i + 8] = any(view.fy(u.y) < 10 for u in threats)
        f[i + 9] = bool(view.defenders(lane))
        push = sum(u.hp for u in view.pushers(lane))
        f[i + 10 + int(np.searchsorted((1, 1000), push, side="right"))] = 1
        mine = view.tower_frac(True, lane)
        theirs = view.tower_frac(False, lane)
        f[i + 13] = 0 < mine < 0.5
        f[i + 14] = mine == 0
        f[i + 15] = 0 < theirs < 0.5
        f[i + 16] = theirs == 0
        f[i + 17] = view.enemy_back_tank() == lane
        i += 18
    f[i] = view.sim.double_elixir
    f[i + 1] = best_spell_spot(view, FIREBALL)[0] >= FIREBALL.cost
    f[i + 2] = best_spell_spot(view, ARROWS)[0] >= ARROWS.cost
    return f


def masks(view: View) -> tuple[np.ndarray, np.ndarray]:
    card = np.zeros(HEADS[0], dtype=bool)
    card[0] = True
    for c in view.p.hand:
        card[c + 1] = DECK[c].cost <= view.elixir
    return card, np.ones(2, dtype=bool)


def _tower_loss(sim: ClashSim, owner: int) -> float:
    """Total tower HP lost by ``owner``, in units of one princess tower."""
    return sum(t.max_hp - t.hp for t in sim.towers if t.owner == owner) / PRINCESS_HP


class ClashEnv:
    heads = HEADS
    n_channels = N_CHANNELS

    def __init__(self, batch: int, opponent_factory, decision_every: float = 1.0, seed: int = 0,
                 leak_penalty: float = 0.02, damage_weight: float = 0.5):
        self.batch = batch
        self.opponent_factory = opponent_factory
        self.decision_every = decision_every
        self.leak_penalty = leak_penalty
        self.damage_weight = damage_weight
        self.seed = seed
        self.coach = Coach()
        self.episode = 0

    def reset(self) -> np.ndarray:
        base = self.seed * 1_000_003 + self.episode * self.batch
        self.sims = [ClashSim(seed=base + i) for i in range(self.batch)]
        self.opponents = [self.opponent_factory(base + i) for i in range(self.batch)]
        self.episode += 1
        self._prev = [self._score(s) for s in self.sims]
        self.done = np.zeros(self.batch, dtype=bool)
        self.history: list[list[dict]] = [[] for _ in range(self.batch)]
        return self.observe()

    def views(self) -> list[View]:
        return [View(s, 0) for s in self.sims]

    def observe(self) -> np.ndarray:
        return np.stack([features(v) for v in self.views()])

    def masks(self) -> list[np.ndarray]:
        ms = [masks(v) for v in self.views()]
        return [np.stack([m[0] for m in ms]), np.stack([m[1] for m in ms])]

    def teacher(self) -> tuple[np.ndarray, np.ndarray]:
        """Coach's move for the brain's side: (batch, 2) actions and which heads apply."""
        acts = np.zeros((self.batch, 2), dtype=np.int64)
        active = np.ones((self.batch, 2), dtype=bool)
        for i, v in enumerate(self.views()):
            card, lane = self.coach.suggest(v)
            acts[i] = (0 if card is None else card + 1, lane)
            active[i, 1] = card is not None  # lane is meaningless when waiting
        return acts, active

    def _score(self, sim: ClashSim) -> tuple[float, float, int, int, float]:
        p0, p1 = sim.players
        return _tower_loss(sim, 1), _tower_loss(sim, 0), p0.crowns, p1.crowns, p0.leaked

    def step(self, actions: np.ndarray, record: bool = False, on_tick=None):
        """Apply brain actions, advance one decision interval. Returns (obs, reward, done).

        ``on_tick(i, sim)`` is called after every simulation tick (for replays).
        """
        actions = np.asarray(actions).reshape(self.batch, 2)
        reward = np.zeros(self.batch, dtype=np.float32)
        for i, sim in enumerate(self.sims):
            if self.done[i]:
                continue
            opp_move = self.opponents[i].decide(sim, 1)
            card, lane = int(actions[i, 0]), int(actions[i, 1])
            played = False
            if card > 0 and sim.can_play(0, card - 1):
                x, y = place(View(sim, 0), card - 1, lane)
                played = sim.play(0, card - 1, x, y)
            if opp_move is not None:
                sim.play(1, opp_move.card, opp_move.x, opp_move.y)
            if record:
                self.history[i].append(dict(t=round(sim.time, 2), card=card, lane=lane, played=played))
            for _ in range(int(round(self.decision_every / sim.dt))):
                sim.step()
                if on_tick is not None:
                    on_tick(i, sim)
                if sim.done:
                    break
            dealt, taken, c0, c1, leaked = self._score(sim)
            pd, pt, pc0, pc1, pl = self._prev[i]
            r = self.damage_weight * ((dealt - pd) - (taken - pt))
            r += 0.3 * ((c0 - pc0) - (c1 - pc1))
            r -= self.leak_penalty * (leaked > pl)
            if sim.done:
                self.done[i] = True
                r += {0: 1.0, 1: -1.0, None: 0.0}[sim.winner]
            reward[i] = float(np.clip(r, -1, 1))
            self._prev[i] = (dealt, taken, c0, c1, leaked)
        return self.observe(), reward, bool(self.done.all())

    def results(self) -> np.ndarray:
        """Per match: +1 win, 0 draw, -1 loss (for finished matches)."""
        return np.array([{0: 1, 1: -1, None: 0}[s.winner] for s in self.sims])
