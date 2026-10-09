"""Batched full-game environment for the fly brain (always player 0).

Observations are binary situation channels (each drives a group of PNs):
elixir, which card roles are in hand / ready, per-lane threats by type,
own pushes, tower health, tempo (double elixir, enemy elixir estimate) and
spell value. The brain answers with action heads, in one of two layouts:

  HEADS (first brains):  role (0 = wait, 1..15 = play a card of ROLES[role - 1]); lane (0 left, 1 right)
  SPLIT_HEADS:           play (0 = wait, 1 = play); role (0..14 = ROLES[i]); lane

The split layout keeps "when to play" apart from "what to play", so the
many waits don't drown out the card choice.

``pick_card`` turns (role, lane) into the best card in hand with that role,
and ``strategy.place`` into a drop spot.
"""

from __future__ import annotations

import numpy as np

from .db import ROLES, load
from .decks import hits_air, random_deck
from .sim import Sim
from .strategy import Coach, View, auto_ability, place, spell_spot

N_ROLES = len(ROLES)
HEADS = (N_ROLES + 1, 2)
SPLIT_HEADS = (2, N_ROLES, 2)
ELIXIR_EDGES = (2, 3, 4, 5, 7, 9)


def _channel_names() -> list[str]:
    n = [f"elixir {a}-{b}" for a, b in ((0, 2), (2, 3), (3, 4), (4, 5), (5, 7), (7, 9), (9, 10))]
    n += [f"{r} ready" for r in ROLES] + [f"{r} in hand" for r in ROLES]
    for s in ("left", "right"):
        n += [f"{s}: no threat", f"{s}: small threat", f"{s}: medium threat", f"{s}: big threat"]
        n += [f"{s}: enemy {k}" for k in ("building-targeter", "air", "swarm", "splash", "ranged", "tank killer",
                                           "building")]
        n += [f"{s}: threat at my tower", f"{s}: I'm defending"]
        n += [f"{s}: no push", f"{s}: small push", f"{s}: strong push", f"{s}: my win condition pushing"]
        n += [f"{s}: my tower low", f"{s}: my tower down", f"{s}: enemy tower low", f"{s}: enemy tower down"]
        n += [f"{s}: enemy tank building at back"]
    n += ["double/triple elixir", "overtime", "enemy low on elixir", "enemy high on elixir",
          "big spell has value", "small spell has value", "my champion on field", "anti-air in hand"]
    return n


CHANNELS = _channel_names()
N_CHANNELS = len(CHANNELS)


def features(view: View) -> np.ndarray:
    db = view.db
    f = np.zeros(N_CHANNELS, dtype=np.float32)
    i = 0
    f[i + int(np.searchsorted(ELIXIR_EDGES, view.elixir, side="right"))] = 1
    i += 7
    playable = set(view.playable())
    for n in view.hand():
        r = ROLES.index(db.cards[n].role)
        f[i + N_ROLES + r] = 1
        if n in playable:
            f[i + r] = 1
    i += 2 * N_ROLES
    for lane in (0, 1):
        th = view.threats(lane)
        prof = view.profile(th)
        f[i + int(np.searchsorted((1, 700, 2000), prof["hp"], side="right"))] = 1
        for k, key in enumerate(("building_targeter", "air", "swarm", "splash", "ranged", "tank_killer", "building")):
            f[i + 4 + k] = prof[key]
        f[i + 11] = any(view.fy(u.y) < 10 for u in th)
        f[i + 12] = bool(view.defenders(lane))
        push = view.pushers(lane)
        f[i + 13 + int(np.searchsorted((1, 1000), sum(u.hp for u in push), side="right"))] = 1
        f[i + 16] = any(u.card and db.cards[u.card].role == "win_condition" for u in push)
        mine, theirs = view.tower_frac(True, lane), view.tower_frac(False, lane)
        f[i + 17] = 0 < mine < 0.5
        f[i + 18] = mine == 0
        f[i + 19] = 0 < theirs < 0.5
        f[i + 20] = theirs == 0
        f[i + 21] = view.enemy_back_tank() == lane
        i += 22
    f[i] = view.sim.elixir_multiplier > 1
    f[i + 1] = view.sim.overtime
    f[i + 2] = view.enemy_elixir() <= 3
    f[i + 3] = view.enemy_elixir() >= 8
    big = [n for n in view.hand() if db.cards[n].role == "big_spell"]
    small = [n for n in view.hand() if db.cards[n].role == "small_spell"]
    f[i + 4] = any(spell_spot(view, n)[0] >= db.cards[n].elixir + 0.5 for n in big)
    f[i + 5] = any(spell_spot(view, n)[0] >= db.cards[n].elixir for n in small)
    f[i + 6] = view.sim.champion_alive(view.me)
    f[i + 7] = any(hits_air(db.cards[n]) and db.cards[n].type != "spell" for n in view.hand())
    return f


def role_mask(view: View) -> np.ndarray:
    m = np.zeros(HEADS[0], dtype=bool)
    m[0] = True
    for n in view.playable():
        m[1 + ROLES.index(view.db.cards[n].role)] = True
    return m


def brain_masks(view: View, heads=HEADS, no_wait: bool = False) -> list[np.ndarray]:
    """Allowed options per head for one situation (``no_wait``: must play if any card can be played)."""
    m = role_mask(view)
    if no_wait and m[1:].any():
        m[0] = False
    if tuple(heads) == SPLIT_HEADS:
        return [np.array([m[0], m[1:].any()]), m[1:] if m[1:].any() else np.ones(N_ROLES, bool), np.ones(2, bool)]
    return [m, np.ones(2, bool)]


def decode(action, heads=HEADS) -> tuple[int, int]:
    """A brain action in either layout -> (role index with 0 = wait, lane)."""
    a = [int(x) for x in np.asarray(action).ravel()]
    if tuple(heads) == SPLIT_HEADS:
        return (1 + a[1] if a[0] == 1 else 0), a[2]
    return a[0], a[1]


def encode(acts: np.ndarray, active: np.ndarray, heads=HEADS) -> tuple[np.ndarray, np.ndarray]:
    """Teacher actions (role with 0 = wait, lane) and active flags -> the given layout."""
    if tuple(heads) != SPLIT_HEADS:
        return acts, active
    play = acts[:, 0] > 0
    return (np.stack([play.astype(np.int64), np.maximum(acts[:, 0] - 1, 0), acts[:, 1]], 1),
            np.stack([np.ones(len(acts), bool), play, active[:, 1] & play], 1))


def pick_card(view: View, role: str) -> str | None:
    """The card in hand with this role that fits the situation best."""
    db = view.db
    cands = [n for n in view.playable() if db.cards[n].role == role]
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    coach = Coach()
    suggested, _ = coach.suggest(view)
    if suggested in cands:            # same role as the coach's pick: use its choice of card
        return suggested
    threats = [u for lane in (0, 1) for u in view.threats(lane)]
    prof = view.profile(threats)
    fit = [n for n in cands if coach.card_fit(view, n, prof)] or cands
    return min(fit, key=lambda n: db.cards[n].elixir)


class RoyaleEnv:
    n_channels = N_CHANNELS

    def __init__(self, batch: int, opponent_factory, fly_deck: list[str] | None = None, fly_deck_share: float = 0.3,
                 decision_every: float = 1.0, seed: int = 0, leak_penalty: float = 0.02, damage_weight: float = 0.5,
                 guard: bool = False, heads=HEADS):
        self.batch = batch
        self.opponent_factory = opponent_factory
        self.fly_deck = fly_deck
        self.fly_deck_share = fly_deck_share
        self.decision_every = decision_every
        self.leak_penalty, self.damage_weight = leak_penalty, damage_weight
        self.guard = guard            # coach guard on the brain's moves (see guard.py)
        self.heads = tuple(heads)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.db = load()
        self.coach = Coach()
        self.episode = 0

    def _deck(self) -> list[str]:
        if self.fly_deck is not None and self.rng.random() < self.fly_deck_share:
            return list(self.fly_deck)
        return random_deck(self.rng, self.db)

    def reset(self) -> np.ndarray:
        base = self.seed * 1_000_003 + self.episode * self.batch
        self.sims = [Sim((self._deck(), random_deck(self.rng, self.db)), seed=base + i, db=self.db)
                     for i in range(self.batch)]
        self.opponents = [self.opponent_factory(base + i) for i in range(self.batch)]
        self.episode += 1
        self._prev = [self._score(s) for s in self.sims]
        self.done = np.zeros(self.batch, dtype=bool)
        return self.observe()

    def views(self) -> list[View]:
        return [View(s, 0) for s in self.sims]

    def observe(self) -> np.ndarray:
        return np.stack([features(v) for v in self.views()])

    def masks(self) -> list[np.ndarray]:
        per = [brain_masks(v, self.heads) for v in self.views()]
        return [np.stack([p[h] for p in per]) for h in range(len(self.heads))]

    def teacher(self) -> tuple[np.ndarray, np.ndarray]:
        acts = np.zeros((self.batch, 2), dtype=np.int64)
        active = np.ones((self.batch, 2), dtype=bool)
        for i, v in enumerate(self.views()):
            card, lane = self.coach.suggest(v)
            acts[i] = (0 if card is None else 1 + ROLES.index(self.db.cards[card].role), lane)
            active[i, 1] = card is not None
        return encode(acts, active, self.heads)

    @staticmethod
    def _score(sim: Sim):
        lost = [sum(t.max_hp - max(t.hp, 0) for t in sim.towers(o)) / 3052.0 for o in (0, 1)]
        return lost[1], lost[0], sim.players[0].crowns, sim.players[1].crowns, sim.players[0].leaked

    def step(self, actions: np.ndarray, record: bool = False, on_tick=None):
        actions = np.asarray(actions).reshape(self.batch, len(self.heads))
        reward = np.zeros(self.batch, dtype=np.float32)
        self.last_cards = [None] * self.batch
        for i, sim in enumerate(self.sims):
            if self.done[i]:
                continue
            opp = self.opponents[i].decide(sim, 1)
            role, lane = decode(actions[i], self.heads)
            if self.guard:
                from .guard import guarded_action

                view = View(sim, 0)
                card, lane, _ = guarded_action(view, role, lane, self.coach)
                if card is not None:
                    x, y = place(view, card, lane)
                    if sim.play(0, card, x, y):
                        self.last_cards[i] = card
            elif role > 0:
                view = View(sim, 0)
                card = pick_card(view, ROLES[role - 1])
                if card is not None:
                    x, y = place(view, card, lane)
                    if sim.play(0, card, x, y):
                        self.last_cards[i] = card
            if opp is not None:
                sim.play(1, opp.card, opp.x, opp.y)
            auto_ability(sim, 0)
            auto_ability(sim, 1)
            for _ in range(int(round(self.decision_every / sim.dt))):
                sim.step()
                if on_tick is not None:
                    on_tick(i, sim)
                if sim.done:
                    break
            dealt, taken, c0, c1, leaked = self._score(sim)
            pd, pt, pc0, pc1, pl = self._prev[i]
            r = self.damage_weight * ((dealt - pd) - (taken - pt)) + 0.3 * ((c0 - pc0) - (c1 - pc1))
            r -= self.leak_penalty * (leaked > pl)
            if sim.done:
                self.done[i] = True
                r += {0: 1.0, 1: -1.0, None: 0.0}[sim.winner]
            reward[i] = float(np.clip(r, -1, 1))
            self._prev[i] = (dealt, taken, c0, c1, leaked)
        return self.observe(), reward, bool(self.done.all())

    def results(self) -> np.ndarray:
        return np.array([{0: 1, 1: -1, None: 0}[s.winner] for s in self.sims])
