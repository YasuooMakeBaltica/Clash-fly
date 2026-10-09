"""Turn a screen :class:`Observation` into a full-game simulator state.

The fly brain was trained on ``features(View(sim, 0))`` from the full
simulator. Rebuilding a ``Sim`` from what is on screen lets the real game
reuse the same features, role masks, card picks and placement. You are
player 0 (bottom). The opponent's deck is unknown (placeholder deck); their
elixir is counted from the troops they deploy (tracking.py) when troop types
are known, otherwise a middling guess.

With the troop detector (troops.py) every seen troop has its real type, so
the game state knows air from ground, tanks, building targeters and so on.
With the colour-badge fallback types are unknown, so seen troops become a
generic ground troop (or a group of small ones when several badges are
close together): lane and size are right, type is not.
"""

from __future__ import annotations

from ..envs.royale.db import DB, load
from ..envs.royale.sim import Sim, Unit
from .perception import Observation

ENEMY_ELIXIR_GUESS = 5.0
_CHAR_CARD: dict[str, str] = {}


def card_of(char: str, db: DB) -> str | None:
    """The (cheapest) card that summons this character, e.g. Skeleton -> Skeletons, HogRider -> Hog Rider.

    Units carry their card so the coach can value them (spell trades) and spot heavy tanks."""
    if not _CHAR_CARD:
        for name, c in sorted(db.cards.items(), key=lambda kv: (kv[1].elixir, kv[0])):
            if c.event or not c.role:
                continue
            for spec, _ in c.summons:
                _CHAR_CARD.setdefault(spec.name, name)
    return _CHAR_CARD.get(char)


def build_sim(obs: Observation, match_time: float, deck: list[str], elixir_margin: float = 0.15,
              db: DB | None = None, enemy_elixir: float | None = None) -> Sim:
    """``elixir_margin`` is taken off the elixir reading: the bar is read to
    about 0.1, and tapping a card the game says you can't afford yet wastes
    the decision."""
    db = db or load()
    sim = Sim((list(deck), list(deck)), seed=0, db=db)
    sim.time = match_time
    me, foe = sim.players
    me.elixir = max(0.0, obs.elixir - elixir_margin)
    me.hand = [n for n in obs.hand if n in deck]
    me.queue = [n for n in deck if n not in me.hand]
    if obs.next_card in me.queue:
        me.queue.remove(obs.next_card)
        me.queue.insert(0, obs.next_card)
    foe.hand, foe.queue = [], []
    foe.elixir = ENEMY_ELIXIR_GUESS if enemy_elixir is None else enemy_elixir
    for t in sim.towers():
        frac = obs.towers.get((t.owner, t.tower, t.lane), 1.0)
        t.hp = frac * t.max_hp
        if frac <= 0:
            t.alive = False
            sim.units.remove(t)
            sim._dead_towers = getattr(sim, "_dead_towers", []) + [t]
        if t.tower == "king":
            t.active = frac < 1.0
    generic, small = db.characters["Knight"], db.characters["Goblin"]
    for u in obs.units:
        if u.char in db.characters:
            spec, card, n = db.characters[u.char], card_of(u.char, db), 1
        elif u.card in db.cards and db.cards[u.card].summons:
            spec, card, n = db.cards[u.card].summons[0][0], u.card, 1
        elif u.size >= 3:
            spec, card, n = small, None, u.size
        else:
            spec, card, n = generic, None, 1
        for _ in range(n):
            unit = Unit(sim, spec, u.owner, u.x, u.y, card=card, deploy=0.0)
            sim.units.append(unit)
    return sim
