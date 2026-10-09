"""Turn a screen :class:`Observation` into a simulator state.

The fly brain was trained on ``features(View(sim, 0))``. Rebuilding a
``ClashSim`` from what is on screen lets the real game reuse the exact same
features, card masks and lane placement code. You are player 0 (bottom).

Troop types are unknown with the badge detector, so seen troops get a
generic card (``UNKNOWN``); the threat-size and lane channels still work,
the threat-type channels (tank / swarm / ...) need a troop classifier.
"""

from __future__ import annotations

from ..envs.clash.cards import CARD_INDEX, DECK, Card
from ..envs.clash.sim import ClashSim, Unit
from .perception import Observation

UNKNOWN = Card("Unknown troop", 3, "troop", "mini_tank", hp=900, damage=150, hit_speed=1.2, range=1.0)
SWARM_GUESS = Card("Unknown swarm", 2, "troop", "swarm", count=1, hp=250, damage=100, hit_speed=1.1, range=0.5)


def build_sim(obs: Observation, match_time: float, elixir_margin: float = 0.15) -> ClashSim:
    """``elixir_margin`` is taken off the elixir reading: the bar is read to
    about 0.1, and tapping a card the game says you can't afford yet wastes
    the decision."""
    sim = ClashSim(seed=0)
    sim.time = match_time
    me, foe = sim.players
    me.elixir = max(0.0, obs.elixir - elixir_margin)
    me.hand = [CARD_INDEX[n] for n in obs.hand if n in CARD_INDEX]
    me.queue = [i for i in range(len(DECK)) if i not in me.hand]
    if obs.next_card in CARD_INDEX and CARD_INDEX[obs.next_card] in me.queue:
        me.queue.remove(CARD_INDEX[obs.next_card])
        me.queue.insert(0, CARD_INDEX[obs.next_card])
    foe.hand, foe.queue = [], []
    for t in sim.towers:
        frac = obs.towers.get((t.owner, t.kind, t.lane), 1.0)
        t.hp = frac * t.max_hp
        if t.kind == "king":
            t.active = frac < 1.0
    sim.units = []
    for i, u in enumerate(obs.units):
        if u.card in CARD_INDEX:
            card, idx = DECK[CARD_INDEX[u.card]], CARD_INDEX[u.card]
        else:
            card, idx = (SWARM_GUESS if u.size >= 3 else UNKNOWN), -1
        for _ in range(max(1, u.size if card is SWARM_GUESS else 1)):
            sim.units.append(Unit(i + 1, u.owner, card, idx, u.x, u.y, card.hp, deploy_left=0.0))
    return sim
