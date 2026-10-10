"""The lookahead coach: copies of the match, position values and simulated defence choices."""

import pytest

from flybrain.envs.royale.db import load
from flybrain.envs.royale.lookahead import Lookahead, char_value, clone, position_value, rollout
from flybrain.envs.royale.sim import Sim, Unit
from flybrain.envs.royale.strategy import View

DB = load()
DECK = ["Knight", "Archers", "Cannon", "Fireball", "Zap", "Hog Rider", "Skeletons", "Musketeer"]
FOE = ["Giant", "Hog Rider", "Minions", "Valkyrie", "Arrows", "Fireball", "Bomber", "Tesla"]


def match_with_hog(hand, elixir=7.0):
    sim = Sim((DECK, FOE), seed=3)
    me = sim.players[0]
    me.hand, me.queue, me.elixir = list(hand), [n for n in DECK if n not in hand], elixir
    sim.units.append(Unit(sim, DB.characters["HogRider"], 1, 3.5, 18.0, card="Hog Rider", deploy=0.0))
    return sim


def test_clone_is_independent_and_shares_card_data():
    sim = match_with_hog(["Knight", "Cannon", "Fireball", "Zap"])
    c = clone(sim)
    hog, hog_copy = sim.units[-1], c.units[-1]
    for _ in range(30):
        c.step()
    assert sim.time == 0 and c.time == pytest.approx(3.0)
    assert hog.y == 18.0 and hog_copy.y < 18.0                 # only the copy moved
    assert c.db is sim.db and hog_copy.spec is hog.spec


def test_position_values():
    sim = Sim((DECK, FOE), seed=1)
    assert position_value(sim, 0) == pytest.approx(5.0)                  # even start: just the elixir
    assert position_value(sim, 1) == pytest.approx(5.0)
    assert char_value(DB.characters["HogRider"], DB) == 4.0
    assert char_value(DB.characters["Skeleton"], DB) == pytest.approx(1 / 3)
    assert 5.0 < char_value(DB.characters["GoblinGiant"], DB) < 6.0   # most of the card, riders the rest
    assert 1.0 < char_value(DB.characters["Golemite"], DB) < 3.0       # a death spawn, by hit points
    sim.units.append(Unit(sim, DB.characters["HogRider"], 1, 3.5, 10.0, card="Hog Rider", deploy=0.0))
    assert position_value(sim, 0) == pytest.approx(1.0)                 # an enemy Hog Rider costs 4


def test_defends_a_hog_rider_better_than_waiting():
    sim = match_with_hog(["Knight", "Cannon", "Fireball", "Zap"])
    coach = Lookahead()
    view = View(sim, 0)
    card, lane, reason = coach.plan(view)
    assert reason == "defend" and lane == 0 and card in ("Knight", "Cannon")
    assert coach.rollouts > 3 and coach.spot[0] == card
    x, y = coach.where(view, card, lane)
    played, waited = rollout(sim, 0, (card, x, y), 8.0), rollout(sim, 0, None, 8.0)
    assert position_value(played, 0) > position_value(waited, 0)
    move = coach.decide(sim, 0)
    assert (move.card, move.x, move.y) == (card, x, y)                  # plays where it simulated
    assert sim.time == 0 and len(sim.units) == 7                        # the real match is untouched


def test_waits_when_the_tower_handles_it():
    sim = Sim((DECK, FOE), seed=3)
    sim.players[0].elixir = 7.0
    sim.units.append(Unit(sim, DB.characters["Skeleton"], 1, 3.5, 13.0, card="Skeletons", deploy=0.0))
    # with tower damage on the enemy side discounted (the copy's opponent never defends, so a troop
    # sent over the bridge looks free), nothing beats letting the tower shoot one Skeleton
    card, _, reason = Lookahead(offense=0.25).plan(View(sim, 0))
    assert card is None and reason == "hold"


def test_spots_stay_off_our_towers():
    from flybrain.envs.royale.lookahead import off_towers, spots

    sim = Sim((DECK, FOE), seed=1)
    assert off_towers(sim, 0, 4.5, 6.0) == (5.1, 6.0)                   # princess tower: 3x3 tiles
    assert off_towers(sim, 0, 9.0, 3.0) == (9.0, 5.1)                   # king tower: 4x4, nudged towards the river
    assert off_towers(sim, 0, 8.0, 9.0) == (8.0, 9.0)                   # open ground: unchanged
    assert off_towers(sim, 1, 9.0, 29.0) == (9.0, 26.9)
    sim = match_with_hog(["Knight", "Cannon", "Fireball", "Zap"])
    sim.units[-1].y = 7.0                                               # the Hog Rider is at our tower
    for name in ("Knight", "Cannon"):
        for x, y in spots(View(sim, 0), name, 0):
            assert not (2.0 < x < 5.0 and 5.0 < y < 8.0)
