"""Counting the opponent's elixir from the troops they deploy."""

import pytest

from flybrain.real.perception import SeenUnit
from flybrain.real.tracking import EnemyElixir


def enemy(char, x, y):
    return SeenUnit(owner=1, x=x, y=y, size=1, char=char)


def test_counts_deployments_once_and_regenerates():
    t = EnemyElixir(start=5.0)
    assert t.update([], 0.0) == 5.0
    skel = [enemy("Skeleton", 4 + 0.4 * i, 20) for i in range(3)]
    e = t.update(skel, 1.0)
    assert e == pytest.approx(5.0 + 1 / 2.8 - 1.0)              # Skeletons cost 1
    assert t.plays[-1][1] == "Skeletons"
    moved = [enemy("Skeleton", 4 + 0.4 * i, 19) for i in range(3)]
    assert t.update(moved, 2.0) == pytest.approx(e + 1 / 2.8)    # same troops walking: not a new card
    e = t.update(moved + [enemy("HogRider", 14, 18)], 3.0)
    assert t.plays[-1][1] == "Hog Rider"
    assert e == pytest.approx(5.0 + 3 / 2.8 - 1.0 - 4.0)


def test_group_size_picks_the_card():
    t = EnemyElixir()
    t.update([enemy("Skeleton", 3 + (i % 5) * 0.5, 22 + (i // 5) * 0.5) for i in range(14)], 1.0)
    assert t.plays[-1][1] == "Skeleton Army"


def test_free_troops_cost_nothing():
    t = EnemyElixir(start=5.0)
    t.update([], 0.0)
    t.update([enemy("Tombstone", 9, 20)], 1.0)
    e = t.update([enemy("Tombstone", 9, 20), enemy("Skeleton", 9, 19)], 2.0)
    assert t.plays[-1][1] == "Tombstone" and e == pytest.approx(5.0 + 2 / 2.8 - 3.0)
    e2 = t.update([enemy("Golemite", 4, 12), enemy("Golemite", 5, 12)], 3.0)   # a Golem died: no card
    assert e2 == pytest.approx(e + 1 / 2.8)


def test_a_missed_frame_is_not_a_new_card():
    t = EnemyElixir(start=5.0)
    t.update([], 0.0)
    t.update([enemy("Giant", 4, 22)], 1.0)
    t.update([], 2.0)                                            # detector missed it once
    e = t.update([enemy("Giant", 4, 20.5)], 3.0)
    assert [p[1] for p in t.plays] == ["Giant"]
    assert e == pytest.approx(5.0 + 1 / 2.8 - 5.0 + 2 / 2.8)


def test_never_negative_and_capped():
    t = EnemyElixir(start=1.0)
    assert t.update([enemy("Golem", 9, 30)], 0.5) == 0.0
    assert t.update([], 200.0) == 10.0
