"""Decks, placement, coach and the brain's interface for the full card pool."""

import numpy as np
import pytest

from flybrain.envs.royale.db import ROLES, load
from flybrain.envs.royale.decks import (DEFAULT_FLY_DECK, check_deck, hits_air, is_splash, load_deck, parse_deck,
                                        random_deck)
from flybrain.envs.royale.env import CHANNELS, HEADS, N_CHANNELS, RoyaleEnv, features, pick_card, role_mask
from flybrain.envs.royale.sim import Sim
from flybrain.envs.royale.strategy import BasicBot, Coach, RandomBot, View, place, play_match

DB = load()


def test_parse_deck_with_loose_names():
    assert parse_deck("pekka, mini pekka, log, e-wiz\nxbow  # siege\ngy, bats, zap") == [
        "P.E.K.K.A", "Mini P.E.K.K.A", "The Log", "Electro Wizard", "X-Bow", "Graveyard", "Bats", "Zap"]


def test_deck_rules():
    with pytest.raises(ValueError):
        check_deck(DEFAULT_FLY_DECK[:7])
    with pytest.raises(ValueError):
        check_deck(DEFAULT_FLY_DECK[:7] + ["Hog Rider"])
    with pytest.raises(ValueError):
        check_deck(["Archer Queen", "Golden Knight"] + DEFAULT_FLY_DECK[:6])
    with pytest.raises(ValueError):
        parse_deck("Hog Rider, Not A Real Card")


def test_fly_deck_file_is_valid():
    from pathlib import Path
    check_deck(load_deck(Path(__file__).resolve().parents[1] / "decks/fly.txt"))


def test_random_decks_are_sensible():
    rng = np.random.default_rng(0)
    for _ in range(200):
        deck = random_deck(rng)
        check_deck(deck)
        cards = [DB.cards[n] for n in deck]
        assert sum(c.role == "win_condition" for c in cards) >= 1
        assert sum(c.type == "spell" for c in cards) >= 2
        assert sum(hits_air(c) and c.type != "spell" for c in cards) >= 2
        assert any(is_splash(c) for c in cards)


@pytest.mark.parametrize("player", [0, 1])
def test_every_card_has_a_legal_placement(player):
    pool = DB.pool()
    deck = random_deck(np.random.default_rng(1))
    sim = Sim((deck, deck), seed=0)
    for _ in range(80):        # some troops on the field so placement uses threat logic too
        sim.step()
    sim._deploy(1 - player, DB.cards["Hog Rider"], 9.0, 12.0 if player == 1 else 20.0)
    view = View(sim, player)
    for name in pool:
        if DB.cards[name].special == "mirror":
            continue
        for lane in (0, 1):
            x, y = place(view, name, lane)
            assert sim.valid_position(player, name, x, y), (name, lane, x, y)


def test_features_and_masks():
    sim = Sim((DEFAULT_FLY_DECK, DEFAULT_FLY_DECK), seed=0)
    v = View(sim, 0)
    f = features(v)
    assert f.shape == (N_CHANNELS,) == (len(CHANNELS),)
    assert set(np.unique(f)) <= {0.0, 1.0}
    m = role_mask(v)
    assert m.shape == (HEADS[0],) and m[0]
    for n in v.playable():
        assert m[1 + ROLES.index(DB.cards[n].role)]
        assert pick_card(v, DB.cards[n].role) is not None


def test_coach_beats_random_bot():
    rng = np.random.default_rng(5)
    wins = [play_match(Coach(), RandomBot(i), (random_deck(rng), random_deck(rng)), seed=i).winner for i in range(8)]
    assert wins.count(0) >= 6


def test_env_runs_a_match_with_the_coach():
    env = RoyaleEnv(2, lambda s: BasicBot(s), fly_deck=DEFAULT_FLY_DECK, fly_deck_share=1.0, decision_every=2.0)
    f, done = env.reset(), False
    assert env.sims[0].decks[0] == DEFAULT_FLY_DECK
    while not done:
        a, _ = env.teacher()
        f, r, done = env.step(a)
    assert set(env.results()) <= {-1, 0, 1}
