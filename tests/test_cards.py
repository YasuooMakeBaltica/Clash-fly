"""Recognising any card from card art, and learning the deck from the hand (no network: fake card art)."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from flybrain.envs.royale.db import load
from flybrain.envs.royale.decks import DEFAULT_FLY_DECK, check_deck, random_deck
from flybrain.envs.royale.sim import Sim
from flybrain.envs.royale.strategy import BasicBot
from flybrain.real.cards import DeckTracker, OfficialMatcher, fill_deck
from flybrain.real.layout import Layout
from flybrain.real.perception import Perception
from flybrain.real.synthetic import render, slot_picture, write_templates

ROOT = Path(__file__).resolve().parents[1]
POOL = load().pool()


@pytest.fixture(scope="module")
def official(tmp_path_factory):
    d = tmp_path_factory.mktemp("official")
    write_templates(d, POOL, full=True)
    return d


def slot_crop(name, rng, size=(89, 106), grey=False):
    img = cv2.resize(slot_picture(name), size, interpolation=cv2.INTER_AREA)
    if grey:                                        # cards you can't afford are shown in grey
        img = (cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) * 0.6).astype(np.uint8)
    img = np.clip(img * rng.uniform(0.85, 1.15) + rng.uniform(-15, 15), 0, 255).astype(np.uint8)
    _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return cv2.imdecode(enc, 1)


def test_matcher_recognises_every_card(official):
    m = OfficialMatcher.from_dir(official)
    assert len(m.names) == len(POOL)
    rng = np.random.default_rng(0)
    for grey in (False, True):
        got = [m.match(slot_crop(n, rng, grey=grey))[0] for n in POOL]
        assert sum(g == n for g, n in zip(got, POOL)) / len(POOL) > 0.95
        assert all(g in (None, n) for g, n in zip(got, POOL))     # unsure is fine, wrong is not


def test_matcher_rejects_non_cards(official):
    m = OfficialMatcher.from_dir(official)
    assert m.match(np.full((106, 89, 3), 40, np.uint8))[0] is None
    assert m.match(np.zeros((0, 0, 3), np.uint8))[0] is None


def test_deck_tracker_and_fill():
    deck = random_deck(np.random.default_rng(3))
    t = DeckTracker(min_sightings=2)
    assert t.complete() is None
    for i in range(12):
        hand = [deck[(i + k) % 8] for k in range(4)]
        t.update(hand, deck[(i + 4) % 8])
    t.update(["Knight", None, None, None], None)                 # a one-off misread is ignored
    assert sorted(t.complete()) == sorted(deck)
    partial = fill_deck(deck[:3], deck[3:5])
    check_deck(partial)
    assert set(deck[:5]) <= set(partial)


def test_perception_reads_any_card_without_saved_pictures(official, tmp_path):
    lay = Layout()
    per = Perception(lay, tmp_path / "none", official_dir=official)
    rng = np.random.default_rng(4)
    for seed in range(3):
        sim = Sim((random_deck(rng), random_deck(rng)), seed=seed)
        obs = per.read(render(sim, lay))
        assert obs.hand == sim.players[0].hand and obs.next_card == sim.players[0].queue[0]


def test_bot_learns_a_changed_deck(official, tmp_path):
    from flybrain.real.bot import Bot, load_brain
    from flybrain.real.fake import FakeAdb

    brain = ROOT / "models/fly_royale.pt"
    if not brain.exists():
        pytest.skip("trained brain not available")
    agent, _ = load_brain(str(brain))
    lay = Layout()
    per = Perception(lay, tmp_path / "none", official_dir=official)
    rng = np.random.default_rng(5)
    real_deck = random_deck(rng)                  # what's actually in the game, not what the deck file says
    sim = Sim((real_deck, random_deck(rng)), seed=7)
    adb = FakeAdb(sim, lay)
    bot = Bot(agent, per, lay, list(DEFAULT_FLY_DECK), adb=adb)
    opp = BasicBot(1)
    bot.start_battle(0.0)
    while sim.time < 120 and not sim.done and bot.learned_deck() is None:
        m = opp.decide(sim, 1)
        bot.step(adb.screencap(), sim.time)
        if m:
            sim.play(1, m.card, m.x, m.y)
        for _ in range(10):
            sim.step()
    assert sorted(bot.learned_deck()) == sorted(real_deck)
    assert adb.plays and all(ok for *_, ok in adb.plays)
