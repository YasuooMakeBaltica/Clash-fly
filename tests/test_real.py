"""Real-game pipeline on fake screenshots: perception, state rebuild, and the bot loop.

These check the logic, not the default screen positions (those need a real
screenshot and `python -m flybrain.real.calibrate`).
"""

from pathlib import Path

import numpy as np
import pytest

from flybrain.envs.royale.decks import DEFAULT_FLY_DECK, random_deck
from flybrain.envs.royale.env import features
from flybrain.envs.royale.sim import Sim
from flybrain.envs.royale.strategy import BasicBot, Coach, View, auto_ability
from flybrain.real.layout import Box, Layout
from flybrain.real.perception import Perception, in_battle
from flybrain.real.state import build_sim
from flybrain.real.synthetic import render, write_templates

ROOT = Path(__file__).resolve().parents[1]
DECK = list(DEFAULT_FLY_DECK)


@pytest.fixture
def perception(tmp_path):
    write_templates(tmp_path, DECK + ["Knight", "Archers", "Giant", "Goblins", "Arrows"])
    return Perception(Layout(), tmp_path)


def play(sim, seconds, bots=(Coach(), BasicBot(0))):
    for _ in range(int(seconds * 10)):
        if round(sim.time * 10) % 10 == 0:
            for p, b in enumerate(bots):
                m = b.decide(sim, p)
                if m:
                    sim.play(p, m.card, m.x, m.y)
                auto_ability(sim, p)
        sim.step()


def test_layout_roundtrip(tmp_path):
    lay = Layout()
    lay.arena = Box(0.01, 0.05, 0.99, 0.8)
    lay.save(tmp_path / "layout.json")
    back = Layout.load(tmp_path / "layout.json")
    assert back.arena == lay.arena and back.hand_slots == lay.hand_slots and back.ability_button == lay.ability_button
    for tx, ty in ((0, 0), (3.5, 6.5), (18, 32), (9, 16)):
        assert np.allclose(lay.frac_to_tile(*lay.tile_to_frac(tx, ty)), (tx, ty))


def test_reads_elixir_hand_and_towers(perception):
    rng = np.random.default_rng(0)
    sim = Sim((DECK, random_deck(rng)), seed=3)
    play(sim, 40)
    obs = perception.read(render(sim, perception.layout))
    assert abs(obs.elixir - sim.players[0].elixir) <= 0.15
    assert obs.hand == sim.players[0].hand
    assert obs.next_card == sim.players[0].queue[0]
    for t in sim.towers():
        assert abs(obs.towers[(t.owner, t.tower, t.lane)] - max(t.hp, 0) / t.max_hp) < 0.08


def test_not_in_battle_on_blank_screen(perception):
    assert not in_battle(np.zeros((960, 540, 3), np.uint8), perception.layout)


def test_rebuilt_state_keeps_most_features(perception):
    rng = np.random.default_rng(1)
    sim = Sim((DECK, random_deck(rng)), seed=5)
    agree = []
    for _ in range(12):
        play(sim, 5)
        obs = perception.read(render(sim, perception.layout))
        rebuilt = build_sim(obs, sim.time, DECK)
        agree.append((features(View(sim, 0)) == features(View(rebuilt, 0))).mean())
    assert np.mean(agree) > 0.85


def test_bot_plays_a_match_through_fake_adb(perception):
    from flybrain.real.bot import Bot, load_brain
    from flybrain.real.fake import FakeAdb

    brain = ROOT / "models/fly_royale.pt"
    if not brain.exists():
        pytest.skip("trained brain not available")
    agent, _ = load_brain(str(brain))
    rng = np.random.default_rng(2)
    sim = Sim((DECK, random_deck(rng)), seed=11)
    adb = FakeAdb(sim, perception.layout)
    bot = Bot(agent, perception, perception.layout, DECK, adb=adb)
    opp = BasicBot(1)
    bot.start_battle(0.0)
    while sim.time < 60 and not sim.done:
        m = opp.decide(sim, 1)
        bot.step(adb.screencap(), sim.time)
        if m:
            sim.play(1, m.card, m.x, m.y)
        for _ in range(10):
            sim.step()
    assert len(adb.plays) >= 3                      # it plays cards
    assert all(ok for *_, ok in adb.plays)          # every tap was a card it could afford, in a legal spot


class WaitingBrain:
    """Always waits when allowed (the failure seen in a real match: never playing at full elixir)."""

    def begin_episode(self, b):
        pass

    heads = (16, 2)

    def act(self, f, masks, learn=False):
        return np.array([[0 if masks[0][0, 0] else int(np.flatnonzero(masks[0][0])[0]), 0]])


def test_bot_never_sits_on_full_elixir(perception):
    from flybrain.real.bot import Bot
    from flybrain.real.fake import FakeAdb

    sim = Sim((DECK, random_deck(np.random.default_rng(6))), seed=2)
    adb = FakeAdb(sim, perception.layout)
    bot = Bot(WaitingBrain(), perception, perception.layout, DECK, adb=adb)
    bot.start_battle(0.0)
    while sim.time < 20 and not adb.plays:
        bot.step(adb.screencap(), sim.time)
        for _ in range(10):
            sim.step()
    assert adb.plays and adb.plays[0][3]                 # played (legally) once elixir was nearly full
    assert sim.time >= 10                                # but waited while it wasn't


def test_battle_log_lines_are_json(tmp_path):
    import json

    from flybrain.real.bot import log_step
    from flybrain.real.perception import Observation, SeenUnit

    obs = Observation(elixir=6.5, hand=["Knight", "Archers", None, "Giant"], next_card="Arrows",
                      towers={(0, "princess", 0): 1.0, (1, "king", None): 0.9},
                      units=[SeenUnit(owner=1, x=4.0, y=20.0, size=1, char="HogRider")])
    out = dict(obs=obs, action="Knight (frontline) left -> tile (4.5,11.0)", card="Knight", role="frontline", lane=0,
               who="coach:defend", enemy_elixir=3.25)
    log_step(tmp_path / "log.jsonl", 12.34, out)
    log_step(tmp_path / "log.jsonl", 13.34, out)
    lines = [json.loads(ln) for ln in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert len(lines) == 2 and lines[0]["t"] == 12.3 and lines[0]["who"] == "coach:defend"
    assert lines[0]["units"] == [[1, 4.0, 20.0, "HogRider"]] and lines[0]["towers"]["foe_king"] == 0.9
