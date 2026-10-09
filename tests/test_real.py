"""Real-game pipeline on fake screenshots: perception, state rebuild, and the bot loop.

These check the logic, not the default screen positions (those need a real
screenshot and `python -m flybrain.real.calibrate`).
"""

import numpy as np
import pytest

from flybrain.envs.clash.env import features
from flybrain.envs.clash.sim import ClashSim
from flybrain.envs.clash.strategy import BasicBot, Coach, View
from flybrain.real.layout import Box, Layout
from flybrain.real.perception import Perception, in_battle
from flybrain.real.state import build_sim
from flybrain.real.synthetic import render, write_templates


@pytest.fixture
def perception(tmp_path):
    write_templates(tmp_path)
    return Perception(Layout(), tmp_path)


def play(sim, seconds, bots=(Coach(), BasicBot(0))):
    for _ in range(int(seconds * 10)):
        if round(sim.time * 10) % 10 == 0:
            for p, b in enumerate(bots):
                m = b.decide(sim, p)
                if m:
                    sim.play(p, m.card, m.x, m.y)
        sim.step()


def test_layout_roundtrip(tmp_path):
    lay = Layout()
    lay.arena = Box(0.01, 0.05, 0.99, 0.8)
    lay.save(tmp_path / "layout.json")
    back = Layout.load(tmp_path / "layout.json")
    assert back.arena == lay.arena and back.hand_slots == lay.hand_slots and back.elixir_hsv == lay.elixir_hsv
    for tx, ty in ((0, 0), (3.5, 6.5), (18, 32), (9, 16)):
        assert np.allclose(lay.frac_to_tile(*lay.tile_to_frac(tx, ty)), (tx, ty))


def test_reads_elixir_hand_and_towers(perception):
    sim = ClashSim(seed=3)
    play(sim, 40)
    obs = perception.read(render(sim, perception.layout))
    assert abs(obs.elixir - sim.players[0].elixir) <= 0.15
    assert obs.hand == [sim.deck[i].name for i in sim.players[0].hand]
    assert obs.next_card == sim.deck[sim.players[0].queue[0]].name
    for t in sim.towers:
        assert abs(obs.towers[(t.owner, t.kind, t.lane)] - t.hp / t.max_hp) < 0.08


def test_not_in_battle_on_blank_screen(perception):
    assert not in_battle(np.zeros((960, 540, 3), np.uint8), perception.layout)


def test_rebuilt_state_gives_nearly_same_features(perception):
    sim = ClashSim(seed=5)
    agree = []
    for _ in range(12):
        play(sim, 5)
        obs = perception.read(render(sim, perception.layout))
        agree.append((features(View(sim, 0)) == features(View(build_sim(obs, sim.time), 0))).mean())
    assert np.mean(agree) > 0.9


def test_bot_plays_a_match_through_fake_adb(perception):
    from pathlib import Path

    from flybrain.real.bot import Bot, load_brain
    from flybrain.real.fake import FakeAdb

    brain = Path(__file__).resolve().parents[1] / "models/fly_v2.pt"
    if not brain.exists():
        pytest.skip("trained brain not available")
    agent, _ = load_brain(str(brain))
    sim = ClashSim(seed=11)
    adb = FakeAdb(sim, perception.layout)
    bot = Bot(agent, perception, perception.layout, adb=adb)
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
    assert all(ok for *_, ok in adb.plays)          # every tap landed on a card it could afford, in a legal spot
