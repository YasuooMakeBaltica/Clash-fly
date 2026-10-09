"""The browser port of the full game (web/royale_core.js) must match the Python."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from flybrain.envs.royale.db import ROLES, load
from flybrain.envs.royale.env import features, pick_card, role_mask
from flybrain.envs.royale.sim import Sim
from flybrain.envs.royale.guard import guard
from flybrain.envs.royale.strategy import Coach, View, place

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
DB = load()

DECK0 = ["Hog Rider", "Musketeer", "Valkyrie", "Archer Queen", "Fireball", "Zap", "Inferno Tower", "Minions"]
DECK1 = ["Golem", "Witch", "Electro Wizard", "Bowler", "The Log", "Freeze", "Baby Dragon", "Goblin Gang"]
SCENARIO = dict(
    decks=[DECK0, DECK1],
    hands=[DECK0[:4], DECK1[:4]], queues=[DECK0[4:], DECK1[4:]],
    # (tick, kind, player, card, x, y)
    events=[
        [0, "deploy", 1, "Golem", 9.0, 30.0], [5, "deploy", 1, "Witch", 8.0, 27.0],
        [10, "deploy", 0, "Inferno Tower", 7.5, 6.0], [20, "deploy", 0, "Hog Rider", 14.5, 13.5],
        [30, "deploy", 1, "Goblin Gang", 14.5, 22.0], [40, "deploy", 0, "Valkyrie", 14.0, 12.0],
        [60, "deploy", 0, "Archer Queen", 4.0, 5.0], [80, "deploy", 1, "Electro Wizard", 4.0, 20.0],
        [90, "deploy", 0, "Minions", 9.0, 10.0], [100, "deploy", 1, "Baby Dragon", 9.0, 22.0],
        [120, "deploy", 0, "Fireball", 4.0, 19.0], [130, "deploy", 1, "The Log", 14.0, 16.0],
        [140, "ability", 0, "", 0, 0], [150, "deploy", 0, "Zap", 9.0, 18.0], [160, "deploy", 1, "Freeze", 7.5, 8.0],
        [170, "deploy", 1, "Bowler", 4.0, 24.0], [180, "deploy", 0, "Musketeer", 6.0, 4.0],
    ],
    snapshot_ticks=[50, 120, 200, 320],
    # placement of cards that need not be in hand (buildings, spells, win conditions ...)
    place_cards=["Cannon", "Inferno Tower", "Tesla", "Tombstone", "X-Bow", "Hog Rider", "Golem", "Musketeer",
                 "Valkyrie", "Minions", "Skeleton Army", "Fireball", "The Log", "Rage", "Miner"],
)


def run_python():
    sim = Sim((DECK0, DECK1), seed=1, shuffle_updates=False)
    for p, h, q in zip(sim.players, SCENARIO["hands"], SCENARIO["queues"]):
        p.hand, p.queue, p.elixir = list(h), list(q), 10
    sim.tower(1, "princess", 0).hp -= 1
    ev = list(SCENARIO["events"])
    tick, snaps = 0, []
    for at in SCENARIO["snapshot_ticks"]:
        while tick < at:
            while ev and ev[0][0] == tick:
                _, kind, player, name, x, y = ev.pop(0)
                if kind == "deploy":
                    sim._deploy(player, DB.cards[name], x, y)
                else:
                    sim.players[player].elixir = 10
                    sim.use_ability(player)
            sim.step()
            tick += 1
        views = [View(sim, p) for p in (0, 1)]
        snaps.append(dict(
            units=[[u.uid, u.spec.name if u.spec else u.tower, u.owner, u.x, u.y, u.hp, u.shield] for u in sim.units],
            projectiles=len(sim.projectiles), effects=len(sim.effects),
            elixir=[p.elixir for p in sim.players],
            features=[features(v).tolist() for v in views],
            masks=[role_mask(v).tolist() for v in views],
            coach=[list(Coach().suggest(v)) for v in views],
            place=[[[list(place(v, n, ln)) for ln in (0, 1)] for n in v.hand()] for v in views],
            picks=[[pick_card(v, r) for r in ROLES] for v in views],
            plan=[list(Coach().plan(v)) for v in views],
            guard=[[[list(guard(v, c, ln)) for ln in (0, 1)] for c in [None] + [pick_card(v, r) for r in ROLES]]
                   for v in views],
            place_any=[[[list(place(v, n, ln)) for ln in (0, 1)] for n in SCENARIO["place_cards"]] for v in views],
        ))
    return snaps


def run_node(tmp_path, scenario, brain=None):
    f = tmp_path / "scenario.json"
    f.write_text(json.dumps(scenario))
    cmd = ["node", str(ROOT / "web/royale_parity.js"), str(f)] + ([str(brain)] if brain else [])
    return json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)


def test_full_simulator_matches(tmp_path):
    py = run_python()
    js = run_node(tmp_path, SCENARIO)["snapshots"]
    for p, j in zip(py, js):
        assert [u[:3] for u in p["units"]] == [u[:3] for u in j["units"]]
        np.testing.assert_allclose([u[3:] for u in p["units"]], [u[3:] for u in j["units"]], atol=1e-6)
        assert (p["projectiles"], p["effects"]) == (j["projectiles"], j["effects"])
        np.testing.assert_allclose(p["elixir"], j["elixir"], atol=1e-9)
        assert p["features"] == j["features"]
        assert p["masks"] == j["masks"]
        assert p["coach"] == j["coach"]
        np.testing.assert_allclose(p["place"], j["place"], atol=1e-9)
        assert p["picks"] == j["picks"]
        assert p["plan"] == j["plan"]
        assert p["guard"] == j["guard"]
        np.testing.assert_allclose(p["place_any"], j["place_any"], atol=1e-9)


def test_royale_brain_matches(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    from export_brain_js import export

    from flybrain.agent import AgentConfig, FlyAgent
    from flybrain.connectome import synthetic
    from flybrain.envs.royale.env import HEADS, N_CHANNELS

    conn = synthetic(n_pn=200, n_kc=500, n_mbon=60, n_dan=20, n_glomeruli=100, seed=3)
    rng = np.random.default_rng(0)
    probe = (rng.random((16, N_CHANNELS)) < 0.15).astype(np.float32)
    agent = FlyAgent(conn, N_CHANNELS, HEADS, cfg=AgentConfig(action_groups="random", seed=0), probe=probe)
    agent.net.W_kc_mbon.mul_(torch.rand_like(agent.net.W_kc_mbon) * 1.5)
    agent.net.cfg.noise = 0.0
    agent.cfg.mbon_noise = 0.0
    brain = tmp_path / "brain.json"
    brain.write_text(json.dumps(export(agent)))
    feats = (rng.random((5, N_CHANNELS)) < 0.15).astype(np.float32)
    js = run_node(tmp_path, dict(SCENARIO, snapshot_ticks=[], brain_features=feats.tolist()), brain)["votes"]
    agent.act(feats, learn=False)
    for h in range(2):
        np.testing.assert_allclose(agent.last["votes"][h].numpy(), [v[h] for v in js], atol=2e-3)
