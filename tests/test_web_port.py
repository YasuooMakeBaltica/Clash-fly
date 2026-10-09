"""The browser port (web/clash_core.js) must match the Python simulator and brain."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from flybrain.agent import AgentConfig, FlyAgent
from flybrain.connectome import synthetic
from flybrain.envs.clash.env import HEADS, N_CHANNELS, features
from flybrain.envs.clash.sim import ClashSim
from flybrain.envs.clash.strategy import View, place

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")

SCENARIO = dict(
    hands=[[2, 0, 5, 7], [4, 1, 3, 6]],
    queues=[[1, 3, 4, 6], [0, 2, 5, 7]],
    elixir=[10.0, 10.0],
    # (tick, player, card, x, y)
    plays=[[0, 0, 2, 3.5, 10.0], [5, 1, 4, 3.5, 22.0], [5, 1, 1, 14.5, 20.0], [40, 0, 5, 14.5, 12.0],
           [90, 1, 6, 14.5, 13.0]],
    snapshot_ticks=[20, 60, 120, 200],
)


def run_python():
    sim = ClashSim(seed=1, shuffle_updates=False)
    for p, h, q, e in zip(sim.players, SCENARIO["hands"], SCENARIO["queues"], SCENARIO["elixir"]):
        p.hand, p.queue, p.elixir = list(h), list(q), e
    tick, snaps = 0, []
    for at, player, card, x, y in SCENARIO["plays"]:
        while tick < at:
            sim.step()
            tick += 1
        assert sim.play(player, card, x, y)
    for at in SCENARIO["snapshot_ticks"]:
        while tick < at:
            sim.step()
            tick += 1
        snaps.append(dict(
            units=[[u.uid, u.owner, u.card_idx, u.x, u.y, u.hp] for u in sim.units],
            towers=[t.hp for t in sim.towers],
            elixir=[p.elixir for p in sim.players],
            features=[features(View(sim, pl)).tolist() for pl in (0, 1)],
            place=[[[list(place(View(sim, pl), c, ln)) for ln in (0, 1)] for c in range(8)] for pl in (0, 1)],
        ))
    return snaps


def run_node(tmp_path, brain=None):
    scen = tmp_path / "scenario.json"
    scen.write_text(json.dumps(SCENARIO | dict(brain_features=getattr(run_node, "feats", []))))
    cmd = ["node", str(ROOT / "web/parity_check.js"), str(scen)] + ([str(brain)] if brain else [])
    return json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)


def test_simulator_matches(tmp_path):
    py = run_python()
    js = run_node(tmp_path)["snapshots"]
    for p, j in zip(py, js):
        assert [u[:3] for u in p["units"]] == [u[:3] for u in j["units"]]
        np.testing.assert_allclose([u[3:] for u in p["units"]], [u[3:] for u in j["units"]], atol=1e-6)
        np.testing.assert_allclose(p["towers"], j["towers"], atol=1e-6)
        np.testing.assert_allclose(p["elixir"], j["elixir"], atol=1e-9)
        assert p["features"] == j["features"]
        np.testing.assert_allclose(p["place"], j["place"], atol=1e-9)


def test_brain_matches(tmp_path):
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from export_brain_js import export

    conn = synthetic(n_pn=120, n_kc=500, n_mbon=33, n_dan=20, n_glomeruli=70, seed=3)
    rng = np.random.default_rng(0)
    probe = (rng.random((16, N_CHANNELS)) < 0.2).astype(np.float32)
    agent = FlyAgent(conn, N_CHANNELS, HEADS, cfg=AgentConfig(action_groups="random", seed=0), probe=probe)
    agent.net.W_kc_mbon.mul_(torch.rand_like(agent.net.W_kc_mbon) * 1.5)   # pretend it learned
    agent.net.cfg.noise = 0.0
    agent.cfg.mbon_noise = 0.0
    brain = tmp_path / "brain.json"
    brain.write_text(json.dumps(export(agent)))
    feats = (rng.random((6, N_CHANNELS)) < 0.2).astype(np.float32)
    run_node.feats = feats.tolist()
    js = run_node(tmp_path, brain)["votes"]
    agent.act(feats, learn=False)
    for h in range(2):
        py = (agent.last["votes"][h] - 1e-3 * 0).numpy()  # includes <=1e-3 tie-break jitter
        np.testing.assert_allclose(py, [v[h] for v in js], atol=2e-3)
