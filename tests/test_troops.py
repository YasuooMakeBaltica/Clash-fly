"""Troop detector plumbing: geometry between screenshots and dataset frames, inference, state rebuild."""

from pathlib import Path

import numpy as np
import pytest
import torch

from flybrain.envs.royale.db import load
from flybrain.real.layout import Layout
from flybrain.real.perception import BadgeDetector, Observation, SeenUnit, default_detector
from flybrain.real.state import build_sim
from flybrain.real.troops import (CLASSES, FRAME_SIZE, INPUT_SIZE, KATA_TO_CHAR, TroopDetector, TroopNet, arena_crop,
                                  frame_box, frame_to_tile)

ROOT = Path(__file__).resolve().parents[1]


def test_every_troop_type_is_a_simulator_character():
    chars = load().characters
    assert all(c in chars for c in CLASSES)
    assert KATA_TO_CHAR["hog-rider"] == "HogRider" and KATA_TO_CHAR["royal-giant-evolution"] == "RoyalGiant"


def test_frame_geometry_roundtrip():
    lay = Layout()
    w, h = 540, 960
    fx0, fy0, fx1, fy1 = frame_box(lay, w, h)
    for tx, ty in ((3.5, 6.5), (14.5, 25.5), (9, 16), (0.5, 31)):
        px, py = lay.tile_to_px(tx, ty, w, h)
        u = (px - fx0) / (fx1 - fx0) * FRAME_SIZE[0]
        v = (py - fy0) / (fy1 - fy0) * FRAME_SIZE[1]
        bx, by = frame_to_tile(u, v)
        assert abs(bx - tx) < 0.1 and abs(by - ty) < 0.1


def test_arena_crop_puts_a_screen_point_where_the_frame_expects_it():
    lay = Layout()
    img = np.zeros((960, 540, 3), np.uint8)
    px, py = lay.tile_to_px(14.5, 25.5, 540, 960)
    img[py - 3:py + 4, px - 3:px + 4] = 255
    crop = arena_crop(img, lay)
    assert crop.shape == (INPUT_SIZE[1], INPUT_SIZE[0], 3)
    ys, xs = np.nonzero(crop[..., 0] > 100)
    u, v = xs.mean() / 0.5, ys.mean() / 0.5
    tx, ty = frame_to_tile(u, v)
    assert abs(tx - 14.5) < 0.3 and abs(ty - 25.5) < 0.3


def test_detector_runs_end_to_end(tmp_path):
    torch.manual_seed(0)
    net = TroopNet(len(CLASSES), width=16)
    path = tmp_path / "troops.pt"
    torch.save(dict(model=net.state_dict(), classes=CLASSES, width=16), path)
    det = TroopDetector(path, threshold=0.0)
    img = np.random.default_rng(0).integers(0, 255, (960, 540, 3), dtype=np.uint8)
    units = det.detect(img, Layout())
    assert units and all(u.char in CLASSES and u.owner in (0, 1) for u in units)
    assert all(0 <= u.x <= 18 and 0 <= u.y <= 32 for u in units)


def test_default_detector_falls_back_without_weights(tmp_path):
    assert isinstance(default_detector(tmp_path / "missing.pt"), BadgeDetector)


def test_state_rebuild_uses_troop_types():
    obs = Observation(elixir=5, hand=[None] * 4, next_card=None, towers={},
                      units=[SeenUnit(owner=1, x=4.0, y=20.0, size=1, char="HogRider"),
                             SeenUnit(owner=1, x=14.0, y=22.0, size=1, char="Minion")])
    sim = build_sim(obs, 30.0, ["Knight", "Archers", "Giant", "Musketeer", "Goblins", "Fireball", "Arrows", "Zap"])
    enemy = [u for u in sim.units if u.owner == 1 and not u.tower]
    names = sorted(u.spec.name for u in enemy)
    assert names == ["HogRider", "Minion"]
    assert sorted(u.card for u in enemy) == ["Hog Rider", "Minions"]
    assert any(u.flying for u in enemy)
