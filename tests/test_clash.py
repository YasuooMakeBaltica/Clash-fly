import numpy as np
import pytest

from flybrain.envs.clash.cards import DECK, CARD_INDEX, KING_HP, PRINCESS_HP
from flybrain.envs.clash.env import CHANNELS, HEADS, N_CHANNELS, ClashEnv, features, masks
from flybrain.envs.clash.sim import ClashSim, abs_y, frame_y
from flybrain.envs.clash.strategy import BasicBot, Coach, RandomBot, View, place, play_match


def test_initial_state():
    s = ClashSim(seed=0)
    assert len(s.towers) == 6
    assert sorted(t.hp for t in s.towers) == [PRINCESS_HP] * 4 + [KING_HP] * 2
    for p in s.players:
        assert len(p.hand) == 4 and len(p.queue) == 4
        assert sorted(p.hand + p.queue) == list(range(8))
        assert p.elixir == 5


def test_elixir_regen_and_cap():
    s = ClashSim(seed=0)
    for _ in range(28):
        s.step()
    assert s.players[0].elixir == pytest.approx(6.0, abs=0.01)
    for _ in range(300):
        s.step()
    assert s.players[0].elixir == 10
    assert s.players[0].leaked > 0


def test_play_spends_elixir_and_cycles_hand():
    s = ClashSim(seed=0)
    p = s.players[0]
    card = next(c for c in p.hand if DECK[c].cost <= 5)
    nxt = p.queue[0]
    assert s.play(0, card, 3.5, 10.0)
    assert p.elixir == 5 - DECK[card].cost
    assert card not in p.hand and nxt in p.hand and p.queue[-1] == card


def test_cannot_deploy_troops_on_enemy_side():
    s = ClashSim(seed=0)
    s.players[0].hand[0] = CARD_INDEX["Knight"]
    assert not s.play(0, CARD_INDEX["Knight"], 3.5, 20.0)
    assert s.play(0, CARD_INDEX["Knight"], 3.5, 12.0)


def test_unit_walks_bridge_and_damages_tower():
    s = ClashSim(seed=0)
    s.players[0].hand[0] = CARD_INDEX["Giant"]
    s.players[0].elixir = 10
    assert s.play(0, CARD_INDEX["Giant"], 3.5, 10.0)
    giant = s.units[0]
    crossed_at = []
    for _ in range(400):
        s.step()
        if 15 <= giant.y <= 17:
            crossed_at.append(giant.x)
    assert all(abs(x - 3.5) < 0.6 for x in crossed_at)  # crossed on the bridge
    assert s.tower(1, "princess", 0).hp < PRINCESS_HP
    assert s.tower(1, "princess", 1).hp == PRINCESS_HP


def test_spell_damages_units_and_towers():
    s = ClashSim(seed=0)
    s.players[1].hand[0] = CARD_INDEX["Goblins"]
    s.play(1, CARD_INDEX["Goblins"], 3.5, abs_y(1, 10.0))
    s.players[0].hand[0] = CARD_INDEX["Arrows"]
    gx, gy = s.units[0].x, s.units[0].y
    s.play(0, CARD_INDEX["Arrows"], gx, gy)
    for _ in range(10):
        s.step()
    assert not any(u.owner == 1 for u in s.units)  # goblins (202 hp) die to arrows (366)


def test_frames_mirror():
    assert frame_y(1, abs_y(1, 4.0)) == 4.0
    assert abs_y(1, 4.0) == 28.0


def test_coach_beats_random_and_basic():
    wins = [play_match(Coach(), RandomBot(i), seed=i).winner for i in range(10)]
    assert wins.count(0) >= 8
    wins = [play_match(Coach(), BasicBot(i), seed=i).winner for i in range(10)]
    assert wins.count(0) >= 7


def test_match_is_fair_for_mirrored_bots():
    res = [play_match(BasicBot(i), BasicBot(100 + i), seed=i).winner for i in range(60)]
    assert abs(res.count(0) - res.count(1)) <= 15


def test_placement_on_own_side():
    s = ClashSim(seed=0)
    for player in (0, 1):
        v = View(s, player)
        for c, card in enumerate(DECK):
            for lane in (0, 1):
                x, y = place(v, c, lane)
                assert s.valid_position(player, c, x, y)


def test_features_and_masks():
    s = ClashSim(seed=0)
    v = View(s, 0)
    f = features(v)
    assert f.shape == (N_CHANNELS,) == (len(CHANNELS),)
    assert set(np.unique(f)) <= {0.0, 1.0}
    card_mask, lane_mask = masks(v)
    assert card_mask[0] and card_mask.sum() == 1 + sum(DECK[c].cost <= 5 for c in s.players[0].hand)
    assert lane_mask.all()


def test_env_runs_a_match():
    env = ClashEnv(2, lambda seed: RandomBot(seed), decision_every=2.0)
    f, done, total = env.reset(), False, np.zeros(2)
    steps = 0
    while not done:
        acts, _ = env.teacher()
        f, r, done = env.step(acts)
        total += r
        steps += 1
    assert steps <= 91
    assert f.shape == (2, N_CHANNELS)
    assert set(env.results()) <= {-1, 0, 1}
    assert HEADS == (9, 2)
