"""Mechanics of the full simulator, one scenario each."""

import math

import numpy as np
import pytest

from flybrain.envs.royale.db import ROLES, load
from flybrain.envs.royale.sim import HEIGHT, LANE_X, Sim, abs_y

DB = load()
FILLER = ["Knight", "Archers", "Giant", "Musketeer", "Mini P.E.K.K.A", "Goblins", "Fireball", "Arrows"]


def sim(deck0=FILLER, deck1=FILLER, seed=0):
    s = Sim((list(deck0), list(deck1)), seed=seed, shuffle_updates=False)
    return s


def drop(s, player, name, x, y):
    s._deploy(player, DB.cards[name], x, y)
    return [u for u in s.units if u.card == name and u.owner == player]


def run(s, seconds):
    for _ in range(int(round(seconds / s.dt))):
        s.step()


def units(s, name, owner=None):
    return [u for u in s.units if u.alive and u.spec is not None and u.spec.name == name
            and (owner is None or u.owner == owner)]


def test_database_covers_the_card_pool():
    pool = DB.pool()
    assert len(pool) >= 105
    assert all(DB.cards[n].role in ROLES for n in pool)
    assert DB.characters["Knight"].hp == 1766 and DB.characters["Knight"].damage == 202
    assert DB.characters["HogRider"].hp == 1696


def test_deck_validation():
    with pytest.raises(ValueError):
        Sim((FILLER[:7], FILLER))
    with pytest.raises(ValueError):
        Sim((FILLER[:7] + ["Not A Card"], FILLER))


def test_hog_rider_jumps_the_river_and_ignores_troops():
    s = sim()
    drop(s, 0, "Hog Rider", 6.0, 12.0)              # not in front of a bridge
    drop(s, 1, "Knight", 6.0, 20.0)
    hog = units(s, "HogRider", 0)[0]
    xs_in_river = []
    for _ in range(60):
        s.step()
        if 15 <= hog.y <= 17:
            xs_in_river.append(hog.x)
    assert xs_in_river and min(abs(x - lx) for x in xs_in_river for lx in LANE_X) > 1.0
    run(s, 4)
    assert s.tower(1, "princess", 0).hp < s.tower(1, "princess", 0).max_hp


def test_ground_troops_use_bridges():
    s = sim()
    drop(s, 0, "Giant", 7.0, 12.0)
    g = units(s, "Giant", 0)[0]
    xs = []
    for _ in range(150):
        s.step()
        if 15 <= g.y <= 17:
            xs.append(g.x)
    assert xs and all(abs(x - LANE_X[0]) < 1.0 for x in xs)


def test_ground_only_attackers_cannot_hit_air():
    s = sim()
    drop(s, 0, "Knight", 9.0, 14.0)          # out of every tower's range
    drop(s, 1, "Minions", 9.0, 16.0)
    run(s, 4)
    minions = units(s, "Minion", 1)
    knight = units(s, "Knight", 0)
    assert all(m.hp == m.max_hp for m in minions)
    assert knight and knight[0].hp < knight[0].max_hp


def test_musketeer_shoots_air():
    s = sim()
    drop(s, 0, "Musketeer", 9.0, 8.0)
    drop(s, 1, "Minions", 9.0, 13.0)
    run(s, 8)
    assert len(units(s, "Minion", 1)) < 3


def test_witch_spawns_skeletons():
    s = sim()
    drop(s, 0, "Witch", 9.0, 3.0)
    run(s, 9)
    assert len(units(s, "Skeleton", 0)) >= 4


def test_golem_splits_and_deals_death_damage():
    s = sim()
    g = drop(s, 0, "Golem", 9.0, 10.0)[0]
    run(s, 3.5)
    skel = drop(s, 1, "Skeletons", 9.0, 10.0)
    g.hp = 1
    s._damage(g, 10)
    run(s, 0.3)
    assert len(units(s, "Golemite", 0)) == 2
    assert not units(s, "Skeleton", 1)       # killed by death damage


def test_inferno_tower_ramps_up():
    s = sim()
    drop(s, 0, "Inferno Tower", 9.0, 9.0)
    run(s, 1.5)
    golem = drop(s, 1, "Golem", 9.0, 14.0)[0]
    golem.deploy_left = 0
    golem.x, golem.y = 9.0, 13.0
    hp = [golem.hp]
    for _ in range(6):
        run(s, 1.0)
        hp.append(golem.hp)
    loss = -np.diff(hp)
    assert loss[-1] > 5 * max(loss[0], 1)


def test_freeze_stops_movement_and_attacks():
    s = sim()
    k = drop(s, 1, "Knight", 9.0, 20.0)[0]
    run(s, 1.2)
    s._deploy(0, DB.cards["Freeze"], k.x, k.y)
    run(s, 0.2)
    y0 = k.y
    run(s, 3.0)
    assert abs(k.y - y0) < 0.05


def test_fireball_travels_and_does_reduced_tower_damage():
    s = sim()
    t = s.tower(1, "princess", 0)
    s._deploy(0, DB.cards["Fireball"], t.x, t.y)
    run(s, 0.3)
    assert t.hp == t.max_hp                 # still in flight
    run(s, 3.0)
    dmg = t.max_hp - t.hp
    assert 0 < dmg < DB.projectiles["FireballSpell"].damage * 0.5


def test_log_pushes_ground_and_skips_air():
    s = sim()
    k = drop(s, 1, "Knight", 9.0, 20.0)[0]
    drop(s, 1, "Minions", 9.0, 21.0)
    k.deploy_left = 0
    for m in units(s, "Minion", 1):
        m.deploy_left = 0
    s._deploy(0, DB.cards["The Log"], 9.0, 18.0)
    ys = []
    for _ in range(12):
        ys.append(k.y)
        s.step()
    assert k.max_hp - k.hp == pytest.approx(DB.projectiles["LogProjectileRolling"].damage)
    assert max(np.diff(ys)) > 0.4                      # knocked back toward its own side
    assert all(u.hp == u.max_hp for u in units(s, "Minion", 1))


def test_tornado_pulls_to_centre():
    s = sim()
    k = drop(s, 1, "Knight", 6.0, 22.0)[0]
    run(s, 1.0)
    k.deploy_left = 0
    d0 = math.dist((k.x, k.y), (9.0, 22.0))
    s._deploy(0, DB.cards["Tornado"], 9.0, 22.0)
    run(s, 0.8)
    assert math.dist((k.x, k.y), (9.0, 22.0)) < d0 - 1.0


def test_clone_makes_one_hp_copies():
    s = sim()
    drop(s, 0, "Knight", 9.0, 8.0)
    run(s, 1.2)
    s._deploy(0, DB.cards["Clone"], 9.0, 8.0)
    run(s, 0.2)
    ks = units(s, "Knight", 0)
    assert len(ks) == 2 and min(k.max_hp for k in ks) == 1


def test_mirror_replays_last_card_for_one_more_elixir():
    s = sim(deck0=["Knight", "Mirror", "Archers", "Giant", "Musketeer", "Goblins", "Fireball", "Arrows"])
    p = s.players[0]
    p.hand = ["Knight", "Mirror", "Archers", "Giant"]
    p.queue = ["Musketeer", "Goblins", "Fireball", "Arrows"]
    p.elixir = 10
    assert not s.can_play(0, "Mirror")
    assert s.play(0, "Knight", 9, 8)
    assert s.card_cost(0, "Mirror") == 4
    assert s.play(0, "Mirror", 9, 9)
    assert p.elixir == pytest.approx(3.0)
    assert len([u for u in s.units if u.card == "Knight"]) == 2


def test_graveyard_spawns_skeletons_over_time():
    s = sim()
    s._deploy(0, DB.cards["Graveyard"], 9.0, 20.0)
    run(s, 2.0)
    early = len(units(s, "Skeleton", 0))
    start = s._uid
    run(s, 9.0)
    assert early == 0 and s._uid - start >= 12        # skeletons keep appearing for ~7 s


def test_miner_tunnels_to_enemy_side():
    s = sim(deck0=["Miner"] + FILLER[1:])
    p = s.players[0]
    p.hand[0] = "Miner"
    p.elixir = 10
    assert s.play(0, "Miner", 4.0, 24.0)
    m = units(s, "Miner", 0)[0]
    assert m.y > 20
    run(s, 6)
    assert s.tower(1, "princess", 0).hp < s.tower(1, "princess", 0).max_hp


def test_troops_cannot_be_placed_on_enemy_side():
    s = sim()
    s.players[0].elixir = 10
    assert not s.valid_position(0, "Knight", 9, 20)
    assert s.valid_position(0, "Fireball", 9, 28)


def test_goblin_barrel_lands_goblins():
    s = sim()
    t = s.tower(1, "princess", 1)
    s._deploy(0, DB.cards["Goblin Barrel"], t.x, t.y - 2)
    run(s, 2.0)
    assert not units(s, "Goblin", 0)                  # still flying
    run(s, 2.0)
    assert len(units(s, "Goblin", 0)) == 3


def test_elixir_collector_and_elixir_golem():
    s = sim()
    drop(s, 0, "Elixir Collector", 9.0, 2.0)
    s.players[0].elixir = 0
    s.players[1].elixir = 0
    run(s, 12)
    gain = s.players[0].elixir - s.players[1].elixir
    assert gain == pytest.approx(1.0, abs=0.15)
    g = drop(s, 1, "Elixir Golem", 9.0, 25.0)[0]
    run(s, 1.2)
    before = s.players[0].elixir
    g.hp = 0
    run(s, 0.1)
    assert s.players[0].elixir >= before + 0.9


def test_one_champion_at_a_time_and_abilities_cost_elixir():
    deck = ["Archer Queen", "Golden Knight"] + FILLER[:6]
    s = sim(deck0=deck)
    p = s.players[0]
    p.hand = ["Archer Queen", "Golden Knight", "Knight", "Archers"]
    p.queue = FILLER[2:6] + []
    p.queue = ["Giant", "Musketeer", "Mini P.E.K.K.A", "Goblins"]
    p.elixir = 10
    assert s.play(0, "Archer Queen", 9, 8)
    assert not s.can_play(0, "Golden Knight")
    run(s, 1.5)
    e = s.players[0].elixir
    assert s.use_ability(0)
    assert s.players[0].elixir == pytest.approx(e - 1, abs=0.05)
    assert not s.use_ability(0)        # on cooldown


def test_prince_charge_hits_harder():
    s = sim()
    pr = drop(s, 0, "Prince", 9.0, 5.0)[0]
    tgt = drop(s, 1, "Giant", 9.0, 11.0)[0]
    tgt.spec = DB.characters["Golem"]          # a target that doesn't walk to towers
    tgt.hp = tgt.max_hp = 99999
    run(s, 6.0)
    assert tgt.max_hp - tgt.hp >= DB.characters["Prince"].charge_damage


def test_collisions_separate_units():
    s = sim()
    a = drop(s, 0, "Knight", 9.0, 5.0)[0]
    b = drop(s, 0, "Valkyrie", 9.0, 5.0)[0]
    run(s, 1.5)
    assert math.dist((a.x, a.y), (b.x, b.y)) > 0.6


def test_mega_knight_landing_damages():
    s = sim()
    drop(s, 1, "Skeletons", 9.0, 9.0)
    run(s, 1.2)
    drop(s, 0, "Mega Knight", 9.0, 9.0)
    run(s, 1.5)
    assert not units(s, "Skeleton", 1)


def test_electro_giant_reflects_and_stuns():
    s = sim()
    eg = drop(s, 0, "Electro Giant", 9.0, 9.0)[0]
    k = drop(s, 1, "Knight", 9.0, 10.5)[0]
    k.deploy_left = eg.deploy_left = 0
    eg.spec = DB.characters["ElectroGiant"]
    run(s, 3.0)
    assert k.hp < k.max_hp


def test_ram_rider_rider_dies_with_ram():
    s = sim()
    drop(s, 0, "Ram Rider", 9.0, 8.0)
    assert units(s, "RamRider", 0)
    ram = units(s, "Ram", 0)[0]
    ram.hp = 0
    run(s, 0.3)
    assert not units(s, "RamRider", 0)


def test_lightning_hits_the_three_biggest():
    s = sim()
    drop(s, 1, "Giant", 9.0, 20.0)            # away from towers (Lightning also hits towers)
    drop(s, 1, "Knight", 8.0, 20.0)
    drop(s, 1, "Musketeer", 10.0, 20.0)
    drop(s, 1, "Skeletons", 9.0, 21.0)
    run(s, 1.2)
    s._deploy(0, DB.cards["Lightning"], 9.0, 20.0)
    run(s, 2.0)
    alive = {u.spec.name: u for u in s.units if u.owner == 1 and not u.tower}
    assert alive["Giant"].hp < alive["Giant"].max_hp and alive["Knight"].hp < alive["Knight"].max_hp
    assert "Musketeer" not in alive                    # 720 hp, killed outright
    assert len(units(s, "Skeleton", 1)) == 3           # small fry left alone


def test_royal_recruits_span_both_lanes():
    s = sim()
    rr = drop(s, 0, "Royal Recruits", 9.0, 8.0)
    xs = sorted(u.x for u in rr)
    assert len(rr) == 6 and xs[0] < 5 and xs[-1] > 13


def test_quiet_match_goes_to_overtime_and_draws():
    s = sim()
    while not s.done:
        s.step()
    assert s.overtime and s.time >= 300 - 1e-6 and s.winner is None


def test_sudden_death_in_overtime():
    s = sim()
    run(s, 181)
    assert s.overtime and not s.done
    t = s.tower(1, "princess", 0)
    s._damage(t, 1e6)
    run(s, 0.2)
    assert s.done and s.winner == 0
