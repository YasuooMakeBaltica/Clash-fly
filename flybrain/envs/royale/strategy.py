"""Basic Clash Royale strategy for any deck: a player-frame view, card
placement, a coach that teaches the fly, and simple bots.

The coach plays by card roles and card traits, so it works with any deck:

1. Finish a tower with a spell when it's in range of a kill.
2. Defend first, with the card that answers the threat best:
   building-targeting tanks -> tank killers, defensive buildings, swarms;
   air -> anything that hits air; swarms -> splash and small spells;
   tank killers / big melee -> swarms and cheap distractions;
   ranged support -> frontline or a spell. Don't over-commit.
3. Spells only for positive trades (hit more elixir than they cost).
4. Counter-push: support surviving troops, Rage or Freeze a big push.
5. Punish: when the opponent is low on elixir or builds a slow push at the
   back, attack the other lane with the win condition.
6. Don't leak elixir: at high elixir start a push (heavy tank at the back,
   fast win condition at the bridge), or drop a spawner/cheap card.
7. Champions use their ability when they're in a fight.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .db import ROLES, load
from .decks import hits_air, is_splash
from .sim import HEIGHT, LANE_X, RIVER_LO, WIDTH, Sim, Unit, abs_y, frame_y, lane_of

THREAT_LINE = 20.0
HEAVY = {"Golem", "Lava Hound", "Electro Giant", "Goblin Giant", "Giant", "Elixir Golem", "Royal Giant"}
DROP_ON_TOWER = {"Miner", "Goblin Barrel", "Graveyard", "Goblin Drill"}
SIEGE = {"X-Bow", "Mortar"}


@dataclass
class Move:
    card: str
    lane: int
    x: float
    y: float


def card_value(u: Unit) -> float:
    """Rough elixir worth of a unit (its card cost split over the card's units)."""
    db = load()
    if u.card is None or u.card not in db.cards:
        return 0.5
    c = db.cards[u.card]
    n = max(1, sum(k for _, k in c.summons))
    return c.elixir / n


class View:
    """What player ``me`` sees, in their own frame (own side is y < 15)."""

    def __init__(self, sim: Sim, me: int):
        self.sim, self.me, self.foe = sim, me, 1 - me
        self.p = sim.players[me]
        self.db = sim.db

    def fy(self, y: float) -> float:
        return frame_y(self.me, y)

    @property
    def elixir(self) -> float:
        return self.p.elixir

    def hand(self) -> list[str]:
        return list(self.p.hand)

    def playable(self) -> list[str]:
        return [c for c in self.p.hand if self.sim.can_play(self.me, c)]

    def enemies(self) -> list[Unit]:
        return [u for u in self.sim.units if u.alive and u.owner == self.foe and not u.tower and not u.timed
                and u.parent is None]

    def mine(self) -> list[Unit]:
        return [u for u in self.sim.units if u.alive and u.owner == self.me and not u.tower and not u.timed
                and u.parent is None]

    def threats(self, lane: int) -> list[Unit]:
        return [u for u in self.enemies() if lane_of(u.x) == lane and self.fy(u.y) < THREAT_LINE]

    def threat_hp(self, lane: int) -> float:
        return sum(u.hp + u.shield for u in self.threats(lane))

    def defenders(self, lane: int) -> list[Unit]:
        return [u for u in self.mine() if lane_of(u.x) == lane and self.fy(u.y) < RIVER_LO + 1 and not u.building]

    def pushers(self, lane: int) -> list[Unit]:
        return [u for u in self.mine() if lane_of(u.x) == lane and self.fy(u.y) >= 9 and not u.building]

    def enemy_back_tank(self) -> int | None:
        for u in self.enemies():
            if u.card in HEAVY and self.fy(u.y) > 24:
                return lane_of(u.x)
        return None

    def tower_frac(self, mine: bool, lane: int) -> float:
        t = self.sim.tower(self.me if mine else self.foe, "princess", lane)
        return max(t.hp, 0) / t.max_hp

    def weak_lane(self) -> int:
        f = [self.tower_frac(False, 0), self.tower_frac(False, 1)]
        if f[0] == f[1]:
            return int(self.sim.rng.integers(2))
        return int(np.argmin(f))

    def enemy_elixir(self) -> float:
        # In the simulator we read it; a real player counts it. Treat as "estimate".
        return self.sim.players[self.foe].elixir

    def profile(self, units: list[Unit]) -> dict:
        p = dict(air=False, building_targeter=False, swarm=False, splash=False, ranged=False, tank_killer=False,
                 tank=False, building=False, hp=0.0)
        for u in units:
            s = u.spec
            p["hp"] += u.hp + u.shield
            p["air"] |= u.flying
            p["building"] |= u.building
            if u.building:
                continue
            p["building_targeter"] |= s.buildings_only
            p["tank"] |= u.max_hp >= 2000
            p["splash"] |= bool(s.splash or (s.projectile is not None and s.projectile.radius > 0))
            p["ranged"] |= s.range >= 4.0
            p["tank_killer"] |= (s.damage / max(s.hit_speed, 0.1) >= 300 and s.range < 3)
        p["swarm"] = sum(1 for u in units if not u.building and u.max_hp <= 400) >= 3
        return p


# ------------------------------------------------------------------ spells
def spell_spot(view: View, name: str, lane: int | None = None) -> tuple[float, float, float]:
    """(elixir value hit, x, y) of the best place for a spell."""
    db = view.db
    c = db.cards[name]
    if c.effect is not None:
        radius, air, ground = c.effect.radius, c.effect.air, c.effect.ground
        dmg = c.effect.damage or (c.effect.buff.dps * c.effect.duration if c.effect.buff is not None else 0)
        if c.effect.projectile is not None:
            dmg = c.effect.projectile.damage
    elif c.projectile is not None:
        p = c.projectile
        radius = p.radius or p.pierce_radius or 1.5
        air, ground = p.air or not p.ground, p.ground or not p.air
        dmg = p.damage * max(1, c.waves)
    else:
        return 0.0, 0.0, 0.0
    if name in ("Freeze", "Rage", "Clone", "Mirror", "Tornado", "Graveyard"):
        return 0.0, 0.0, 0.0
    best = (0.0, 0.0, 0.0)
    for u in view.enemies():
        if lane is not None and lane_of(u.x) != lane:
            continue
        v = 0.0
        for o in view.enemies():
            if (o.flying and not air) or (not o.flying and not ground):
                continue
            if math.dist((u.x, u.y), (o.x, o.y)) <= radius + o.radius:
                v += min(1.0, dmg / max(o.hp + o.shield, 1)) * card_value(o)
        if v > best[0]:
            best = (v, u.x, u.y)
    return best


# --------------------------------------------------------------- placement
def place(view: View, name: str, lane: int) -> tuple[float, float]:
    """Where to drop card ``name`` in ``lane``; absolute (x, y)."""
    db, me = view.db, view.me
    c = db.cards[name]
    if c.special == "mirror" and view.p.last_card:
        c = db.cards[view.p.last_card]
        name = c.name
    lx = LANE_X[lane]
    tc = 1.0 if lane == 0 else -1.0
    enemy_tower = view.sim.lane_target(me, lane)
    if name in DROP_ON_TOWER:
        off = 0.0 if name in ("Miner", "Goblin Drill") else -1.0
        return float(np.clip(enemy_tower.x + tc * 0.5, 0.5, WIDTH - 0.5)), abs_y(me, view.fy(enemy_tower.y) + off - 1.5)
    if c.type == "spell" and not c.summons:
        if name in ("Rage", "Freeze", "Clone"):
            pushers = view.pushers(lane) or view.mine()
            if pushers:
                lead = max(pushers, key=lambda u: view.fy(u.y))
                return lead.x, lead.y
            return lx, abs_y(me, 20.0)
        if name == "Tornado":
            th = view.threats(lane)
            if th:
                lead = min(th, key=lambda u: view.fy(u.y))
                return float(np.clip(lead.x + tc * 1.5, 0.5, WIDTH - 0.5)), lead.y
        v, x, y = spell_spot(view, name, lane)
        if c.special == "rolling":
            if v > 0:
                return x, abs_y(me, max(0.6, view.fy(y) - 2.0))
            return lx, abs_y(me, 16.5)
        if v > 0:
            return x, y
        return enemy_tower.x, enemy_tower.y
    if c.type == "spell":  # Royal Delivery
        th = view.threats(lane)
        if th:
            lead = min(th, key=lambda u: view.fy(u.y))
            return lead.x, abs_y(me, min(14.0, view.fy(lead.y)))
        return lx, abs_y(me, 10.0)
    threats = view.threats(lane)
    if c.type == "building":
        if name in SIEGE:
            return lx + tc * 1.0, abs_y(me, 14.0 if name == "X-Bow" else 12.5)
        if c.role == "spawner":
            return WIDTH / 2 - tc * 2.0, abs_y(me, 2.5)
        # Centre, about 6 tiles from the river: close enough to the bridge that building-targeting troops
        # (Hog Rider, Giant ...) turn to it, and both princess towers reach troops attacking it.
        # (Head-to-head, 400 games: 51% wins / 43% losses vs the old spot 6 tiles from the edge.)
        return WIDTH / 2 - tc * 1.5, abs_y(me, 9.0)
    if threats:
        lead = min(threats, key=lambda u: view.fy(u.y))
        lead_fy = view.fy(lead.y)
        if c.role in ("ranged", "splash_air", "champion") and not c.name in ("Golden Knight", "Mighty Miner", "Monk"):
            return lx + 1.5 * tc, abs_y(me, 4.5)
        if lead_fy > RIVER_LO:
            return lx + tc, abs_y(me, 11.0)
        if c.role in ("swarm", "cycle", "air"):
            return lead.x, abs_y(me, max(0.5, lead_fy - 1.0))
        return float(np.clip(lead.x + tc, 0.5, WIDTH - 0.5)), abs_y(me, max(0.5, lead_fy - 2.5))
    if c.role == "win_condition":
        if name in HEAVY and view.elixir < 9 and view.sim.elixir_multiplier == 1:
            return WIDTH / 2 - tc * 1.5, abs_y(me, 1.0)
        return lx, abs_y(me, 14.0)
    pushers = view.pushers(lane)
    if pushers:
        lead = max(pushers, key=lambda u: view.fy(u.y))
        return float(np.clip(lead.x, 0.5, WIDTH - 0.5)), abs_y(me, min(14.0, view.fy(lead.y) - 2.0))
    if c.role in ("ranged", "splash_air"):
        return lx, abs_y(me, 10.0)
    return lx, abs_y(me, 14.0)


# ------------------------------------------------------------------- coach
COUNTER_ROLES = {
    "building_targeter": ["tank_killer", "building", "swarm", "cycle", "frontline", "champion", "splash_ground"],
    "air": ["splash_air", "ranged", "air", "building", "champion", "small_spell"],
    "swarm": ["splash_ground", "splash_air", "small_spell", "frontline"],
    "tank_killer": ["swarm", "cycle", "building", "frontline", "splash_air"],
    "ranged": ["frontline", "tank_killer", "small_spell", "big_spell", "air"],
    "tank": ["tank_killer", "building", "swarm", "champion"],
    "other": ["frontline", "tank_killer", "splash_ground", "ranged", "swarm", "champion", "cycle"],
}


COACH_DEFAULTS = dict(
    enough_defense=0.8,    # skip defending a lane when our defenders have this share of the attackers' hit points
    hold_line=10.0,        # attackers closer than this (own-frame y) with no fitting card: save elixir instead
    trade_margin=1.0,      # cast a spell when it hits this much more elixir than it costs
    push_hp=800.0,         # support a counter-push of at least this many hit points
    punish_below=2.5,      # opponent elixir at or below this: attack with the win condition
    leak_single=9.0,       # play something at this elixir (single elixir) ...
    leak_double=6.5,       # ... and at this elixir in double elixir
)


class Coach:
    name = "coach"

    def __init__(self, placer=None, **params):
        self.placer = placer or place          # where cards go (strategy.place unless overridden)
        self.p = {**COACH_DEFAULTS, **params}

    def card_fit(self, view: View, name: str, prof: dict) -> bool:
        c = view.db.cards[name]
        if prof["air"] and not hits_air(c):
            only_air = prof["air"] and not prof["building_targeter"]
            if only_air or all(u.flying for u in view.enemies() if view.fy(u.y) < THREAT_LINE):
                return False
        if prof["swarm"] and c.role in ("small_spell", "big_spell"):
            return spell_spot(view, name)[0] >= c.elixir * 0.8
        if c.type == "spell" and not c.summons:
            return spell_spot(view, name)[0] >= c.elixir + 0.5
        if c.role in ("win_condition", "spawner", "utility") and c.name not in ("Goblin Giant",):
            return False
        return True

    def suggest(self, view: View) -> tuple[str | None, int]:
        card, lane, _ = self.plan(view)
        return card, lane

    def plan(self, view: View) -> tuple[str | None, int, str | None]:
        """(card or None, lane, reason): reason is finish, defend, hold (threat but no fitting card),
        trade, counterpush, punish, leak, or None."""
        db = view.db
        play = view.playable()
        roles = {n: db.cards[n].role for n in play}

        # 1. finish a tower with a spell
        for n in play:
            c = db.cards[n]
            if c.type == "spell" and c.projectile is not None and c.special != "rolling":
                dmg = c.projectile.damage * max(1, c.waves) * c.projectile.tower_factor
                for lane in (0, 1):
                    t = view.sim.tower(view.foe, "princess", lane)
                    if t.alive and t.hp <= dmg:
                        return n, lane, "finish"

        # 2. defend
        lanes = sorted((0, 1), key=lambda ln: -view.threat_hp(ln))
        for lane in lanes:
            threats = view.threats(lane)
            if not threats:
                continue
            prof = view.profile(threats)
            def_hp = sum(u.hp for u in view.defenders(lane))
            if def_hp >= self.p["enough_defense"] * prof["hp"] and not prof["building_targeter"]:
                continue
            keys = [k for k in ("air", "building_targeter", "tank", "swarm", "tank_killer", "ranged") if prof[k]] or ["other"]
            for key in keys:
                for role in COUNTER_ROLES[key]:
                    cands = [n for n in play if roles[n] == role and self.card_fit(view, n, prof)]
                    if cands:
                        return min(cands, key=lambda n: db.cards[n].elixir), lane, "defend"
            if min(view.fy(u.y) for u in threats) < self.p["hold_line"]:
                return None, lane, "hold"

        # 3. positive spell trades
        for n in play:
            c = db.cards[n]
            if c.role in ("small_spell", "big_spell"):
                for lane in (0, 1):
                    if spell_spot(view, n, lane)[0] >= c.elixir + self.p["trade_margin"]:
                        return n, lane, "trade"

        # 4. counter-push: support survivors, boost a big push
        for lane in (0, 1):
            push = view.pushers(lane)
            push_hp = sum(u.hp for u in push)
            if push_hp >= self.p["push_hp"] and view.elixir >= 4:
                if any(view.fy(u.y) > 20 for u in push) and push_hp >= 1500:
                    for n in play:
                        if n in ("Rage", "Freeze") and view.threats(lane) == []:
                            return n, lane, "counterpush"
                for role in ("ranged", "splash_air", "splash_ground", "air", "swarm", "frontline", "champion"):
                    cands = [n for n in play if roles[n] == role]
                    if cands:
                        return cands[0], lane, "counterpush"

        # 5. punish
        wins = [n for n in play if roles[n] == "win_condition"]
        back = view.enemy_back_tank()
        if wins and back is not None:
            return wins[0], 1 - back, "punish"
        if wins and view.enemy_elixir() <= self.p["punish_below"] and view.elixir >= db.cards[wins[0]].elixir:
            return wins[0], view.weak_lane(), "punish"

        # 6. don't leak elixir
        full = self.p["leak_single"] if view.sim.elixir_multiplier == 1 else self.p["leak_double"]
        if view.elixir >= full:
            lane = view.weak_lane()
            if wins:
                return wins[0], lane, "leak"
            for role in ("spawner", "champion", "frontline", "tank_killer", "splash_ground", "ranged", "swarm", "cycle"):
                cands = [n for n in play if roles[n] == role]
                if cands:
                    return cands[0], lane, "leak"
        return None, 0, None

    def decide(self, sim: Sim, player: int) -> Move | None:
        view = View(sim, player)
        card, lane = self.suggest(view)
        if card is None:
            return None
        return Move(card, lane, *self.placer(view, card, lane))


def auto_ability(sim: Sim, player: int) -> bool:
    """Use the champion's ability when it's fighting and elixir allows."""
    for u in sim.units:
        if u.alive and u.owner == player and u.spec is not None and u.spec.ability and u.deploy_left <= 0:
            ab = u.spec.ability
            if u.ability_cd > 0 or sim.players[player].elixir < ab.cost + 1:
                return False
            near = [o for o in sim.units if o.alive and o.owner != player and not o.timed
                    and math.dist((o.x, o.y), (u.x, u.y)) <= 5.5]
            if near or (u.target is not None and u.target.building and math.dist((u.x, u.y), (u.target.x, u.target.y)) < 4):
                return sim.use_ability(player)
    return False


class RandomBot:
    name = "random"

    def __init__(self, seed: int | None = None, rate: float = 0.5):
        self.rng = np.random.default_rng(seed)
        self.rate = rate

    def decide(self, sim: Sim, player: int) -> Move | None:
        view = View(sim, player)
        play = view.playable()
        if not play or self.rng.random() > self.rate:
            return None
        card = str(self.rng.choice(play))
        lane = int(self.rng.integers(2))
        c = sim.db.cards[card]
        if c.type == "spell" or card in DROP_ON_TOWER:
            x, y = place(view, card, lane)
        else:
            x = float(self.rng.uniform(0.5, WIDTH - 0.5))
            y = abs_y(player, float(self.rng.uniform(1.0, 14.0)))
        return Move(card, lane, x, y)


class BasicBot:
    """Defends with a random troop, otherwise waits for 8 elixir and plays a random card in a random lane."""

    name = "basic"

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def decide(self, sim: Sim, player: int) -> Move | None:
        view = View(sim, player)
        play = view.playable()
        troops = [c for c in play if sim.db.cards[c].type != "spell"]
        for lane in (0, 1):
            if troops and any(view.fy(u.y) < 12 for u in view.threats(lane)) and not view.defenders(lane):
                card = str(self.rng.choice(troops))
                return Move(card, lane, *place(view, card, lane))
        if view.elixir >= 8 and play:
            card = str(self.rng.choice(play))
            lane = int(self.rng.integers(2))
            return Move(card, lane, *place(view, card, lane))
        return None


def play_match(bot0, bot1, decks, seed: int | None = None, decision_every: float = 1.0) -> Sim:
    sim = Sim(decks, seed=seed)
    next_decision = 0.0
    while not sim.done:
        if sim.time >= next_decision - 1e-9:
            moves = [bot0.decide(sim, 0), bot1.decide(sim, 1)]
            for player, m in enumerate(moves):
                if m is not None:
                    sim.play(player, m.card, m.x, m.y)
                auto_ability(sim, player)
            next_decision += decision_every
        sim.step()
    return sim
