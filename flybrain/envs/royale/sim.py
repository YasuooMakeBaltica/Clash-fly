"""Full Clash Royale match simulator, driven by the card database (db.py).

Modelled: the 18x32 arena (river, bridges), princess/king towers (king wakes
when hit or a princess falls), elixir (double at 2:00, overtime with triple
elixir and sudden death, tower-HP tiebreak), 8-card decks cycling through a
4-card hand, deploy times, ground and air units, collisions (units push each
other; buildings don't move), lane pathing over bridges and river jumps,
target selection (air/ground, buildings-only, troops-only, sight range),
melee and projectile attacks, splash, self-centred splash, chain lightning,
piercing projectiles (Bowler, Executioner, Magic Archer, Log), scatter shots
(Hunter), minimum range (Mortar), multi-target (Electro Wizard), charge
(Prince, Ram), dashes (Bandit, Mega Knight), ramping damage (Inferno,
Mighty Miner), hit combos (Monk), shields, kamikaze units, spawners and
spawn-on-deploy, attached riders (Goblin Giant, Ram Rider), death damage,
death spawns, death effects, timed bombs, building decay, buffs (freeze,
stun, slow, rage, poison, earthquake, heal, invisibility, curses), area
spells, Tornado pull, Clone, Mirror, Graveyard, Lightning, rolling spells,
spell travel time, tunnelling (Miner, Goblin Drill), Elixir Collector,
Elixir Golem, Electro Giant reflect, Fisherman hook, champions (one at a
time) and their abilities.

Not modelled: card evolutions (no stats in the data), exact animation
timings, projectile deflection, and some fine targeting rules.
"""

from __future__ import annotations

import math

import numpy as np

from .db import DB, load

WIDTH, HEIGHT = 18.0, 32.0
RIVER_LO, RIVER_HI, RIVER_Y = 15.0, 17.0, 16.0
LANE_X = (3.5, 14.5)
DT = 0.1
REGULATION, OVERTIME_END = 180.0, 300.0
DOUBLE_AT, TRIPLE_AT = 120.0, 240.0
ELIXIR_RATE = 1 / 2.8
START_ELIXIR, MAX_ELIXIR = 5.0, 10.0
TUNNEL_SPEED = 10.0
CHARGE_AFTER = 2.5          # tiles walked without attacking before a charge starts
TOWER = dict(princess=dict(hp=3052, damage=109, hit_speed=0.8, range=7.5, radius=1.5),
             king=dict(hp=4824, damage=109, hit_speed=1.0, range=7.0, radius=2.0))


def frame_y(player: int, y: float) -> float:
    return y if player == 0 else HEIGHT - y


abs_y = frame_y


def lane_of(x: float) -> int:
    return 0 if x < WIDTH / 2 else 1


def side_of(y: float):
    return 0 if y < RIVER_LO else (1 if y > RIVER_HI else None)


class Unit:
    """A troop, building, crown tower or timed object (bomb, bottle, delivery)."""

    def __init__(self, sim: "Sim", spec, owner: int, x: float, y: float, card: str | None = None,
                 deploy: float | None = None, tower: str | None = None, lane: int | None = None):
        sim._uid += 1
        self.uid, self.sim, self.spec, self.owner = sim._uid, sim, spec, owner
        self.x, self.y = float(x), float(y)
        self.card, self.tower, self.lane = card, tower, lane
        if tower:
            t = TOWER[tower]
            self.hp = self.max_hp = t["hp"]
            self.radius, self.building, self.flying = t["radius"], True, False
            self.deploy_left, self.active = 0.0, tower == "princess"
            self.timed = False
        else:
            self.hp = self.max_hp = spec.hp
            self.radius, self.building, self.flying = spec.radius, spec.building, spec.flying
            self.deploy_left = spec.deploy_time if deploy is None else deploy
            self.active = True
            self.timed = spec.building and spec.hp <= 0
        self.shield = 0 if tower else spec.shield
        self.alive = True
        self.life_left = (spec.life_time if (spec is not None and spec.life_time) else None)
        self.cooldown = 0.0
        self.target = None
        self.retarget = 0.0
        self.buffs: dict[str, list] = {}       # name -> [buff, time_left, owner, tick_acc]
        self.walked = 0.0
        self.charging = False
        self.dash_cd = 0.0
        self.dashing = 0.0
        self.lock_time = 0.0
        self.combo = 0
        self.idle = 0.0
        self.parent: Unit | None = None
        self.children: list[Unit] = []
        self.ability_cd = 0.0
        self.ability_left = 0.0
        self.souls = 0
        self.hook_cd = 0.0
        self.elixir_timer = 0.0
        if spec is not None and spec.spawn_character is not None and not spec.spawn_attach:
            self.spawn_timer = spec.spawn_start or spec.spawn_pause or 1.0
            self.spawned = 0
        else:
            self.spawn_timer, self.spawned = None, 0

    # ----------------------------------------------------------- stats
    def factor(self, kind: str) -> float:
        f = 1.0
        for b, _, _, _ in self.buffs.values():
            f *= getattr(b, kind)
        if kind == "hit_speed" and self.ability_left > 0 and self.spec.ability and getattr(self.spec.ability, "hit_speed", 0):
            f *= self.spec.ability.hit_speed
        return f

    @property
    def invisible(self) -> bool:
        if any(b.invisible for b, *_ in self.buffs.values()):
            return True
        s = self.spec
        if s is None:
            return False
        if s.invisible_when_idle and self.idle >= 1.8:
            return True
        if s.ability and getattr(s.ability, "invisible", False) and self.ability_left > 0:
            return True
        if s.hides_when_idle and self.idle >= 0.8:
            return True
        return False

    @property
    def stunned(self) -> bool:
        return self.factor("hit_speed") <= 0.0


class Projectile:
    def __init__(self, sim, spec, owner, x, y, target=None, tx=None, ty=None, damage=None, source=None,
                 pierce=False, stop_on_hit=False, direction=None, chain_left=0, tower_factor=None):
        self.sim, self.spec, self.owner = sim, spec, owner
        self.x, self.y = x, y
        self.target = target
        self.tx = tx if tx is not None else (target.x if target else x)
        self.ty = ty if ty is not None else (target.y if target else y)
        self.damage = spec.damage if damage is None else damage
        self.source = source
        self.pierce, self.stop_on_hit = pierce, stop_on_hit
        self.hit: set[int] = set()
        self.travelled = 0.0
        self.returning = False
        self.chain_left = chain_left
        self.tower_factor = spec.tower_factor if tower_factor is None else tower_factor
        if direction is None:
            dx, dy = self.tx - x, self.ty - y
            d = math.hypot(dx, dy) or 1.0
            direction = (dx / d, dy / d)
        self.dir = direction
        self.alive = True


class Effect:
    def __init__(self, sim, spec, owner, x, y, delay=0.0, source=None):
        self.sim, self.spec, self.owner, self.x, self.y = sim, spec, owner, x, y
        self.delay = delay
        self.time_left = max(spec.duration, DT)
        self.next_hit = 0.0
        self.started = False
        self.struck: set[int] = set()
        self.spawn_next = spec.spawn_initial_delay if spec.spawn_character else None
        self.source = source
        self.alive = True


class Player:
    def __init__(self, deck: list[str], rng):
        order = list(rng.permutation(len(deck)))
        self.deck = list(deck)
        self.hand = [deck[i] for i in order[:4]]
        self.queue = [deck[i] for i in order[4:]]
        self.elixir = START_ELIXIR
        self.crowns = 0
        self.leaked = 0.0
        self.spent = 0.0
        self.last_card: str | None = None


class Sim:
    dt = DT

    def __init__(self, decks: tuple[list[str], list[str]], seed: int | None = None, db: DB | None = None,
                 shuffle_updates: bool = True):
        self.db = db or load()
        for d in decks:
            if len(d) != 8 or len(set(d)) != 8:
                raise ValueError(f"a deck needs 8 different cards: {d}")
            for n in d:
                if n not in self.db.cards:
                    raise ValueError(f"unknown card {n!r}")
        self.rng = np.random.default_rng(seed)
        self.shuffle_updates = shuffle_updates
        self.decks = [list(decks[0]), list(decks[1])]
        self.reset()

    # ------------------------------------------------------------ set-up
    def reset(self) -> None:
        self.time = 0.0
        self._uid = 0
        self.units: list[Unit] = []
        self.projectiles: list[Projectile] = []
        self.effects: list[Effect] = []
        self.events: list[dict] = []
        self.players = [Player(self.decks[0], self.rng), Player(self.decks[1], self.rng)]
        for owner in (0, 1):
            for lane, x in enumerate(LANE_X):
                self.units.append(Unit(self, None, owner, x, abs_y(owner, 6.5), tower="princess", lane=lane))
            king = Unit(self, None, owner, WIDTH / 2, abs_y(owner, 3.0), tower="king")
            king.active = False
            self.units.append(king)
        self.done = False
        self.winner: int | None = None
        self.overtime = False

    # ----------------------------------------------------------- queries
    def tower(self, owner: int, kind: str, lane: int | None = None) -> Unit:
        for u in self.units:
            if u.tower == kind and u.owner == owner and (kind == "king" or u.lane == lane):
                return u
        for u in getattr(self, "_dead_towers", []):
            if u.tower == kind and u.owner == owner and (kind == "king" or u.lane == lane):
                return u
        raise KeyError((owner, kind, lane))

    def towers(self, owner: int | None = None) -> list[Unit]:
        out = [u for u in self.units if u.tower] + list(getattr(self, "_dead_towers", []))
        return [u for u in out if owner is None or u.owner == owner]

    def lane_target(self, owner: int, lane: int) -> Unit:
        p = self.tower(1 - owner, "princess", lane)
        return p if p.alive else self.tower(1 - owner, "king")

    @property
    def elixir_multiplier(self) -> float:
        if self.time >= TRIPLE_AT:
            return 3.0
        return 2.0 if self.time >= DOUBLE_AT else 1.0

    def card_cost(self, player: int, name: str) -> int:
        c = self.db.cards[name]
        if c.special == "mirror":
            last = self.players[player].last_card
            return (self.db.cards[last].elixir + 1) if last else 99
        return c.elixir

    def champion_alive(self, player: int) -> bool:
        return any(u.alive and u.owner == player and u.card and self.db.cards[u.card].champion and not u.parent
                   for u in self.units)

    def can_play(self, player: int, name: str) -> bool:
        p = self.players[player]
        if self.done or name not in p.hand or p.elixir < self.card_cost(player, name):
            return False
        c = self.db.cards[name]
        if c.special == "mirror":
            last = p.last_card
            if not last or self.db.cards[last].champion:
                return False
            c = self.db.cards[last]
        if c.champion and self.champion_alive(player):
            return False
        return True

    def valid_position(self, player: int, name: str, x: float, y: float) -> bool:
        if not (0.5 <= x <= WIDTH - 0.5 and 0.5 <= y <= HEIGHT - 0.5):
            return False
        c = self.db.cards[name]
        if c.special == "mirror":
            c = self.db.cards[self.players[player].last_card] if self.players[player].last_card else c
        if c.type == "spell" and not c.summons:
            return True
        if c.type == "spell" or c.deploy_anywhere:
            return side_of(y) is not None
        fy = frame_y(player, y)
        if fy <= RIVER_LO - 0.5:
            return True
        enemy_princess = self.tower(1 - player, "princess", lane_of(x))
        return fy <= 20.0 and not enemy_princess.alive and c.type != "building"

    # ----------------------------------------------------------- actions
    def play(self, player: int, name: str, x: float, y: float) -> bool:
        if not self.can_play(player, name) or not self.valid_position(player, name, x, y):
            return False
        p = self.players[player]
        cost = self.card_cost(player, name)
        p.elixir -= cost
        p.spent += cost
        p.hand[p.hand.index(name)] = p.queue.pop(0)
        p.queue.append(name)
        card = self.db.cards[name]
        if card.special == "mirror":
            card = self.db.cards[p.last_card]
        else:
            p.last_card = name
        self.events.append(dict(t=round(self.time, 2), kind="play", player=player, card=card.name, x=round(x, 2),
                                y=round(y, 2)))
        self._deploy(player, card, x, y)
        return True

    def _deploy(self, player: int, card, x: float, y: float) -> None:
        king = self.tower(player, "king")
        if card.projectile is not None and card.type == "spell":
            spec = card.projectile
            if card.special == "rolling":
                d = 1 if player == 0 else -1
                self.projectiles.append(Projectile(self, spec, player, x, y, tx=x, ty=y + d, pierce=True,
                                                   direction=(0.0, float(d))))
            else:
                for w in range(card.waves):
                    pr = Projectile(self, spec, player, king.x, king.y, tx=x, ty=y)
                    pr.travelled = -w * 0.3 * spec.speed   # later waves start a bit behind
                    self.projectiles.append(pr)
        if card.effect is not None:
            delay = 0.0
            if card.type == "spell" and not card.summons:
                delay = 0.3 if card.effect.duration <= 0.01 else 0.0
            self.effects.append(Effect(self, card.effect, player, x, y, delay=delay))
        if not card.summons:
            return
        tunnel = 0.0
        if card.deploy_anywhere and card.type != "spell":
            tunnel = math.dist((king.x, king.y), (x, y)) / TUNNEL_SPEED
        positions = self._formation(card, player, x, y)
        k = 0
        for spec, count in card.summons:
            for _ in range(count):
                px, py = positions[k % len(positions)]
                k += 1
                u = Unit(self, spec, player, px, py, card=card.name,
                         deploy=spec.deploy_time + tunnel + 0.1 * (k - 1) * (card.radius > 0))
                self._add(u)
                if spec.spawn_attach and spec.spawn_character is not None:
                    for i in range(max(1, spec.spawn_number)):
                        child = Unit(self, spec.spawn_character, player, px, py, card=card.name, deploy=u.deploy_left)
                        child.parent = u
                        u.children.append(child)
                        self._add(child)
        if card.projectile is not None and card.type != "spell":     # Mega Knight landing damage
            pr = Projectile(self, card.projectile, player, x, y, tx=x, ty=y)
            pr.travelled = -1.0 * card.projectile.speed
            self.projectiles.append(pr)

    def _formation(self, card, player, x, y):
        n = sum(c for _, c in card.summons)
        if n <= 1:
            return [(x, y)]
        if card.full_lane:
            xs = np.linspace(2.5, WIDTH - 2.5, n)
            return [(float(px), y) for px in xs]
        if card.name == "Royal Hogs":
            return [(float(np.clip(x + dx, 0.5, WIDTH - 0.5)), y) for dx in (-1.8, -0.6, 0.6, 1.8)]
        r = card.radius or 0.7
        out = []
        for i in range(n):
            a = 2 * math.pi * i / n + math.pi / 2
            out.append((float(np.clip(x + r * math.cos(a), 0.5, WIDTH - 0.5)), float(np.clip(y + r * math.sin(a), 0.5, HEIGHT - 0.5))))
        return out

    def _add(self, u: Unit) -> Unit:
        self.units.append(u)
        if u.spec is not None and u.spec.spawn_effect is not None and not u.parent:
            self.effects.append(Effect(self, u.spec.spawn_effect, u.owner, u.x, u.y, delay=u.deploy_left))
        return u

    def use_ability(self, player: int) -> bool:
        """Activate the player's champion ability, if ready and affordable."""
        for u in self.units:
            if u.alive and u.owner == player and u.spec is not None and u.spec.ability and u.deploy_left <= 0:
                ab = u.spec.ability
                p = self.players[player]
                if u.ability_cd > 0 or p.elixir < ab.cost:
                    return False
                p.elixir -= ab.cost
                p.spent += ab.cost
                u.ability_cd = ab.cooldown
                self._ability(u, ab)
                self.events.append(dict(t=round(self.time, 2), kind="ability", player=player, card=u.card))
                return True
        return False

    def _ability(self, u: Unit, ab) -> None:
        if getattr(ab, "duration", 0):
            u.ability_left = ab.duration
        if getattr(ab, "chain", 0):                   # Golden Knight: dash from enemy to enemy
            hit = set()
            for _ in range(ab.chain):
                cands = [o for o in self.units if self._enemy(u, o) and not o.building and o.uid not in hit
                         and math.dist((u.x, u.y), (o.x, o.y)) <= u.spec.dash_secondary_range]
                if not cands:
                    break
                o = min(cands, key=lambda o: math.dist((u.x, u.y), (o.x, o.y)))
                u.x, u.y = o.x, o.y
                self._damage(o, u.spec.dash_damage, u)
                hit.add(o.uid)
        if getattr(ab, "summon", None):               # Skeleton King: skeletons from collected souls
            n = int(np.clip(ab.min_count + u.souls, ab.min_count, ab.max_count))
            u.souls = 0
            spec = self.db.characters[ab.summon]
            for i in range(n):
                a = 2 * math.pi * i / n
                self._add(Unit(self, spec, u.owner, u.x + 1.5 * math.cos(a), u.y + 1.5 * math.sin(a), deploy=0.5))
        if getattr(ab, "bomb_damage", 0):             # Mighty Miner: bomb, then switch lanes
            self._area_damage(u.owner, u.x, u.y, ab.bomb_radius, ab.bomb_damage, True, True, 1.0, source=u)
            u.x = WIDTH - u.x
            u.target = None

    # ---------------------------------------------------------- dynamics
    def step(self) -> None:
        if self.done:
            return
        dt = self.dt
        self.time += dt
        rate = ELIXIR_RATE * self.elixir_multiplier
        for p in self.players:
            gain = rate * dt
            p.leaked += max(0.0, gain - (MAX_ELIXIR - p.elixir))
            p.elixir = min(MAX_ELIXIR, p.elixir + gain)
        for e in list(self.effects):
            self._update_effect(e, dt)
        self.effects = [e for e in self.effects if e.alive]
        for pr in list(self.projectiles):
            self._update_projectile(pr, dt)
        self.projectiles = [p for p in self.projectiles if p.alive]
        order = self.rng.permutation(len(self.units)) if self.shuffle_updates else range(len(self.units))
        units = list(self.units)
        for i in order:
            u = units[i]
            if u.alive:
                self._update_unit(u, dt)
        self._collide()
        for u in units:
            if u.alive and u.hp <= 0 and not u.timed:
                self._kill(u)
        self.units = [u for u in self.units if u.alive]
        self._check_end()

    def _check_end(self) -> None:
        c0, c1 = self.players[0].crowns, self.players[1].crowns
        if self.done:
            return
        if self.time >= REGULATION - 1e-9 and not self.overtime:
            if c0 != c1:
                return self._finish()
            self.overtime = True
            self._overtime_crowns = (c0, c1)
        if self.overtime and (c0, c1) != self._overtime_crowns:
            return self._finish()
        if self.time >= OVERTIME_END - 1e-9:
            lows = [min(t.hp / t.max_hp for t in self.towers(o)) for o in (0, 1)]
            self.done = True
            self.winner = 0 if lows[0] > lows[1] else 1 if lows[1] > lows[0] else None

    def _finish(self) -> None:
        self.done = True
        c0, c1 = self.players[0].crowns, self.players[1].crowns
        self.winner = 0 if c0 > c1 else 1 if c1 > c0 else None

    # ------------------------------------------------------------ helpers
    def _enemy(self, u: Unit, o: Unit) -> bool:
        return o.alive and o.owner != u.owner

    def _targetable(self, attacker: Unit, o: Unit) -> bool:
        if not self._enemy(attacker, o) or o.timed or o.parent is not None or o.invisible:
            return False
        if o.deploy_left > 0:          # troops and towers ignore units still deploying (spells don't)
            return False
        s = attacker.spec
        if attacker.tower:
            return True
        if o.flying and not s.attacks_air:
            return False
        if not o.flying and not s.attacks_ground:
            return False
        if s.buildings_only and not o.building:
            return False
        if s.troops_only and o.building:
            return False
        return True

    @staticmethod
    def _gap(a: Unit, b: Unit) -> float:
        return math.dist((a.x, a.y), (b.x, b.y)) - b.radius - (a.radius if not a.tower else 0.0)

    def _damage(self, o: Unit, amount: float, source: Unit | None = None, tower_factor: float = 1.0,
                building_factor: float = 1.0) -> None:
        if not o.alive or amount <= 0 or o.timed:
            return
        if o.tower:
            amount *= tower_factor
        elif o.building:
            amount *= building_factor
        if o.spec is not None and o.spec.ability and o.ability_left > 0 and getattr(o.spec.ability, "damage_taken", 0):
            amount *= o.spec.ability.damage_taken
        if o.shield > 0:
            o.shield = max(0.0, o.shield - amount)
        else:
            o.hp -= amount
        o.idle = 0.0
        if o.tower == "king":
            o.active = True
        if (source is not None and source.alive and o.spec is not None and o.spec.reflect_damage
                and math.dist((o.x, o.y), (source.x, source.y)) <= o.spec.reflect_radius + source.radius + o.radius):
            self._damage(source, o.spec.reflect_damage, None, tower_factor=0.8)
            self._buff(source, self.db.buffs["ZapFreeze"], 0.5, o.owner)

    def _buff(self, o: Unit, buff, time: float, owner: int) -> None:
        if buff is None or not o.alive or o.tower and (buff.speed == 0 or buff.switch_team):
            if buff is not None and o.tower and buff.speed == 0:
                pass  # towers can be frozen in the real game; keep it simple: they can
            else:
                return
        if buff.hit_speed <= 0 and not o.tower:  # stun/freeze resets charges and ramps
            o.lock_time, o.charging, o.walked = 0.0, False, 0.0
            if o.spec is not None and o.spec.load_first_hit:
                o.cooldown = o.spec.hit_speed
        cur = o.buffs.get(buff.name)
        if cur is None or cur[1] < time:
            o.buffs[buff.name] = [buff, time, owner, 0.0]

    def _area_damage(self, owner, x, y, radius, damage, air, ground, tower_factor, source=None, buff=None,
                     buff_time=0.0, pushback=0.0, exclude=None):
        hit = []
        for o in self.units:
            if not o.alive or o.owner == owner or o.timed or o.parent is not None:
                continue
            if (o.flying and not air) or (not o.flying and not ground):
                continue
            if math.dist((x, y), (o.x, o.y)) <= radius + o.radius:
                if exclude is not None and o.uid in exclude:
                    continue
                self._damage(o, damage, source, tower_factor=tower_factor)
                if buff is not None:
                    self._buff(o, buff, buff_time, owner)
                if pushback and not o.building and not (o.spec and o.spec.ignore_pushback):
                    self._push(o, x, y, pushback)
                hit.append(o)
        return hit

    def _push(self, o: Unit, x: float, y: float, dist: float) -> None:
        dx, dy = o.x - x, o.y - y
        d = math.hypot(dx, dy) or 1.0
        o.x = float(np.clip(o.x + dx / d * dist, 0.5, WIDTH - 0.5))
        o.y = float(np.clip(o.y + dy / d * dist, 0.5, HEIGHT - 0.5))
        o.walked, o.charging = 0.0, False

    # ------------------------------------------------------------- units
    def _update_unit(self, u: Unit, dt: float) -> None:
        for name in list(u.buffs):
            b, t, owner, acc = u.buffs[name]
            if b.dps or b.heal_per_second:
                freq = b.hit_frequency or 1.0
                acc += dt
                while acc >= freq:
                    acc -= freq
                    if b.dps:
                        self._damage(u, b.dps * freq, None, tower_factor=b.tower_factor, building_factor=b.building_factor)
                    if b.heal_per_second and not u.tower:
                        u.hp = min(u.max_hp, u.hp + b.heal_per_second * freq)
            t -= dt
            if t <= 0:
                del u.buffs[name]
            else:
                u.buffs[name] = [b, t, owner, acc]
        u.ability_cd = max(0.0, u.ability_cd - dt)
        u.ability_left = max(0.0, u.ability_left - dt)
        if u.timed:
            u.deploy_left -= dt
            if u.deploy_left <= 0:
                self._kill(u)
            return
        if u.deploy_left > 0:
            u.deploy_left -= dt
            return
        if u.tower:
            if u.active:
                self._attack_logic(u, dt, can_move=False)
            return
        s = u.spec
        if u.life_left is not None:
            u.life_left -= dt
            u.hp -= u.max_hp / s.life_time * dt
            if u.life_left <= 0:
                u.hp = 0
                return
        if getattr(s, "elixir_every", 0):
            u.elixir_timer += dt
            if u.elixir_timer >= s.elixir_every:
                u.elixir_timer -= s.elixir_every
                p = self.players[u.owner]
                p.elixir = min(MAX_ELIXIR, p.elixir + 1)
        if u.parent is not None:
            if not u.parent.alive:
                u.hp = 0
                return
            u.x, u.y = u.parent.x, u.parent.y
        if u.spawn_timer is not None:
            u.spawn_timer -= dt * u.factor("spawn_speed")
            if u.spawn_timer <= 0:
                n = max(1, s.spawn_number)
                for i in range(n):
                    a = 2 * math.pi * (i + u.spawned) / max(n, 3)
                    r = s.spawn_radius or 1.0
                    self._add(Unit(self, s.spawn_character, u.owner, u.x + r * math.cos(a) * 0.7,
                                   u.y + r * math.sin(a) * 0.7, card=u.card, deploy=0.4))
                u.spawned += n
                u.spawn_timer = s.spawn_pause or s.spawn_interval or 5.0
                if s.spawn_limit and u.spawned >= s.spawn_limit:
                    u.alive = False
                    return
        if s.heal_when_idle and u.idle >= 1.0:
            u.hp = min(u.max_hp, u.hp + 0.03 * u.max_hp * dt)
        u.dash_cd = max(0.0, u.dash_cd - dt)
        u.hook_cd = max(0.0, u.hook_cd - dt)
        if u.dashing > 0:
            u.dashing -= dt
            return
        self._attack_logic(u, dt, can_move=s.speed > 0 and u.parent is None)

    def _find_target(self, u: Unit):
        sight = TOWER[u.tower]["range"] if u.tower else u.spec.sight
        best, best_d = None, sight
        for o in self.units:
            if self._targetable(u, o):
                d = self._gap(u, o)
                if d <= best_d:
                    best, best_d = o, d
        if best is None and not u.tower and u.spec.speed > 0:
            if u.spec.buildings_only:
                blds = [o for o in self.units if self._targetable(u, o)]
                if blds:
                    best = min(blds, key=lambda o: math.dist((u.x, u.y), (o.x, o.y)))
            if best is None and not u.spec.troops_only:
                best = self.lane_target(u.owner, lane_of(u.x))
        return best

    def _attack_logic(self, u: Unit, dt: float, can_move: bool) -> None:
        hit_f = u.factor("hit_speed")
        t = u.target
        rng = TOWER[u.tower]["range"] if u.tower else u.spec.range
        if t is not None and (not t.alive or t.invisible or (t.parent is not None)):
            t = None
        if t is not None and not t.building and not u.tower and self._gap(u, t) > u.spec.sight + 1.5:
            t = None
        if t is not None and u.tower and self._gap(u, t) > rng:
            t = None
        u.retarget -= dt
        if t is None or (u.retarget <= 0 and self._gap(u, t) > rng):
            u.retarget = 0.5
            nt = self._find_target(u)
            if nt is not t:
                u.lock_time = 0.0
                if nt is not None and not u.tower:
                    u.cooldown = max(u.cooldown, u.spec.load_time * 0.5)
            t = nt
        u.target = t
        if hit_f <= 0:          # frozen or stunned
            return
        u.cooldown -= dt * hit_f
        u.idle += dt
        if t is None:
            return
        gap = self._gap(u, t)
        min_rng = 0.0 if u.tower else u.spec.min_range
        if not u.tower and self._special_move(u, t, gap):
            return
        if gap <= rng and gap >= min_rng - 1e-6:
            u.lock_time += dt
            u.idle = 0.0
            if u.cooldown <= 0:
                self._attack(u, t)
                u.cooldown = TOWER[u.tower]["hit_speed"] if u.tower else u.spec.hit_speed
                u.walked, u.charging = 0.0, False
            return
        if can_move:
            self._move(u, t, dt)

    def _special_move(self, u: Unit, t: Unit, gap: float) -> bool:
        s = u.spec
        # dash (Bandit, Mega Knight)
        if s.dash_damage and s.dash_max and u.dash_cd <= 0 and s.dash_min <= gap <= s.dash_max and not t.building \
                and not u.charging:
            dx, dy = t.x - u.x, t.y - u.y
            d = math.hypot(dx, dy) or 1.0
            u.x, u.y = t.x - dx / d * (t.radius + u.radius), t.y - dy / d * (t.radius + u.radius)
            if s.dash_radius:
                self._area_damage(u.owner, u.x, u.y, s.dash_radius, s.dash_damage, False, True, 1.0, source=u,
                                  pushback=0.5)
            else:
                self._damage(t, s.dash_damage, u)
            u.dash_cd = s.dash_cooldown or 1.0
            u.dashing = 0.3
            u.cooldown = max(u.cooldown, s.hit_speed * 0.5)
            return True
        # Fisherman hook
        if s.special_range and u.hook_cd <= 0 and s.special_min_range <= gap <= s.special_range:
            if t.building:
                dx, dy = t.x - u.x, t.y - u.y
                d = math.hypot(dx, dy) or 1.0
                u.x, u.y = t.x - dx / d * (t.radius + u.radius + 0.5), t.y - dy / d * (t.radius + u.radius + 0.5)
            elif not (t.spec and t.spec.ignore_pushback and t.spec.mass >= 18):
                dx, dy = u.x - t.x, u.y - t.y
                d = math.hypot(dx, dy) or 1.0
                t.x, t.y = u.x - dx / d * (t.radius + u.radius + 0.3), u.y - dy / d * (t.radius + u.radius + 0.3)
                if s.special_projectile and s.special_projectile.buff:
                    self._buff(t, s.special_projectile.buff, s.special_projectile.buff_time, u.owner)
            u.hook_cd = 8.0
            return True
        return False

    def _move(self, u: Unit, t: Unit, dt: float) -> None:
        s = u.spec
        speed = s.speed * u.factor("speed")
        if s.charge_damage and u.walked >= CHARGE_AFTER:
            u.charging = True
        if u.charging:
            speed *= s.charge_speed or 2.0
        if speed <= 0:
            return
        wx, wy = t.x, t.y
        if not u.flying and not s.jump:
            su, st = side_of(u.y), side_of(t.y)
            if st is None:
                st = 0 if t.y < RIVER_Y else 1
            if su is not None and su != st:
                bx = min(LANE_X, key=lambda lx: abs(lx - u.x))
                wx, wy = bx, RIVER_Y
        dx, dy = wx - u.x, wy - u.y
        d = math.hypot(dx, dy)
        if d < 1e-6:
            return
        step = min(d, speed * dt)
        u.x += dx / d * step
        u.y += dy / d * step
        u.walked += step

    def _attack(self, u: Unit, t: Unit) -> None:
        if u.tower:
            self._damage(t, TOWER[u.tower]["damage"], u)
            return
        s = u.spec
        dmg = s.damage
        pushback = 0.0
        if s.vd is not None:
            if s.vd.mode == "time":
                st = s.vd.stage_time
                stage = 0 if u.lock_time < st[0] else (1 if u.lock_time < st[0] + st[1] else 2)
                dmg = s.vd.damage[stage]
            else:
                dmg = s.vd.damage[u.combo % 3]
                if u.combo % 3 == 2:
                    pushback = s.vd.pushback
                u.combo += 1
        if u.charging and s.charge_damage:
            dmg = s.charge_damage
        targets = [t]
        if s.targets > 1:
            others = sorted((o for o in self.units if o is not t and self._targetable(u, o)
                             and self._gap(u, o) <= s.range), key=lambda o: self._gap(u, o))
            targets += others[: s.targets - 1]
        for tg in targets:
            if s.projectile is not None:
                self._shoot(u, tg, s.projectile)
            else:
                if s.splash:
                    cx, cy = (u.x, u.y) if s.self_splash else (tg.x, tg.y)
                    self._area_damage(u.owner, cx, cy, s.splash, dmg, s.attacks_air, s.attacks_ground,
                                      s.tower_factor, source=u, pushback=pushback)
                else:
                    self._damage(tg, dmg, u, tower_factor=s.tower_factor)
                    if pushback and not tg.building:
                        self._push(tg, u.x, u.y, pushback)
            if s.buff_on_damage is not None:
                self._buff(tg, s.buff_on_damage, s.buff_on_damage_time, u.owner)
        if s.attack_pushback:
            self._push(u, t.x, t.y, s.attack_pushback * 0.5)
        if s.heal_when_idle:   # Battle Healer heals nearby friends when she hits
            for o in self.units:
                if o.alive and o.owner == u.owner and not o.building and math.dist((o.x, o.y), (u.x, u.y)) <= 4.0:
                    o.hp = min(o.max_hp, o.hp + 48)
        if s.kamikaze:
            u.hp = 0
            u.alive_kamikaze = True

    def _shoot(self, u: Unit, t: Unit, spec) -> None:
        s = u.spec
        if s.projectiles > 1 and spec.pierce_range:          # scatter (Hunter)
            base = math.atan2(t.y - u.y, t.x - u.x)
            for i in range(s.projectiles):
                a = base + (i - (s.projectiles - 1) / 2) * 0.06
                self.projectiles.append(Projectile(self, spec, u.owner, u.x, u.y, tx=t.x, ty=t.y, source=u,
                                                   pierce=True, stop_on_hit=True, direction=(math.cos(a), math.sin(a))))
            return
        if spec.pierce_range > 1.0:                           # Bowler, Executioner, Magic Archer
            self.projectiles.append(Projectile(self, spec, u.owner, u.x, u.y, tx=t.x, ty=t.y, source=u, pierce=True))
            return
        self.projectiles.append(Projectile(self, spec, u.owner, u.x, u.y, target=t, source=u,
                                           chain_left=spec.chain_count))

    # ------------------------------------------------------- projectiles
    def _update_projectile(self, pr: Projectile, dt: float) -> None:
        spec = pr.spec
        step = spec.speed * dt
        if pr.travelled < 0:                       # delayed start (spell waves, landing)
            pr.travelled += step
            return
        if pr.pierce:
            pr.x += pr.dir[0] * step
            pr.y += pr.dir[1] * step
            pr.travelled += step
            r = spec.pierce_radius or spec.radius or 0.5
            for o in self.units:
                if not o.alive or o.owner == pr.owner or o.uid in pr.hit or o.timed or o.parent is not None:
                    continue
                if (o.flying and not spec.air) or (not o.flying and not spec.ground):
                    continue
                if math.dist((pr.x, pr.y), (o.x, o.y)) <= r + o.radius:
                    pr.hit.add(o.uid)
                    self._damage(o, pr.damage, pr.source, tower_factor=pr.tower_factor)
                    if spec.buff is not None:
                        self._buff(o, spec.buff, spec.buff_time, pr.owner)
                    if spec.pushback and not o.building and not (o.spec and o.spec.ignore_pushback):
                        o.x = float(np.clip(o.x + pr.dir[0] * spec.pushback, 0.5, WIDTH - 0.5))
                        o.y = float(np.clip(o.y + pr.dir[1] * spec.pushback, 0.5, HEIGHT - 0.5))
                    if pr.stop_on_hit:
                        pr.alive = False
                        return
            limit = spec.pierce_range or 6.0
            if pr.travelled >= limit or not (0 <= pr.x <= WIDTH and 0 <= pr.y <= HEIGHT):
                if spec.returns and not pr.returning:
                    pr.returning = True
                    pr.travelled = 0.0
                    pr.hit.clear()
                    pr.dir = (-pr.dir[0], -pr.dir[1])
                    return
                if spec.spawn_character is not None:    # Barbarian Barrel
                    for _ in range(max(1, spec.spawn_count)):
                        self._add(Unit(self, spec.spawn_character, pr.owner, pr.x, pr.y, deploy=0.5))
                pr.alive = False
            return
        if spec.homing and pr.target is not None and pr.target.alive:
            pr.tx, pr.ty = pr.target.x, pr.target.y
        dx, dy = pr.tx - pr.x, pr.ty - pr.y
        d = math.hypot(dx, dy)
        if d <= step:
            pr.x, pr.y = pr.tx, pr.ty
            self._impact(pr)
            pr.alive = False
        else:
            pr.x += dx / d * step
            pr.y += dy / d * step

    def _impact(self, pr: Projectile) -> None:
        spec = pr.spec
        hit = []
        if spec.radius > 0:
            hit = self._area_damage(pr.owner, pr.x, pr.y, spec.radius, pr.damage, spec.air or not spec.ground,
                                    spec.ground or not spec.air, pr.tower_factor, source=pr.source, buff=spec.buff,
                                    buff_time=spec.buff_time, pushback=spec.pushback)
        elif pr.target is not None and pr.target.alive:
            self._damage(pr.target, pr.damage, pr.source, tower_factor=pr.tower_factor)
            if spec.buff is not None:
                self._buff(pr.target, spec.buff, spec.buff_time, pr.owner)
            hit = [pr.target]
        if spec.chain_count and hit:
            done = {o.uid for o in hit}
            cur = hit[0]
            for _ in range(spec.chain_count - 1):
                cands = [o for o in self.units if o.alive and o.owner != pr.owner and o.uid not in done and not o.timed
                         and o.parent is None and math.dist((cur.x, cur.y), (o.x, o.y)) <= spec.chain_radius]
                if not cands:
                    break
                cur = min(cands, key=lambda o: math.dist((cur.x, cur.y), (o.x, o.y)))
                done.add(cur.uid)
                self._damage(cur, pr.damage, pr.source, tower_factor=pr.tower_factor)
                if spec.buff is not None:
                    self._buff(cur, spec.buff, spec.buff_time, pr.owner)
        if spec.spawn_character is not None:
            for i in range(max(1, spec.spawn_count)):
                a = 2 * math.pi * i / max(1, spec.spawn_count)
                self._add(Unit(self, spec.spawn_character, pr.owner, pr.x + 0.6 * math.cos(a) * (spec.spawn_count > 1),
                               pr.y + 0.6 * math.sin(a) * (spec.spawn_count > 1), deploy=0.5))
        if spec.spawn_projectile is not None:           # Firecracker shards fly on past the target
            for i in range(max(1, spec.spawn_projectile_count)):
                a = math.atan2(pr.dir[1], pr.dir[0]) + (i - 2) * 0.12
                self.projectiles.append(Projectile(self, spec.spawn_projectile, pr.owner, pr.x, pr.y, source=pr.source,
                                                   pierce=True, direction=(math.cos(a), math.sin(a)),
                                                   tx=pr.x + math.cos(a), ty=pr.y + math.sin(a)))

    # ----------------------------------------------------------- effects
    def _update_effect(self, e: Effect, dt: float) -> None:
        if e.delay > 0:
            e.delay -= dt
            return
        spec = e.spec
        if not e.started:
            e.started = True
            if spec.damage and not spec.hit_biggest:
                self._area_damage(e.owner, e.x, e.y, spec.radius, spec.damage, spec.air, spec.ground,
                                  spec.tower_factor, buff=spec.buff, buff_time=spec.buff_time, pushback=spec.pushback)
            if spec.clone:
                for o in list(self.units):
                    if (o.alive and o.owner == e.owner and not o.building and not o.tower and o.parent is None
                            and math.dist((o.x, o.y), (e.x, e.y)) <= spec.radius + o.radius):
                        c = Unit(self, o.spec, o.owner, o.x + 0.4, o.y, card=o.card, deploy=0.0)
                        c.hp = c.max_hp = 1
                        c.shield = 0
                        self._add(c)
        e.next_hit -= dt
        if e.next_hit <= 0:
            e.next_hit = spec.hit_interval or 1e9
            if spec.hit_biggest and spec.projectile is not None:      # Lightning
                cands = [o for o in self.units if o.alive and o.owner != e.owner and not o.timed and o.parent is None
                         and o.uid not in e.struck and math.dist((o.x, o.y), (e.x, e.y)) <= spec.radius + o.radius]
                if cands and len(e.struck) < 3:
                    o = max(cands, key=lambda o: o.hp + o.shield)
                    e.struck.add(o.uid)
                    p = spec.projectile
                    self._damage(o, p.damage, None, tower_factor=p.tower_factor)
                    if p.buff is not None:
                        self._buff(o, p.buff, p.buff_time, e.owner)
            elif spec.buff is not None and not (spec.damage and spec.duration <= 0.01):
                for o in self.units:
                    if not o.alive or o.timed or o.parent is not None:
                        continue
                    if spec.only_own != (o.owner == e.owner):
                        continue
                    if spec.ignore_buildings and o.building:
                        continue
                    if (o.flying and not spec.air) or (not o.flying and not spec.ground):
                        continue
                    if math.dist((o.x, o.y), (e.x, e.y)) <= spec.radius + o.radius:
                        self._buff(o, spec.buff, spec.buff_time or spec.hit_interval, e.owner)
        if spec.buff is not None and spec.buff.pull:                 # Tornado drags units to its centre
            for o in self.units:
                if (o.alive and o.owner != e.owner and not o.building and o.parent is None
                        and math.dist((o.x, o.y), (e.x, e.y)) <= spec.radius + o.radius):
                    dx, dy = e.x - o.x, e.y - o.y
                    d = math.hypot(dx, dy)
                    if d > 0.2:
                        k = min(d, 4.5 * dt / max(1.0, (o.spec.mass if o.spec else 6) / 6))
                        o.x += dx / d * k
                        o.y += dy / d * k
        if e.spawn_next is not None:                                 # Graveyard
            e.spawn_next -= dt
            if e.spawn_next <= 0:
                e.spawn_next = spec.spawn_interval or 0.5
                a = self.rng.uniform(0, 2 * math.pi)
                r = self.rng.uniform(0.5, spec.radius)
                x = float(np.clip(e.x + r * math.cos(a), 0.5, WIDTH - 0.5))
                y = float(np.clip(e.y + r * math.sin(a), 0.5, HEIGHT - 0.5))
                self._add(Unit(self, spec.spawn_character, e.owner, x, y, deploy=0.5))
        e.time_left -= dt
        if e.time_left <= 0:
            e.alive = False

    # --------------------------------------------------------- collisions
    def _collide(self) -> None:
        movers = [u for u in self.units if u.alive and not u.building and not u.timed and u.parent is None
                  and u.deploy_left <= 0]
        solids = [u for u in self.units if u.alive and u.building and not u.timed]
        for i, a in enumerate(movers):
            for b in movers[i + 1:]:
                if a.flying != b.flying:
                    continue
                dx, dy = b.x - a.x, b.y - a.y
                d = math.hypot(dx, dy)
                overlap = a.radius + b.radius - d
                if overlap > 0:
                    if d < 1e-6:
                        dx, dy, d = 0.01 * (1 if a.uid < b.uid else -1), 0.0, 0.01
                    ma, mb = a.spec.mass, b.spec.mass
                    wa, wb = mb / (ma + mb), ma / (ma + mb)
                    push = min(overlap, 0.3) * 0.5
                    a.x -= dx / d * push * wa * 2
                    a.y -= dy / d * push * wa * 2
                    b.x += dx / d * push * wb * 2
                    b.y += dy / d * push * wb * 2
            if not a.flying:
                for b in solids:
                    dx, dy = a.x - b.x, a.y - b.y
                    d = math.hypot(dx, dy)
                    overlap = a.radius + b.radius - d
                    if overlap > 0 and d > 1e-6:
                        a.x += dx / d * min(overlap, 0.3)
                        a.y += dy / d * min(overlap, 0.3)
            a.x = float(np.clip(a.x, 0.5, WIDTH - 0.5))
            a.y = float(np.clip(a.y, 0.5, HEIGHT - 0.5))

    # ------------------------------------------------------------- deaths
    def _kill(self, u: Unit) -> None:
        if not u.alive and not u.timed:
            return
        u.alive = False
        s = u.spec
        if u.tower:
            self._dead_towers = getattr(self, "_dead_towers", []) + [u]
            u.hp = 0
            attacker = 1 - u.owner
            self.events.append(dict(t=round(self.time, 2), kind="tower", player=attacker, tower=u.tower, lane=u.lane))
            if u.tower == "king":
                self.players[attacker].crowns = 3
                self._finish()
            else:
                self.players[attacker].crowns += 1
                self.tower(u.owner, "king").active = True
            return
        for c in u.children:
            if c.alive:
                c.hp = 0
        if s.death_damage:
            self._area_damage(u.owner, u.x, u.y, s.death_damage_radius, s.death_damage, s.attacks_air or True,
                              True, s.tower_factor, pushback=s.death_pushback)
        for spec, count in ((s.death_spawn, s.death_spawn_count), (s.death_spawn2, s.death_spawn_count2)):
            if spec is not None:
                for i in range(count):
                    a = 2 * math.pi * i / max(1, count)
                    r = s.death_spawn_radius if count > 1 else 0.0
                    self._add(Unit(self, spec, u.owner, u.x + r * math.cos(a), u.y + r * math.sin(a), card=u.card,
                                   deploy=0.3))
        if s.death_projectile is not None:
            pr = Projectile(self, s.death_projectile, u.owner, u.x, u.y, tx=u.x, ty=u.y, source=None)
            self._impact(pr)
        if s.death_effect is not None:
            self.effects.append(Effect(self, s.death_effect, u.owner, u.x, u.y))
        if s.elixir_on_death:
            p = self.players[1 - u.owner]
            p.elixir = min(MAX_ELIXIR, p.elixir + s.elixir_on_death)
        for b, _, owner, _ in u.buffs.values():           # Mother Witch curse
            if b.death_spawn is not None and owner != u.owner and not u.building:
                self._add(Unit(self, b.death_spawn, owner, u.x, u.y, deploy=0.5))
        if not u.building and not u.timed:                 # souls for an enemy Skeleton King
            for o in self.units:
                if (o.alive and o.owner != u.owner and o.spec is not None and o.spec.ability is not None
                        and getattr(o.spec.ability, "summon", None) and math.dist((o.x, o.y), (u.x, u.y)) <= 5.0):
                    o.souls += 1

    # ----------------------------------------------------------- snapshot
    def snapshot(self) -> dict:
        return dict(
            t=round(self.time, 2),
            elixir=[round(p.elixir, 2) for p in self.players],
            hand=[list(p.hand) for p in self.players],
            next=[p.queue[0] for p in self.players],
            crowns=[p.crowns for p in self.players],
            towers=[[t.owner, t.tower, t.lane, round(max(t.hp, 0) / t.max_hp, 3), int(t.active)] for t in self.towers()],
            units=[[u.uid, u.owner, u.spec.name, round(u.x, 2), round(u.y, 2),
                    round(max(u.hp, 0) / max(u.max_hp, 1), 3), int(u.deploy_left > 0), int(u.flying), int(u.building)]
                   for u in self.units if not u.tower and not u.timed],
            projectiles=[[p.owner, p.spec.name, round(p.x, 2), round(p.y, 2)] for p in self.projectiles if p.travelled >= 0],
            effects=[[e.owner, e.spec.name, round(e.x, 2), round(e.y, 2), e.spec.radius] for e in self.effects if e.delay <= 0],
        )
