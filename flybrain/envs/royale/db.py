"""Card database for the full simulator (built by scripts/build_card_db.py).

Every card gets one *role*: the kind of play it is. The fly brain chooses a
role (and a lane); a helper then picks the card in hand with that role. This
way what the brain learns carries over between decks.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

DATA = Path(__file__).resolve().parent / "data/cards.json"

ROLES = (
    "win_condition",   # goes for towers: Hog Rider, Giant, Balloon, Miner, X-Bow, Graveyard ...
    "tank_killer",     # high single-target damage: P.E.K.K.A, Mini P.E.K.K.A, Inferno Dragon, Prince ...
    "frontline",       # sturdy melee that holds a lane: Knight, Ice Golem, Giant Skeleton, Bandit ...
    "splash_ground",   # area damage on the ground: Valkyrie, Bowler, Mega Knight, Dark Prince ...
    "splash_air",      # area damage that also hits air: Wizard, Baby Dragon, Executioner, Witch ...
    "ranged",          # single-target ranged, hits air: Musketeer, Archers, Dart Goblin ...
    "swarm",           # many small units: Goblin Gang, Skeleton Army, Barbarians, Guards ...
    "air",             # flying attackers: Minions, Mega Minion, Bats, Phoenix ...
    "cycle",           # 1-2 elixir spirits and skeletons
    "building",        # defensive buildings: Cannon, Tesla, Inferno Tower, Bomb Tower ...
    "spawner",         # Goblin Hut, Furnace, Barbarian Hut, Elixir Collector
    "small_spell",     # Zap, The Log, Arrows, Snowball, Barbarian Barrel, Royal Delivery, Tornado
    "big_spell",       # Fireball, Poison, Lightning, Rocket, Earthquake
    "utility",         # Freeze, Rage, Clone, Mirror
    "champion",        # Archer Queen, Golden Knight, Skeleton King, Mighty Miner, Monk
)

ROLE_OF = {
    # win conditions
    **{n: "win_condition" for n in (
        "Giant", "Hog Rider", "Balloon", "Golem", "Royal Giant", "Lava Hound", "Miner", "Battle Ram", "Ram Rider",
        "Goblin Barrel", "Graveyard", "Wall Breakers", "Royal Hogs", "Goblin Giant", "Electro Giant", "Elixir Golem",
        "X-Bow", "Mortar", "Goblin Drill", "Skeleton Barrel", "Three Musketeers")},
    **{n: "tank_killer" for n in (
        "P.E.K.K.A", "Mini P.E.K.K.A", "Inferno Dragon", "Prince", "Lumberjack", "Hunter", "Elite Barbarians",
        "Sparky", "Night Witch", "Raging Prince")},
    **{n: "frontline" for n in (
        "Knight", "Ice Golem", "Giant Skeleton", "Bandit", "Fisherman", "Battle Healer", "Rascals", "Cannon Cart")},
    **{n: "splash_ground" for n in ("Valkyrie", "Bowler", "Bomber", "Dark Prince", "Mega Knight", "Royal Ghost")},
    **{n: "splash_air" for n in (
        "Wizard", "Baby Dragon", "Executioner", "Witch", "Electro Dragon", "Firecracker", "Ice Wizard", "Princess",
        "Magic Archer", "Skeleton Dragons", "Electro Wizard", "Zappies", "Mother Witch")},
    **{n: "ranged" for n in ("Musketeer", "Archers", "Dart Goblin", "Spear Goblins", "Flying Machine")},
    **{n: "swarm" for n in (
        "Goblins", "Goblin Gang", "Skeleton Army", "Barbarians", "Guards", "Royal Recruits", "Minion Horde")},
    **{n: "air" for n in ("Minions", "Mega Minion", "Bats", "Phoenix")},
    **{n: "cycle" for n in ("Skeletons", "Ice Spirit", "Fire Spirit", "Electro Spirit", "Heal Spirit")},
    **{n: "building" for n in ("Cannon", "Tesla", "Inferno Tower", "Bomb Tower", "Goblin Cage", "Tombstone")},
    **{n: "spawner" for n in ("Goblin Hut", "Furnace", "Barbarian Hut", "Elixir Collector", "Party Hut")},
    **{n: "small_spell" for n in (
        "Zap", "The Log", "Arrows", "Giant Snowball", "Barbarian Barrel", "Royal Delivery", "Tornado")},
    **{n: "big_spell" for n in ("Fireball", "Poison", "Lightning", "Rocket", "Earthquake")},
    **{n: "utility" for n in ("Freeze", "Rage", "Clone", "Mirror")},
    **{n: "champion" for n in ("Archer Queen", "Golden Knight", "Skeleton King", "Mighty Miner", "Monk")},
}


def _ns(d: dict) -> SimpleNamespace:
    return SimpleNamespace(**d)


class DB:
    def __init__(self, path: Path = DATA):
        raw = json.loads(Path(path).read_text())
        self.level = raw["level"]
        self.buffs = {k: _ns(dict(v, name=k)) for k, v in raw["buffs"].items()}
        self.projectiles = {k: _ns(dict(v, name=k)) for k, v in raw["projectiles"].items()}
        self.effects = {k: _ns(dict(v, name=k)) for k, v in raw["effects"].items()}
        self.characters = {k: _ns(dict(v, name=k)) for k, v in raw["characters"].items()}
        self.abilities = {k: _ns(dict(v, name=k)) for k, v in raw["abilities"].items()}
        # resolve references to objects
        for p in self.projectiles.values():
            p.buff = self.buffs.get(p.buff)
            p.spawn_character = self.characters.get(p.spawn_character)
            p.spawn_projectile = self.projectiles.get(p.spawn_projectile)
        for e in self.effects.values():
            e.buff = self.buffs.get(e.buff)
            e.projectile = self.projectiles.get(e.projectile)
            e.spawn_character = self.characters.get(e.spawn_character)
        for b in self.buffs.values():
            b.death_spawn = self.characters.get(b.death_spawn)
        for c in self.characters.values():
            for k in ("spawn_character", "death_spawn", "death_spawn2"):
                setattr(c, k, self.characters.get(getattr(c, k)))
            for k in ("projectile", "death_projectile", "special_projectile"):
                setattr(c, k, self.projectiles.get(getattr(c, k)))
            c.spawn_effect = self.effects.get(c.spawn_effect)
            c.death_effect = self.effects.get(c.death_effect)
            c.buff_on_damage = self.buffs.get(c.buff_on_damage)
            c.ability = self.abilities.get(c.ability)
            c.vd = _ns(c.variable_damage) if c.variable_damage else None
        self.cards: dict[str, SimpleNamespace] = {}
        for d in raw["cards"]:
            c = _ns(dict(d))
            c.summons = [(self.characters[n], k) for n, k in c.summons]
            c.effect = self.effects.get(c.effect)
            c.projectile = self.projectiles.get(c.projectile)
            c.role = ROLE_OF.get(c.name)
            c.champion = c.rarity == "Champion"
            c.full_lane = getattr(c, "full_lane", False)
            self.cards[c.name] = c

    def pool(self, include_event: bool = False) -> list[str]:
        """Card names available for decks (event-only cards left out by default)."""
        return [n for n, c in self.cards.items() if include_event or (not c.event and c.role)]


@lru_cache(maxsize=1)
def load() -> DB:
    return DB()
