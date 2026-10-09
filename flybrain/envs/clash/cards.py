"""Card and tower stats for the simulator.

Approximate values for tournament-standard levels (level 11), rounded. They
are close enough for strategy to matter (Giant tanks, Mini P.E.K.K.A shreds
tanks, Arrows clears swarms) but are not exact game data. Distances are in
tiles, times in seconds.

Speeds: slow 0.75, medium 1.0, fast 1.5, very fast 2.0 tiles/s.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Card:
    name: str
    cost: int
    kind: str                 # "troop" or "spell"
    role: str                 # tank, tank_killer, mini_tank, ranged, swarm, spell
    count: int = 1            # units spawned
    hp: float = 0.0
    damage: float = 0.0       # per hit (troops) or area damage (spells)
    hit_speed: float = 1.0
    range: float = 1.0        # attack range (melee ~1)
    speed: float = 1.0
    sight: float = 5.5        # how far a troop looks for targets
    targets: str = "ground"   # ground, air_ground, buildings
    radius: float = 0.0       # spell radius
    tower_damage: float = 0.0  # spell damage to crown towers
    delay: float = 1.0        # spell travel time


DECK: tuple[Card, ...] = (
    Card("Knight", 3, "troop", "mini_tank", hp=1766, damage=202, hit_speed=1.2, range=1.2, speed=1.0),
    Card("Archers", 3, "troop", "ranged", count=2, hp=304, damage=107, hit_speed=0.9, range=5.0,
         speed=1.0, targets="air_ground"),
    Card("Giant", 5, "troop", "tank", hp=4091, damage=254, hit_speed=1.5, range=1.2, speed=0.75,
         sight=7.5, targets="buildings"),
    Card("Musketeer", 4, "troop", "ranged", hp=720, damage=218, hit_speed=1.0, range=6.0,
         speed=1.0, targets="air_ground"),
    Card("Mini P.E.K.K.A", 4, "troop", "tank_killer", hp=1361, damage=720, hit_speed=1.6, range=0.8,
         speed=1.5),
    Card("Goblins", 2, "troop", "swarm", count=4, hp=202, damage=120, hit_speed=1.1, range=0.5,
         speed=2.0),
    Card("Fireball", 4, "spell", "spell", damage=689, radius=2.5, tower_damage=207, delay=1.0),
    Card("Arrows", 3, "spell", "spell", damage=366, radius=4.0, tower_damage=93, delay=0.8),
)
CARD_INDEX = {c.name: i for i, c in enumerate(DECK)}

# Crown towers
PRINCESS_HP = 3052.0
PRINCESS_DAMAGE = 109.0
PRINCESS_HIT_SPEED = 0.8
PRINCESS_RANGE = 7.5
KING_HP = 4824.0
KING_DAMAGE = 109.0
KING_HIT_SPEED = 1.0
KING_RANGE = 7.0
TOWER_SIZE = 1.5          # half-width; added to attack ranges against towers

# Arena
WIDTH, HEIGHT = 18.0, 32.0
RIVER_LO, RIVER_HI = 15.0, 17.0
LANE_X = (3.5, 14.5)      # bridge / lane centres (left, right)
DEPLOY_TIME = 1.0
MATCH_TIME = 180.0
DOUBLE_ELIXIR_AT = 120.0
ELIXIR_RATE = 1 / 2.8     # per second
START_ELIXIR = 5.0
MAX_ELIXIR = 10.0
