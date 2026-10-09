"""Build the full Clash Royale card table used by flybrain/envs/royale.

    python scripts/build_card_db.py                 # downloads RoyaleAPI data, writes the table
    python scripts/build_card_db.py --raw path/     # use already-downloaded cards.json + cards_stats.json

Source: RoyaleAPI's cr-api-data (https://github.com/RoyaleAPI/cr-api-data),
a datamine of the game files. Output is flybrain/envs/royale/data/cards.json:
every card at one level (tournament standard, 11 by default) converted to
simulator units:

  distances in tiles (game units / 1000), times in seconds, speeds in
  tiles/second (game speed / 60; Medium = 60 = 1 tile/s), multipliers as
  factors (a -35% slow becomes 0.65, Rage's 135 becomes 1.35).

Each card lists what it summons; every character, projectile and area
effect a card can produce (spawned skeletons, death bombs, ...) is in the
"characters" / "projectiles" / "effects" sections.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "flybrain/envs/royale/data/cards.json"
BASE = "https://raw.githubusercontent.com/RoyaleAPI/cr-api-data/master/docs/json/"
START_LEVEL = {"Common": 1, "Rare": 3, "Epic": 6, "Legendary": 9, "Champion": 11}


def load_raw(raw_dir: Path | None):
    if raw_dir is None:
        get = lambda f: json.loads(urllib.request.urlopen(BASE + f, timeout=60).read())
        return get("cards.json"), get("cards_stats.json")
    return json.loads((raw_dir / "cards.json").read_text()), json.loads((raw_dir / "cards_stats.json").read_text())


class Builder:
    def __init__(self, cards, stats, level: int):
        self.cards, self.level = cards, level
        self.idx: dict[str, dict] = {}
        for sec in ("troop", "building", "spell", "projectile", "characters", "character_buff"):
            for r in stats[sec]:
                if r.get("name"):
                    self.idx.setdefault(r["name"], {})[sec] = r
        self.characters: dict[str, dict] = {}
        self.projectiles: dict[str, dict] = {}
        self.effects: dict[str, dict] = {}
        self.buffs: dict[str, dict] = {}

    # ------------------------------------------------------------ helpers
    def at_level(self, rec: dict, key: str, rarity: str | None = None):
        arr = rec.get(f"{key}_per_level")
        base = rec.get(key) or 0
        if not arr:
            return base
        start = START_LEVEL.get(rarity or rec.get("rarity") or "Common", 1)
        i = min(max(self.level - start, 0), len(arr) - 1)
        return arr[i]

    @staticmethod
    def mult(v) -> float:
        """Game percent field -> factor. -35 -> 0.65, 135 -> 1.35, 0 -> 1."""
        if not v:
            return 1.0
        return v / 100 if v > 0 else 1 + v / 100

    @staticmethod
    def tower_factor(pct) -> float:
        return 1 + (pct or 0) / 100

    def rec(self, name, *secs):
        for s in secs:
            if s in self.idx.get(name, {}):
                return self.idx[name][s]
        return None

    # -------------------------------------------------------------- buffs
    def buff(self, name: str | None, data: dict | None = None) -> str | None:
        if not name:
            return None
        if name in self.buffs:
            return name
        b = data or self.rec(name, "character_buff") or {}
        self.buffs[name] = dict(
            speed=self.mult(b.get("speed_multiplier")),
            hit_speed=self.mult(b.get("hit_speed_multiplier")),
            spawn_speed=self.mult(b.get("spawn_speed_multiplier")),
            dps=b.get("damage_per_second") or 0,
            heal_per_second=b.get("heal_per_second") or 0,
            hit_frequency=(b.get("hit_frequency") or 0) / 1000,
            tower_factor=self.tower_factor(b.get("crown_tower_damage_percent")),
            building_factor=1 + (b.get("building_damage_percent") or 0) / 100,
            pull=bool(b.get("attract_percentage")),
            invisible=bool(b.get("invisible")),
            clone=bool(b.get("clone")),
            switch_team=bool(b.get("switch_team")),
            death_spawn=self.character(b.get("death_spawn")) if b.get("death_spawn") else None,
            damage_multiplier=self.mult(b.get("damage_multiplier")),
        )
        return name

    # --------------------------------------------------------- projectiles
    def projectile(self, name: str | None) -> str | None:
        if not name or name in self.projectiles:
            return name
        p = self.rec(name, "projectile")
        if p is None:
            return None
        self.projectiles[name] = {}  # reserve (recursion)
        tb = p.get("target_buff")
        if tb:
            self.buff(tb, p.get("target_buff_data"))
        self.projectiles[name] = dict(
            speed=(p.get("speed") or 600) / 60,
            damage=self.at_level(p, "damage"),
            radius=(p.get("radius") or 0) / 1000,
            air=bool(p.get("aoe_to_air")), ground=bool(p.get("aoe_to_ground")),
            homing=bool(p.get("homing")),
            tower_factor=self.tower_factor(p.get("crown_tower_damage_percent")),
            pushback=(p.get("pushback") or 0) / 1000,
            buff=tb, buff_time=(p.get("buff_time") or 0) / 1000,
            chain_count=p.get("chained_hit_count") or 0, chain_radius=(p.get("chained_hit_radius") or 0) / 1000,
            pierce_range=(p.get("projectile_range") or 0) / 1000,
            pierce_radius=(p.get("projectile_radius") or 0) / 1000,
            returns=bool(p.get("pingpong_visual_time")),
            spawn_character=self.character(p.get("spawn_character")) if p.get("spawn_character") else None,
            spawn_count=p.get("spawn_character_count") or (1 if p.get("spawn_character") else 0),
            spawn_projectile=self.projectile(p.get("spawn_projectile")),
            spawn_projectile_count=p.get("spawn_count") or 0,
            drag=bool(p.get("drag_back_as_attractor")),
        )
        return name

    # -------------------------------------------------------- area effects
    def effect(self, name: str | None) -> str | None:
        if not name or name in self.effects:
            return name
        s = self.rec(name, "spell")
        if s is None:
            return None
        self.effects[name] = {}
        bname = s.get("buff")
        if bname:
            self.buff(bname, s.get("buff_data"))
        self.effects[name] = dict(
            radius=(s.get("radius") or 0) / 1000,
            duration=(s.get("life_duration") or 0) / 1000,
            hit_interval=(s.get("hit_speed") or 0) / 1000,
            damage=self.at_level(s, "damage"),
            tower_factor=self.tower_factor(s.get("crown_tower_damage_percent")),
            pushback=(s.get("pushback") or 0) / 1000,
            buff=bname, buff_time=(s.get("buff_time") or 0) / 1000,
            only_own=bool(s.get("only_own_troops")),
            ignore_buildings=bool(s.get("ignore_buildings")),
            air=bool(s.get("hits_air")), ground=bool(s.get("hits_ground", True)),
            clone=bool(s.get("clone")),
            hit_biggest=bool(s.get("hit_biggest_targets")),
            projectile=self.projectile(s.get("projectile")),
            spawn_character=self.character(s.get("spawn_character")) if s.get("spawn_character") else None,
            spawn_interval=(s.get("spawn_interval") or 0) / 1000,
            spawn_initial_delay=(s.get("spawn_initial_delay") or 0) / 1000,
            spawn_min_radius=(s.get("spawn_min_radius") or 0) / 1000,
        )
        return name

    # ---------------------------------------------------------- characters
    def character(self, name: str | None) -> str | None:
        if not name or name in self.characters:
            return name
        c = self.rec(name, "characters", "building")
        if c is None:
            return None
        self.characters[name] = {}
        is_building = "building" in self.idx[name] and "characters" not in self.idx[name]
        vd = None
        if c.get("variable_damage2"):
            d1 = self.at_level(c, "damage")
            scale = d1 / max(c.get("damage") or 1, 1)
            dmg3 = [d1, round(c["variable_damage2"] * scale), round((c.get("variable_damage3") or 0) * scale)]
            if c.get("variable_damage_time1"):   # ramps up while locked on one target (Inferno)
                vd = dict(mode="time", damage=dmg3,
                          stage_time=[c["variable_damage_time1"] / 1000, (c.get("variable_damage_time2") or 2000) / 1000])
            else:                                # combo by hit count (Monk: hit, hit, big hit)
                vd = dict(mode="combo", damage=dmg3, stage_time=[],
                          pushback=(c.get("melee_pushback3") or 0) / 1000)
        hp = self.at_level(c, "hitpoints")
        scale_hp = hp / max(c.get("hitpoints") or 1, 1)
        dmg = self.at_level(c, "damage")
        scale_dmg = dmg / max(c.get("damage") or 1, 1) if c.get("damage") else scale_hp
        bod = c.get("buff_on_damage")
        if bod:
            self.buff(bod)
        self.characters[name] = dict(
            building=is_building,
            hp=hp,
            shield=round((c.get("shield_hitpoints") or 0) * scale_hp),
            damage=dmg,
            hit_speed=(c.get("hit_speed") or 1000) / 1000,
            load_time=(c.get("load_time") or 0) / 1000,
            load_first_hit=bool(c.get("load_first_hit")),
            range=(c.get("range") or 0) / 1000,
            min_range=(c.get("minimum_range") or 0) / 1000,
            sight=(c.get("sight_range") or c.get("range") or 5500) / 1000,
            speed=(c.get("speed") or 0) / 60,
            deploy_time=(c.get("deploy_time") or 1000) / 1000,
            flying=bool(c.get("flying_height")),
            attacks_air=bool(c.get("attacks_air")), attacks_ground=bool(c.get("attacks_ground", True)),
            buildings_only=bool(c.get("target_only_buildings")),
            troops_only=bool(c.get("target_only_troops")),
            splash=(c.get("area_damage_radius") or 0) / 1000,
            self_splash=bool(c.get("self_as_aoe_center")),
            projectile=self.projectile(c.get("custom_first_projectile") or c.get("projectile")),
            projectiles=c.get("multiple_projectiles") or 1,
            targets=c.get("multiple_targets") or 1,
            radius=(c.get("collision_radius") or 500) / 1000,
            mass=c.get("mass") or (100 if is_building else 4),
            life_time=(c.get("life_time") or 0) / 1000,
            ignore_pushback=bool(c.get("ignore_pushback")),
            jump=bool(c.get("jump_enabled")),
            kamikaze=bool(c.get("kamikaze")),
            tower_factor=self.tower_factor(c.get("crown_tower_damage_percent")),
            spawn_character=self.character(c.get("spawn_character")) if c.get("spawn_character") else None,
            spawn_number=c.get("spawn_number") or 0,
            spawn_interval=(c.get("spawn_interval") or 0) / 1000,
            spawn_pause=(c.get("spawn_pause_time") or 0) / 1000,
            spawn_start=(c.get("spawn_start_time") or 0) / 1000,
            spawn_radius=(c.get("spawn_radius") or 0) / 1000,
            spawn_attach=bool(c.get("spawn_attach")),
            spawn_limit=c.get("spawn_limit") or 0,
            death_spawn=self.character(c.get("death_spawn_character")) if c.get("death_spawn_character") else None,
            death_spawn_count=c.get("death_spawn_count") or (1 if c.get("death_spawn_character") else 0),
            death_spawn2=self.character(c.get("death_spawn_character2")) if c.get("death_spawn_character2") else None,
            death_spawn_count2=c.get("death_spawn_count2") or 0,
            death_spawn_radius=(c.get("death_spawn_radius") or 0) / 1000,
            death_damage=round((c.get("death_damage") or 0) * scale_dmg),
            death_damage_radius=(c.get("death_damage_radius") or 0) / 1000,
            death_pushback=(c.get("death_push_back") or 0) / 1000,
            death_projectile=self.projectile(c.get("death_spawn_projectile")),
            elixir_on_death=(c.get("mana_on_death_for_opponent") or 0) / 1000,
            charge_damage=round((c.get("damage_special") or 0) * scale_dmg),
            charge_speed=(c.get("charge_speed_multiplier") or 0) / 100,
            dash_damage=round((c.get("dash_damage") or 0) * scale_dmg),
            dash_min=(c.get("dash_min_range") or 0) / 1000,
            dash_max=(c.get("dash_max_range") or 0) / 1000,
            dash_radius=(c.get("dash_radius") or 0) / 1000,
            dash_cooldown=(c.get("dash_cooldown") or 0) / 1000,
            dash_count=c.get("dash_count") or 0,
            dash_secondary_range=(c.get("dash_secondary_range") or 0) / 1000,
            variable_damage=vd,
            buff_on_damage=bod, buff_on_damage_time=(c.get("buff_on_damage_time") or 0) / 1000,
            spawn_effect=self.effect(c.get("spawn_area_object")),
            invisible_when_idle=bool(c.get("buff_when_not_attacking") == "Invisibility"),
            heal_when_idle=bool(c.get("buff_when_not_attacking") == "BattleHealerSelf"),
            special_range=(c.get("special_range") or 0) / 1000,
            special_min_range=(c.get("special_min_range") or 0) / 1000,
            special_projectile=self.projectile(c.get("projectile_special")),
            reflect_damage=round((c.get("reflected_attack_damage") or 0) * scale_dmg),
            reflect_radius=(c.get("reflected_attack_radius") or 0) / 1000,
            ability=c.get("ability"),
            hides_when_idle=bool(c.get("hides_when_not_attacking")),
            attack_pushback=(c.get("attack_push_back") or 0) / 1000,
        )
        return name

    # ---------------------------------------------------------------- cards
    def card(self, c: dict) -> dict:
        key = c["sc_key"]
        out = dict(name=c["name"], key=c["key"], elixir=c["elixir"], type=c["type"].lower(), rarity=c["rarity"],
                   event=False, summons=[], radius=0.0, effect=None, projectile=None, waves=1,
                   deploy_anywhere=False, special=None)
        t = self.rec(key, "troop")
        b = self.rec(key, "building")
        s = self.rec(key, "spell")
        if t:
            out["event"] = bool(t.get("not_visible"))
            out["summons"] = [[self.character(t["summon_character"]), t.get("summon_number") or 1]]
            if t.get("summon_character_second"):
                out["summons"].append([self.character(t["summon_character_second"]), t.get("summon_character_second_count") or 1])
            out["radius"] = (t.get("summon_radius") or 0) / 1000
            out["full_lane"] = bool(t.get("full_lane_deploy"))
            out["deploy_anywhere"] = bool(t.get("touchdown_limited_deploy"))
            if t.get("projectile"):
                out["projectile"] = self.projectile(t["projectile"])  # Mega Knight landing
        elif b:
            out["event"] = bool(b.get("not_visible"))
            ch = b.get("summon_character") or key
            out["summons"] = [[self.character(ch), 1]]
            out["deploy_anywhere"] = bool(b.get("touchdown_limited_deploy"))
            if b.get("spawn_area_object"):
                out["effect"] = self.effect(b["spawn_area_object"])
        elif s:
            out["effect"] = self.effect(key)
            if s.get("summon_character"):
                ch = self.character(s["summon_character"])
                if ch:
                    out["summons"] = [[ch, 1]]
        return out


# Spells and buildings whose stats live under other names in the data.
MANUAL = {
    "Fireball": dict(projectile="FireballSpell"),
    "Arrows": dict(projectile="ArrowsSpell", waves=3),
    "Rocket": dict(projectile="RocketSpell"),
    "Giant Snowball": dict(projectile="SnowballSpell"),
    "The Log": dict(projectile="LogProjectileRolling", special="rolling"),
    "Barbarian Barrel": dict(projectile="BarbLogProjectileRolling", special="rolling"),
    "Goblin Barrel": dict(projectile="GoblinBarrel", spawn_count=3),
    "Mirror": dict(special="mirror"),
    "Royal Delivery": dict(summons=[["RoyalDelivery", 1]]),
    "Lumberjack": dict(death_effect="BarbarianRage"),
    "Ice Golem": dict(death_effect="FreezeIceGolemite"),
    "Goblin Drill": dict(summons=[["GoblinDrill", 1]]),
    "Graveyard": dict(deploy_anywhere=True),
    "Heal Spirit": dict(death_effect="HealSpirit", no_card_effect=True),
}

# Champion abilities (not described in the stats file). Values from the game wiki.
ABILITIES = {
    "ArcherQueenRapid": dict(cost=1, cooldown=17, duration=3.5, invisible=True, hit_speed=2.8),
    "GoldenKnightChain": dict(cost=1, cooldown=13, chain=10),
    "SkeletonKing": dict(cost=2, cooldown=20, summon="SkeletonKingSkeleton", min_count=6, max_count=16),
    "MightyMinerLaneSwitch": dict(cost=1, cooldown=13, bomb_damage=210, bomb_radius=2.5),
    "Deflect": dict(cost=1, cooldown=17, duration=4.0, damage_taken=0.35),
}
EXTRA_CHARACTERS = ["SkeletonKingSkeleton", "VoodooHog"]

ELIXIR_COLLECTOR = dict(  # not in the stats file; level 11 values from the game wiki
    building=True, hp=1070, shield=0, damage=0, hit_speed=1, load_time=0, load_first_hit=False, range=0, min_range=0,
    sight=0, speed=0, deploy_time=1, flying=False, attacks_air=False, attacks_ground=False, buildings_only=False,
    troops_only=False, splash=0, self_splash=False, projectile=None, projectiles=1, targets=1, radius=1.0, mass=100,
    life_time=86, ignore_pushback=True, jump=False, kamikaze=False, tower_factor=1, spawn_character=None, spawn_number=0,
    spawn_interval=0, spawn_pause=0, spawn_start=0, spawn_radius=0, spawn_attach=False, spawn_limit=0, death_spawn=None,
    death_spawn_count=0, death_spawn2=None, death_spawn_count2=0, death_spawn_radius=0, death_damage=0,
    death_damage_radius=0, death_pushback=0, death_projectile=None, elixir_on_death=0, charge_damage=0, charge_speed=0,
    dash_damage=0, dash_min=0, dash_max=0, dash_radius=0, dash_cooldown=0, dash_count=0, dash_secondary_range=0,
    variable_damage=None, buff_on_damage=None, buff_on_damage_time=0, spawn_effect=None, invisible_when_idle=False,
    heal_when_idle=False, special_range=0, special_min_range=0, special_projectile=None, reflect_damage=0,
    reflect_radius=0, ability=None, hides_when_idle=False, attack_pushback=0,
    elixir_every=9.0, elixir_on_destroy=1,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, help="directory with cards.json and cards_stats.json")
    ap.add_argument("--level", type=int, default=11, help="card level (11 = tournament standard)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    cards, stats = load_raw(args.raw)
    bld = Builder(cards, stats, args.level)
    out_cards = []
    for c in cards:
        if c["name"] == "Party Rocket":
            continue  # event card with no stats in the data
        card = bld.card(c)
        m = MANUAL.get(c["name"], {})
        if "projectile" in m:
            card["projectile"] = bld.projectile(m["projectile"])
        for k in ("waves", "special", "deploy_anywhere"):
            if k in m:
                card[k] = m[k]
        if "spawn_count" in m and card["projectile"]:
            bld.projectiles[card["projectile"]]["spawn_count"] = m["spawn_count"]
        if "summons" in m:
            card["summons"] = [[bld.character(n), k] for n, k in m["summons"]]
        if "death_effect" in m:
            eff = bld.effect(m["death_effect"])
            for name, _ in card["summons"]:
                bld.characters[name]["death_effect"] = eff
        if m.get("no_card_effect"):
            card["effect"] = None
        if c["name"] == "Elixir Collector":
            bld.characters["ElixirCollector"] = ELIXIR_COLLECTOR
            card["summons"] = [["ElixirCollector", 1]]
        out_cards.append(card)
    for n in EXTRA_CHARACTERS:
        bld.character(n)
    bld.buff("Rage"); bld.buff("ZapFreeze"); bld.buff("IceWizardSlowDown")
    for ch in bld.characters.values():
        ch.setdefault("death_effect", None)
    data = dict(level=args.level, source="RoyaleAPI cr-api-data (github.com/RoyaleAPI/cr-api-data)",
                cards=out_cards, characters=bld.characters, projectiles=bld.projectiles,
                effects=bld.effects, buffs=bld.buffs, abilities=ABILITIES)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=1, sort_keys=True))
    n_event = sum(c["event"] for c in out_cards)
    print(f"wrote {args.out}: {len(out_cards)} cards ({n_event} event-only), {len(bld.characters)} characters, "
          f"{len(bld.projectiles)} projectiles, {len(bld.effects)} area effects, {len(bld.buffs)} buffs")


if __name__ == "__main__":
    main()
