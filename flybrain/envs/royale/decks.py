"""Decks: read the fly's deck from a file, and build random sensible decks.

A deck file is 8 card names, one per line or comma-separated. Lines starting
with # are comments. Names are matched loosely ("pekka", "mini pekka",
"log", "e-wiz"/"electro wizard" ...).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .db import DB, load

ALIASES = {
    "pekka": "P.E.K.K.A", "mini pekka": "Mini P.E.K.K.A", "minipekka": "Mini P.E.K.K.A", "log": "The Log",
    "ewiz": "Electro Wizard", "e wiz": "Electro Wizard", "e-wiz": "Electro Wizard", "edrag": "Electro Dragon",
    "xbow": "X-Bow", "x bow": "X-Bow", "musk": "Musketeer", "3m": "Three Musketeers", "rg": "Royal Giant",
    "mk": "Mega Knight", "gy": "Graveyard", "barrel": "Goblin Barrel", "gob barrel": "Goblin Barrel",
    "skarmy": "Skeleton Army", "skelly army": "Skeleton Army", "bb": "Barbarian Barrel", "snowball": "Giant Snowball",
    "hog": "Hog Rider", "ram": "Battle Ram", "ice spirit": "Ice Spirit", "egolem": "Elixir Golem", "egiant": "Electro Giant",
    "aq": "Archer Queen", "gk": "Golden Knight", "sk": "Skeleton King", "mm": "Mighty Miner", "nw": "Night Witch",
    "mw": "Mother Witch", "lumber": "Lumberjack", "ld": "Inferno Dragon", "inferno": "Inferno Tower",
    "it": "Inferno Tower", "bowl": "Bowler", "exe": "Executioner", "princess": "Princess", "bats": "Bats",
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower().replace(".", "").replace("-", " ")).strip()


def resolve(name: str, db: DB | None = None) -> str:
    db = db or load()
    if name in db.cards:
        return name
    n = _norm(name)
    for k, v in ALIASES.items():
        if _norm(k) == n:
            return v
    by_norm = {_norm(c): c for c in db.cards}
    if n in by_norm:
        return by_norm[n]
    matches = [c for k, c in by_norm.items() if n and n in k]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"unknown card {name!r}" + (f"; did you mean one of {matches}?" if matches else ""))


def parse_deck(text: str, db: DB | None = None) -> list[str]:
    db = db or load()
    lines = [ln.split("#", 1)[0] for ln in text.splitlines()]
    names = [n.strip() for ln in lines for n in ln.split(",") if n.strip()]
    deck = [resolve(n, db) for n in names]
    check_deck(deck, db)
    return deck


def load_deck(path: str | Path, db: DB | None = None) -> list[str]:
    return parse_deck(Path(path).read_text(), db)


def check_deck(deck: list[str], db: DB | None = None) -> None:
    db = db or load()
    if len(deck) != 8:
        raise ValueError(f"a deck needs exactly 8 cards, got {len(deck)}: {deck}")
    if len(set(deck)) != 8:
        raise ValueError(f"duplicate cards in deck: {deck}")
    for n in deck:
        if n not in db.cards:
            raise ValueError(f"unknown card {n!r}")
    if sum(db.cards[n].champion for n in deck) > 1:
        raise ValueError("a deck can have at most one champion")


def hits_air(card, db: DB | None = None) -> bool:
    if card.type == "spell":
        if card.effect is not None:
            return card.effect.air
        if card.projectile is not None:
            return card.projectile.air or card.projectile.radius > 0 and not card.projectile.ground
        return False
    return any(spec.attacks_air or (spec.spawn_character is not None and spec.spawn_character.attacks_air)
               for spec, _ in card.summons)


def is_splash(card) -> bool:
    for spec, _ in card.summons:
        if spec.splash or (spec.projectile is not None and (spec.projectile.radius > 0 or spec.projectile.pierce_range > 1)):
            return True
    return False


def random_deck(rng: np.random.Generator, db: DB | None = None, max_tries: int = 500) -> list[str]:
    """A random deck that is playable: one win condition, a small and a big
    spell (or two spells), at least two cards that hit air, at least one
    splash card, at most one champion, average elixir 2.6-4.6."""
    db = db or load()
    pool = db.pool()
    by_role: dict[str, list[str]] = {}
    for n in pool:
        by_role.setdefault(db.cards[n].role, []).append(n)
    for _ in range(max_tries):
        deck = [rng.choice(by_role["win_condition"])]
        spells = by_role["small_spell"] + by_role["big_spell"]
        deck.append(rng.choice(by_role["small_spell"]))
        second = [n for n in (by_role["big_spell"] if rng.random() < 0.8 else spells) if n not in deck]
        deck.append(rng.choice(second))
        rest = [n for n in pool if n not in deck and db.cards[n].role not in ("win_condition",)]
        while len(deck) < 8:
            c = rng.choice(rest)
            if c in deck:
                continue
            if db.cards[c].champion and any(db.cards[d].champion for d in deck):
                continue
            if db.cards[c].type == "spell" and sum(db.cards[d].type == "spell" for d in deck) >= 3:
                continue
            deck.append(c)
        cards = [db.cards[n] for n in deck]
        air = sum(hits_air(c) and c.type != "spell" for c in cards)
        splash = sum(is_splash(c) for c in cards)
        avg = np.mean([c.elixir for c in cards])
        if air >= 2 and splash >= 1 and 2.6 <= avg <= 4.6:
            return [str(n) for n in deck]
    raise RuntimeError("could not build a valid random deck")


DEFAULT_FLY_DECK = ["Hog Rider", "Musketeer", "Cannon", "Ice Golem", "Skeletons", "Ice Spirit", "Fireball", "The Log"]
