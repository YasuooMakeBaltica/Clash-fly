"""Test a coach change head-to-head, in duplicate format.

Every random deck pairing is played twice on the same seed with the two
coaches swapped between seats, so deck and seat luck cancel exactly. Two
identical coaches score exactly 0, and any difference comes from strategy.
Reports A's margin (wins minus losses per game) with its standard error.

    python scripts/duel_coaches.py --a "hold_line=7" --pairs 300
    python scripts/duel_coaches.py --a "place:building_y=6" --b "place:building_y=9"

Settings: coach parameters (strategy.COACH_DEFAULTS) as name=value, and
placement parameters (strategy.PLACE_DEFAULTS) as place:name=value,
comma-separated.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from flybrain.envs.royale import strategy as S  # noqa: E402
from flybrain.envs.royale.decks import random_deck  # noqa: E402


def coach_factory(spec: str):
    coach, place = {}, {}
    for item in filter(None, (s.strip() for s in spec.split(","))):
        key, val = item.split("=")
        if key.startswith("place:"):
            place[key[6:]] = float(val)
        else:
            coach[key] = float(val)
    placer = (lambda view, name, lane: S.place(view, name, lane, **place)) if place else None
    return lambda: S.Coach(placer=placer, **coach)


def duel(make_a, make_b, pairs: int, seed: int = 0):
    """(margin of A per game, standard error, share of pairings where the result differed)."""
    rng = np.random.default_rng(seed)
    score = []
    for i in range(pairs):
        decks = (random_deck(rng), random_deck(rng))
        s = 0
        for a_seat in (0, 1):
            bots = [None, None]
            bots[a_seat], bots[1 - a_seat] = make_a(), make_b()
            sim = S.play_match(bots[0], bots[1], decks, seed=seed * 100000 + i)
            s += 1 if sim.winner == a_seat else (-1 if sim.winner == 1 - a_seat else 0)
        score.append(s)
    sc = np.array(score, float)
    return sc.mean() / 2, sc.std(ddof=1) / 2 / np.sqrt(len(sc)), float((sc != 0).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", default="", help="coach A's settings (default: the current coach)")
    ap.add_argument("--b", default="", help="coach B's settings (default: the current coach)")
    ap.add_argument("--pairs", type=int, default=300, help="deck pairings (each played twice)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t = time.time()
    m, se, diff = duel(coach_factory(args.a), coach_factory(args.b), args.pairs, args.seed)
    print(f"A [{args.a or 'current'}] vs B [{args.b or 'current'}]: margin {m:+.3f} ± {se:.3f} per game "
          f"({2 * args.pairs} games, results differed in {diff:.0%} of pairings, {time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
