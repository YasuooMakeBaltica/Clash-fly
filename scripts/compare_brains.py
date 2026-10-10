"""Compare trained brains on the same simulated matches (same seeds, decks and opponents).

    python scripts/compare_brains.py models/fly_royale.pt runs/new.pt --matches 24
    python scripts/compare_brains.py runs/new.pt --no-pure          # only with the coach guard
    python scripts/compare_brains.py models/fly_royale.pt --no-pure --lookahead --opponents basic,coach,lookahead

Each brain plays the scripted bots with the fly deck (decks/fly.txt) and with
random decks, pure and with the coach guard (guard.py) that the real bot uses.
The last column is the average win rate over all rows.
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore", message="Sparse")

import torch  # noqa: E402

from flybrain.envs.royale.decks import load_deck  # noqa: E402
from flybrain.envs.royale.lookahead import Lookahead  # noqa: E402
from flybrain.real.bot import load_brain  # noqa: E402
from train_royale import evaluate  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("brains", nargs="+")
    ap.add_argument("--matches", type=int, default=24)
    ap.add_argument("--opponents", default="basic,coach,random")
    ap.add_argument("--no-pure", action="store_true", help="skip the runs without the guard")
    ap.add_argument("--fly-deck", default=str(ROOT / "decks/fly.txt"))
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--json", help="also write the results here")
    ap.add_argument("--lookahead", action="store_true", help="the guard's coach looks ahead (lookahead.py), as in the real bot")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    fly = load_deck(args.fly_deck)
    results = {}
    for path in args.brains:
        agent, _ = load_brain(path)
        for guard in ((True,) if args.no_pure else (False, True)):
            rows = {}
            for opp in args.opponents.split(","):
                for share, deck in ((1.0, "fly deck"), (0.0, "random decks")):
                    t = time.time()
                    r = evaluate(agent, opp, args.matches, fly, share, seed=args.seed, guard=guard,
                                 coach=Lookahead() if args.lookahead and guard else None)
                    rows[f"{opp} / {deck}"] = r["win"]
                    print(f"{Path(path).name:24s} {'guard' if guard else 'pure ':5s} vs {opp:6s} {deck:12s} "
                          f"win {r['win']:.2f} draw {r['draw']:.2f}  ({time.time() - t:.0f}s)", flush=True)
            rows["mean"] = sum(rows.values()) / len(rows)
            results[f"{path} {'guard' if guard else 'pure'}"] = rows
            print(f"{Path(path).name:24s} {'guard' if guard else 'pure ':5s} mean win {rows['mean']:.3f}", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
