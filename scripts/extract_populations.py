"""Milestone 1: extract PN/KC/MBON/DAN/APL populations from FlyWire Codex CSVs.

    python scripts/extract_populations.py --data data/flywire --out data/mushroom_body.npz
    python scripts/extract_populations.py --data synthetic           # no download needed
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flybrain.connectome import load  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="synthetic", help="Codex CSV directory, a saved .npz, or 'synthetic'")
    ap.add_argument("--out", help="save the extracted subcircuit here (.npz) for fast loading")
    ap.add_argument("--min-syn", type=int, default=5, help="drop connections with fewer synapses")
    args = ap.parse_args()

    kwargs = {} if args.data == "synthetic" or args.data.endswith(".npz") else {"min_syn": args.min_syn}
    conn = load(args.data, **kwargs)
    print(conn.summary())
    if args.out:
        conn.save(args.out)
        print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
