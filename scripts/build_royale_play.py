"""Bundle the full-card-pool browser game into one HTML file.

    python scripts/export_brain_js.py --brain models/fly_royale.pt --out web/brain_royale.json
    python scripts/build_royale_play.py --brain-json web/brain_royale.json --out runs/beat-the-fly-full.html
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flybrain.envs.royale.decks import load_deck  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--brain-json", required=True)
    ap.add_argument("--fly-deck", default=str(ROOT / "decks/fly.txt"))
    ap.add_argument("--out", default=str(ROOT / "runs/beat-the-fly-full.html"))
    args = ap.parse_args()
    esc = lambda s: s.replace("</script", "<\\/script")
    html = (ROOT / "web/royale_play_template.html").read_text()
    html = html.replace("/*CLASH_CORE*/", esc((ROOT / "web/clash_core.js").read_text()))
    html = html.replace("/*ROYALE_CORE*/", esc((ROOT / "web/royale_core.js").read_text()))
    html = html.replace("/*CARDS*/null", esc((ROOT / "flybrain/envs/royale/data/cards.json").read_text()))
    html = html.replace("/*BRAIN*/null", esc(Path(args.brain_json).read_text()))
    html = html.replace("/*FLY_DECK*/null", json.dumps(load_deck(args.fly_deck)))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(html)
    print(f"wrote {args.out} ({len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
