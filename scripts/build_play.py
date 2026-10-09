"""Bundle the browser game: web/play_template.html + web/clash_core.js + an exported brain.

    python scripts/export_brain_js.py --brain runs/clash/brain.pt --out web/brain.json
    python scripts/build_play.py --brain-json web/brain.json --out runs/clash/beat-the-fly.html
"""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--brain-json", required=True)
    ap.add_argument("--out", default="runs/clash/beat-the-fly.html")
    args = ap.parse_args()
    html = (ROOT / "web/play_template.html").read_text()
    core = (ROOT / "web/clash_core.js").read_text().replace("</script", "<\\/script")
    html = html.replace("/*CORE*/", core).replace("/*BRAIN*/null", Path(args.brain_json).read_text())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(html)
    print(f"wrote {args.out} ({len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
