"""Edit the fly's decks in your browser.

    python -m flybrain.deck_editor          (or double-click edit_deck.bat on Windows)

Opens http://127.0.0.1:8765 with every card (search, role filter, elixir),
the deck files in decks/, and a Save button. Saving checks the deck (8
different cards, at most one champion) and writes the file. decks/fly.txt is
the deck the bot and training use by default; the bot picks up changes
between battles.

Only listens on this computer (127.0.0.1).
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .envs.royale.db import load
from .envs.royale.decks import check_deck, parse_deck

ROOT = Path(__file__).resolve().parents[1]
DECKS = ROOT / "decks"
PAGE = ROOT / "web/deck_editor.html"
HEADER = ("# {title}\n# 8 cards, one per line. Edit here or with: python -m flybrain.deck_editor\n")


def deck_files(decks_dir: Path) -> list[str]:
    decks_dir.mkdir(parents=True, exist_ok=True)
    names = sorted(p.stem for p in decks_dir.glob("*.txt"))
    if "fly" in names:
        names.remove("fly")
        names.insert(0, "fly")
    return names


def read_deck(decks_dir: Path, name: str) -> dict:
    path = decks_dir / f"{name}.txt"
    try:
        return dict(name=name, cards=parse_deck(path.read_text()), error="")
    except FileNotFoundError:
        return dict(name=name, cards=[], error="")
    except ValueError as e:                       # a hand-edited file with a problem
        text = path.read_text()
        cards = [ln.split("#", 1)[0].strip() for ln in text.splitlines()]
        db = load()
        return dict(name=name, cards=[c for c in cards if c in db.cards], error=str(e))


def write_deck(decks_dir: Path, name: str, cards: list[str]) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", name):
        raise ValueError("deck names can use letters, digits, - and _ (up to 40)")
    check_deck(cards)
    title = "The fly brain's deck (used by the bot and training)" if name == "fly" else f"Deck: {name}"
    (decks_dir / f"{name}.txt").write_text(HEADER.format(title=title) + "\n".join(cards) + "\n")


def make_handler(decks_dir: Path):
    db = load()
    cards = [dict(name=n, elixir=c.elixir, role=c.role, type=c.type, rarity=c.rarity, champion=c.champion)
             for n, c in sorted(db.cards.items()) if not c.event and c.role]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, PAGE.read_text(), "text/html")
            if self.path == "/api/state":
                return self._send(200, json.dumps(dict(cards=cards, decks=deck_files(decks_dir))))
            m = re.fullmatch(r"/api/deck/([A-Za-z0-9_-]{1,40})", self.path)
            if m:
                return self._send(200, json.dumps(read_deck(decks_dir, m.group(1))))
            self._send(404, json.dumps(dict(error="not found")))

        def do_POST(self):
            if self.path != "/api/save":
                return self._send(404, json.dumps(dict(error="not found")))
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                write_deck(decks_dir, str(body.get("name", "")), list(body.get("cards", [])))
                self._send(200, json.dumps(dict(ok=True, decks=deck_files(decks_dir))))
            except (ValueError, TypeError) as e:
                self._send(400, json.dumps(dict(ok=False, error=str(e))))

    return Handler


def serve(port: int = 8765, decks_dir: Path = DECKS, open_browser: bool = True) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(Path(decks_dir)))
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(f"http://127.0.0.1:{server.server_port}")).start()
    return server


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    server = serve(args.port, open_browser=not args.no_browser)
    print(f"Deck editor at http://127.0.0.1:{server.server_port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()
