import json
import threading
import urllib.request

import pytest

from flybrain.deck_editor import serve
from flybrain.envs.royale.decks import load_deck


@pytest.fixture
def editor(tmp_path):
    (tmp_path / "fly.txt").write_text("Hog Rider\nMusketeer\nCannon\nIce Golem\nSkeletons\nIce Spirit\nFireball\nThe Log\n")
    server = serve(port=0, decks_dir=tmp_path, open_browser=False)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", tmp_path
    server.shutdown()


def get(url):
    return json.loads(urllib.request.urlopen(url).read())


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        return json.loads(urllib.request.urlopen(req).read())
    except urllib.error.HTTPError as e:
        return json.loads(e.read())


def test_page_and_state(editor):
    url, _ = editor
    assert b"Fly Deck Editor" in urllib.request.urlopen(url + "/").read()
    st = get(url + "/api/state")
    assert len(st["cards"]) >= 105 and st["decks"][0] == "fly"
    assert get(url + "/api/deck/fly")["cards"][0] == "Hog Rider"


def test_save_valid_and_reject_invalid(editor):
    url, d = editor
    deck = ["P.E.K.K.A", "Mini P.E.K.K.A", "The Log", "Electro Wizard", "X-Bow", "Graveyard", "Bats", "Zap"]
    assert post(url + "/api/save", dict(name="fly", cards=deck))["ok"]
    assert load_deck(d / "fly.txt") == deck
    r = post(url + "/api/save", dict(name="fly", cards=deck[:7]))
    assert not r["ok"] and "8 cards" in r["error"]
    r = post(url + "/api/save", dict(name="fly", cards=["Archer Queen", "Golden Knight"] + deck[:6]))
    assert not r["ok"] and "champion" in r["error"]
    r = post(url + "/api/save", dict(name="../evil", cards=deck))
    assert not r["ok"]
    assert post(url + "/api/save", dict(name="ladder2", cards=deck))["ok"]
    assert (d / "ladder2.txt").exists()
