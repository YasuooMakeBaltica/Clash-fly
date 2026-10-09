"""Recognise any card in hand from the official card art, and learn the deck.

Card images come from RoyaleAPI's asset repository (the game's own card
art). They are downloaded to templates/official/ the first time they are
needed; nothing is stored in this repository.

Matching compares the inner part of each hand slot (skipping the frame and
the elixir badge) with the inner part of every card image, in colour, after
normalising brightness. A few small shifts and zooms of the slot are tried,
so slightly-off calibration still works.

``DeckTracker`` watches the hand and the next-card slot during battles and
works out the 8-card deck as the cards cycle; the bot saves it to
decks/fly.txt once all 8 are seen.
"""

from __future__ import annotations

import concurrent.futures as cf
import urllib.request
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from ..envs.royale.db import load

BASE = "https://raw.githubusercontent.com/RoyaleAPI/cr-api-assets/master/cards-150/"
ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_DIR = ROOT / "templates/official"
SIZE = (30, 36)                       # (w, h) of the compared patch
INNER = (0.14, 0.10, 0.86, 0.70)       # x0, y0, x1, y1 of the card art used for matching
# The part of an official card image that a hand slot shows: the game zooms
# the art in and drops the frame (fitted on a real 540x960 battle screenshot).
CARD_ART = (0.075, 0.06, 0.925, 0.95)


def slot_view(img: np.ndarray, art=CARD_ART) -> np.ndarray:
    """What a hand slot shows of a full card image."""
    h, w = img.shape[:2]
    return img[int(art[1] * h):int(art[3] * h), int(art[0] * w):int(art[2] * w)]


def download_official(dest: str | Path = OFFICIAL_DIR, workers: int = 8) -> list[str]:
    """Download the card images for every card in the pool. Returns names that failed."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    db = load()

    def get(name):
        out = dest / f"{name}.png"
        if out.exists():
            return None
        try:
            data = urllib.request.urlopen(BASE + db.cards[name].key + ".png", timeout=30).read()
            out.write_bytes(data)
            return None
        except Exception:
            return name

    with cf.ThreadPoolExecutor(workers) as ex:
        return [r for r in ex.map(get, db.pool()) if r]


def _flatten(img: np.ndarray) -> np.ndarray:
    """BGRA -> BGR on a grey background (card images have transparent corners)."""
    if img.ndim == 3 and img.shape[2] == 4:
        a = img[:, :, 3:4].astype(np.float32) / 255.0
        return (img[:, :, :3] * a + 128 * (1 - a)).astype(np.uint8)
    return img


def _inner(img: np.ndarray, zoom: float = 0.0, dx: float = 0.0, dy: float = 0.0) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = INNER
    cx, cy = (x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy
    hw, hh = (x1 - x0) / 2 * (1 - zoom), (y1 - y0) / 2 * (1 - zoom)
    X0, X1 = int(max(0, (cx - hw) * w)), int(min(w, (cx + hw) * w))
    Y0, Y1 = int(max(0, (cy - hh) * h)), int(min(h, (cy + hh) * h))
    return img[Y0:Y1, X0:X1]


def _vec(patch: np.ndarray, grey: bool = False) -> np.ndarray:
    """Brightness-normalised colour (LAB) or grey (L only) vector of a patch."""
    p = cv2.resize(patch, SIZE, interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(p, cv2.COLOR_BGR2LAB).astype(np.float32)
    if grey:
        lab = lab[:, :, :1]
    lab -= lab.reshape(-1, lab.shape[2]).mean(0)
    lab /= lab.reshape(-1, lab.shape[2]).std(0) + 1e-3
    v = lab.ravel()
    return v / (np.linalg.norm(v) + 1e-6)


def is_greyed(crop: np.ndarray, max_saturation: float = 25.0) -> bool:
    """Cards you can't afford yet are shown without colour."""
    return float(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 1].mean()) < max_saturation


class OfficialMatcher:
    """Matches a hand-slot crop against card art. ``images`` maps card name -> BGR(A) image."""

    JITTER = [(0.0, 0.0, 0.0), (0.08, 0.0, 0.0), (0.0, 0.03, 0.0), (0.0, -0.03, 0.0), (0.0, 0.0, 0.03),
              (0.0, 0.0, -0.03), (-0.06, 0.0, 0.0), (0.08, 0.0, 0.03)]

    def __init__(self, images: dict[str, np.ndarray], threshold: float = 0.55, margin: float = 0.03):
        self.names = sorted(images)
        self.threshold, self.margin = threshold, margin
        inner = [_inner(slot_view(_flatten(images[n]))) for n in self.names]
        self.bank = np.stack([_vec(p) for p in inner]) if self.names else None
        self.grey_bank = np.stack([_vec(p, grey=True) for p in inner]) if self.names else None

    @classmethod
    def from_dir(cls, directory: str | Path = OFFICIAL_DIR, **kw) -> "OfficialMatcher":
        d = Path(directory)
        cards = load().cards
        imgs = {}
        if d.exists():
            for p in sorted(d.glob("*.png")):
                if p.stem in cards:
                    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                    if im is not None:
                        imgs[p.stem] = im
        return cls(imgs, **kw)

    def scores(self, crop: np.ndarray, allowed: list[str] | None = None) -> dict[str, float]:
        if self.bank is None or crop is None or crop.size == 0:
            return {}
        crop = _flatten(crop)
        grey = is_greyed(_inner(crop))
        bank = self.grey_bank if grey else self.bank
        best = np.full(len(self.names), -1.0)
        for zoom, dx, dy in self.JITTER:
            patch = _inner(crop, zoom, dx, dy)
            if patch.size == 0:
                continue
            best = np.maximum(best, bank @ _vec(patch, grey))
        out = dict(zip(self.names, best.tolist()))
        if allowed is not None:
            out = {k: v for k, v in out.items() if k in allowed}
        return out

    def match(self, crop: np.ndarray, allowed: list[str] | None = None) -> tuple[str | None, float]:
        s = self.scores(crop, allowed)
        if not s:
            return None, 0.0
        ranked = sorted(s.items(), key=lambda kv: -kv[1])
        name, top = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else -1.0
        if top >= self.threshold and top - second >= self.margin:
            return name, top
        return None, top


class DeckTracker:
    """Works out the 8-card deck from what shows up in the hand during battles."""

    def __init__(self, min_sightings: int = 3):
        self.seen: Counter = Counter()
        self.min_sightings = min_sightings

    def update(self, hand: list[str | None], next_card: str | None) -> None:
        for n in hand + [next_card]:
            if n:
                self.seen[n] += 1

    def known(self) -> list[str]:
        db = load()
        out, champ = [], False
        for n, k in self.seen.most_common():
            if k < self.min_sightings or len(out) == 8:
                break
            if db.cards[n].champion:
                if champ:
                    continue
                champ = True
            out.append(n)
        return out

    def complete(self) -> list[str] | None:
        k = self.known()
        return k if len(k) == 8 else None


def fill_deck(known: list[str], hand: list[str]) -> list[str]:
    """An 8-card deck containing the known cards and the current hand (for the simulator; fillers are never played)."""
    db = load()
    deck = list(dict.fromkeys([n for n in hand if n] + list(known)))[:8]
    champ = any(db.cards[n].champion for n in deck)
    for n in ("Knight", "Archers", "Giant", "Musketeer", "Mini P.E.K.K.A", "Goblins", "Fireball", "Arrows", "Zap",
              "Skeletons", "Cannon", "Bats"):
        if len(deck) == 8:
            break
        if n not in deck and not (champ and db.cards[n].champion):
            deck.append(n)
    return deck
