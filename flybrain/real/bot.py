"""Play real Clash Royale with the fly brain, through LDPlayer and adb.

    python -m flybrain.real.bot --adb "C:\\LDPlayer\\LDPlayer9\\adb.exe" --dry-run
    python -m flybrain.real.bot --adb "C:\\LDPlayer\\LDPlayer9\\adb.exe"          # actually taps
    python -m flybrain.real.bot --adb ... --learn --save models/fly_live.pt         # keeps learning

Once per second: screenshot -> read elixir, hand, towers, troops -> rebuild
a full-game simulator state -> the fly brain votes on a card role and a lane
(same inputs as in training) -> the card in hand with that role is chosen
and placed by the same helpers as in the simulator -> adb taps the card and
the spot. Start a battle yourself (or leave the bot running between
battles); it waits while no battle is on screen.

The bot recognises the cards in your hand from the official card art
(downloaded once to templates/official/) and works out your 8-card deck as
the cards cycle during a battle. When it has seen all 8 it writes them to
decks/fly.txt, so changing decks in the game needs no setup (--no-auto-deck
turns this off). Card pictures saved with ``calibrate templates`` still take
priority if the official art doesn't match your screen.

Supercell's terms of service prohibit automation. Use an alt account.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch

from ..envs.royale.db import ROLES, load
from ..envs.royale.decks import load_deck
from ..envs.royale.env import HEADS, brain_masks, decode, features, pick_card
from ..envs.royale.guard import guarded_action
from ..envs.royale.strategy import Coach, View, place
from .adb import Adb
from .cards import OFFICIAL_DIR, DeckTracker, download_official, fill_deck
from .layout import Layout
from .perception import BadgeDetector, Perception, default_detector, in_battle
from .state import build_sim
from ..deck_editor import write_deck

warnings.filterwarnings("ignore", message="Sparse")
ROOT = Path(__file__).resolve().parents[2]


def load_brain(path: str, device: str = "cpu"):
    """A full-card-pool brain from scripts/train_royale.py."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from train_royale import make_agent

    ckpt = torch.load(path, weights_only=False, map_location=device)
    args = argparse.Namespace(**ckpt["args"])
    args.device = device
    agent = make_agent(args)
    agent.load_state_dict(ckpt["agent"])
    agent.cfg.epsilon = 0.0
    return agent, ckpt["args"]


def draw_overlay(img: np.ndarray, lay: Layout, obs=None, note: str = "") -> np.ndarray:
    """Layout boxes, tile grid and what perception read, drawn on a screenshot."""
    out = img.copy()
    h, w = out.shape[:2]

    def rect(box, col, label=""):
        x0, y0, x1, y1 = box.px(w, h)
        cv2.rectangle(out, (x0, y0), (x1, y1), col, 2)
        if label:
            cv2.putText(out, label, (x0 + 2, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

    for tx in range(0, 19, 3):
        cv2.line(out, lay.tile_to_px(tx, 0, w, h), lay.tile_to_px(tx, 32, w, h), (255, 255, 255), 1)
    for ty in range(0, 33, 4):
        cv2.line(out, lay.tile_to_px(0, ty, w, h), lay.tile_to_px(18, ty, w, h), (255, 255, 255), 1)
    rect(lay.arena, (0, 255, 255), "arena")
    rect(lay.elixir_bar, (255, 0, 255), f"elixir {obs.elixir:.1f}" if obs else "elixir")
    for i, b in enumerate(lay.hand_slots):
        rect(b, (0, 255, 0), (obs.hand[i] or "?") if obs else f"slot {i + 1}")
    rect(lay.next_slot, (0, 200, 0), (obs.next_card or "?") if obs else "next")
    for key, b in lay.tower_bars().items():
        rect(b, (255, 128, 0) if key[0] == 0 else (0, 0, 255), f"{obs.towers.get(key, 1):.2f}" if obs else "")
    if obs:
        for u in obs.units:
            cx, cy = lay.tile_to_px(u.x, u.y, w, h)
            cv2.circle(out, (cx, cy), 9, (255, 128, 0) if u.owner == 0 else (0, 0, 255), 2)
            cv2.putText(out, u.char or str(u.size), (cx + 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    if note:
        cv2.putText(out, note, (6, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return out


def troop_summary(units, owner: int) -> str:
    """e.g. 'HogRider, Skeleton x3' (or just a count when types are unknown)."""
    from collections import Counter

    seen = [u for u in units if u.owner == owner]
    if not seen:
        return "-"
    if all(u.char is None for u in seen):
        return str(sum(u.size for u in seen))
    return ", ".join(f"{n} x{k}" if k > 1 else n for n, k in Counter(u.char or "?" for u in seen).most_common())


def tower_score(towers: dict) -> tuple[float, float, int, int]:
    """(damage dealt, damage taken) in princess-tower units, and crowns (you, enemy) from destroyed towers."""
    dealt = sum(1 - v for (o, _, _), v in towers.items() if o == 1)
    taken = sum(1 - v for (o, _, _), v in towers.items() if o == 0)

    def crowns(owner):
        if towers.get((owner, "king", None), 1) <= 0:
            return 3
        return sum(1 for (o, k, _), v in towers.items() if o == owner and k == "princess" and v <= 0)

    return dealt, taken, crowns(1), crowns(0)


class Bot:
    """One decision per call to :meth:`step`; ``adb`` does the tapping (None = dry run)."""

    def __init__(self, agent, perception: Perception, layout: Layout, deck: list[str], adb=None, learn: bool = False,
                 auto_deck: bool = True, guard: bool = True):
        self.agent, self.per, self.lay, self.adb, self.learn = agent, perception, layout, adb, learn
        self.db = load()
        self.tracker = DeckTracker() if auto_deck else None
        self.leak_at: float | None = 9.5
        self.guard = guard
        self.coach = Coach()
        self.set_deck(deck)
        self.battle_start = None
        self.prev_score = None
        self.played = 0
        self.champion_played_at = None
        self.last_ability = -1e9

    def set_deck(self, deck: list[str]) -> None:
        self.deck = list(deck)
        self.champion = next((n for n in self.deck if self.db.cards[n].champion), None)
        self.per.hand_reader.deck_hint = self.deck

    def learned_deck(self) -> list[str] | None:
        """All 8 cards seen this battle (None until then)."""
        return self.tracker.complete() if self.tracker else None

    def _battle_deck(self, obs) -> list[str]:
        """The deck file, unless the hand shows cards that aren't in it (then: what's been seen so far)."""
        if self.tracker is None:
            return self.deck
        learned = self.tracker.complete()
        if learned:
            return learned
        if all(n in self.deck for n in obs.hand + [obs.next_card] if n):
            return self.deck
        return fill_deck(self.tracker.known(), [n for n in obs.hand if n])

    def start_battle(self, now: float) -> None:
        if self.tracker is not None:
            self.tracker = DeckTracker(self.tracker.min_sightings)
        self.battle_start, self.prev_score, self.played = now, None, 0
        self.champion_played_at, self.last_ability = None, -1e9
        self.per.reset()
        self.agent.begin_episode(1)

    def end_battle(self) -> tuple[int, int, str]:
        _, _, c_me, c_foe = tower_score(self.per.towers.hp)
        result = "win" if c_me > c_foe else "loss" if c_foe > c_me else "draw"
        if self.learn:
            self.agent.reward(np.array([{"win": 1.0, "loss": -1.0}.get(result, 0.0)], np.float32))
        self.battle_start = None
        return c_me, c_foe, result

    def _maybe_ability(self, view: View, now: float, w: int, h: int) -> bool:
        """Tap the champion ability when our champion is probably alive and enemies are near our side."""
        if self.champion is None or self.champion_played_at is None or self.adb is None:
            return False
        ab = self.db.characters[self.db.cards[self.champion].summons[0][0].name].ability
        if ab is None or now - self.last_ability < ab.cooldown + 1 or now - self.champion_played_at > 60:
            return False
        if view.elixir < ab.cost + 1 or not any(view.threats(lane) for lane in (0, 1)):
            return False
        cx, cy = self.lay.ability_button.center()
        self.adb.tap(int(cx * w), int(cy * h))
        self.last_ability = now
        return True

    def step(self, img: np.ndarray, now: float) -> dict:
        h, w = img.shape[:2]
        obs = self.per.read(img)
        if self.tracker is not None:
            self.tracker.update(obs.hand, obs.next_card)
        deck = self._battle_deck(obs)
        champion = next((n for n in deck if self.db.cards[n].champion), None)
        if champion != self.champion and champion is not None:
            self.champion, self.champion_played_at = champion, None
        sim = build_sim(obs, now - self.battle_start, deck, db=self.db)
        view = View(sim, 0)
        heads = tuple(getattr(self.agent, "heads", HEADS))
        # Full elixir: waiting would waste it, so the brain must pick a card.
        no_wait = self.leak_at is not None and view.elixir >= self.leak_at
        masks = brain_masks(view, heads, no_wait)
        a = self.agent.act(features(view)[None], [m[None] for m in masks], learn=self.learn)[0]
        if self.learn:
            score = tower_score(obs.towers)
            if self.prev_score is not None:
                r = 0.5 * ((score[0] - self.prev_score[0]) - (score[1] - self.prev_score[1])) \
                    + 0.3 * ((score[2] - self.prev_score[2]) - (score[3] - self.prev_score[3]))
                if r:
                    self.agent.reward(np.array([np.clip(r, -1, 1)], np.float32))
            self.prev_score = score
        role_i, lane = decode(a, heads)
        out = dict(obs=obs, action="wait", card=None, role=ROLES[role_i - 1] if role_i > 0 else None, lane=lane,
                   slot_xy=None, target_xy=None, who="fly")
        if self.guard:
            card, lane, out["who"] = guarded_action(view, role_i, lane, self.coach)
        else:
            card = pick_card(view, ROLES[role_i - 1]) if role_i > 0 else None
        if card is not None and card in obs.hand:
            role = self.db.cards[card].role
            x, y = place(view, card, lane)
            cx, cy = self.lay.hand_slots[obs.hand.index(card)].center()
            who = "" if out["who"] == "fly" else f" [{out['who']}]"
            out.update(card=card, role=role, lane=lane, slot_xy=(int(cx * w), int(cy * h)),
                       target_xy=self.lay.tile_to_px(x, y, w, h),
                       action=f"{card} ({role}) {'left' if lane == 0 else 'right'} -> tile ({x:.1f},{y:.1f}){who}")
            if self.adb is not None:
                self.adb.play_card(out["slot_xy"], out["target_xy"])
                self.played += 1
                if card == self.champion:
                    self.champion_played_at = now
        elif out["who"] == "veto":
            out["action"] = "wait (spell vetoed: nothing worth hitting)"
        if out["card"] is None and self._maybe_ability(view, now, w, h):
            out["action"] = "champion ability"
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adb", default="adb", help="path to adb (LDPlayer ships one)")
    ap.add_argument("--serial", help="device serial from 'adb devices' (e.g. emulator-5554)")
    ap.add_argument("--connect", help="adb connect to this host:port first (e.g. 127.0.0.1:5555)")
    ap.add_argument("--brain", default=str(ROOT / "models/fly_royale.pt"))
    ap.add_argument("--deck", default=str(ROOT / "decks/fly.txt"), help="your in-game battle deck (8 cards)")
    ap.add_argument("--layout", default="layout.json", help="from 'calibrate pick' (defaults are estimates)")
    ap.add_argument("--templates", default="templates/cards", help="card pictures from 'calibrate templates' (optional)")
    ap.add_argument("--official", default=str(OFFICIAL_DIR), help="official card art (downloaded on first run)")
    ap.add_argument("--troops", default=str(ROOT / "models/troops.pt"), help="trained troop detector weights")
    ap.add_argument("--no-guard", action="store_true", help="pure fly brain: no coach rules on top of its moves")
    ap.add_argument("--no-auto-deck", action="store_true", help="don't learn the deck from the hand; use --deck only")
    ap.add_argument("--dry-run", action="store_true", help="read the screen and decide, but never tap")
    ap.add_argument("--learn", action="store_true", help="keep learning from tower damage during real matches")
    ap.add_argument("--save", default="models/fly_live.pt", help="where --learn saves the brain after each match")
    ap.add_argument("--every", type=float, default=1.0, help="seconds between decisions")
    ap.add_argument("--matches", type=int, default=0, help="stop after this many matches (0 = run until Ctrl+C)")
    ap.add_argument("--debug-dir", default="runs/real", help="saves an annotated screenshot every few seconds")
    args = ap.parse_args()

    adb = Adb(args.adb, args.serial)
    if args.connect:
        print(adb.connect(args.connect))
    lay = Layout.load(args.layout)
    if not Path(args.layout).exists():
        print(f"warning: {args.layout} not found, using estimated screen positions. Run calibrate first.")
    if not Path(args.official).exists() or len(list(Path(args.official).glob("*.png"))) < len(load().pool()):
        print("downloading the official card pictures (once)...")
        failed = download_official(args.official)
        if failed:
            print(f"warning: couldn't download {len(failed)} card pictures ({failed[:5]}...); check your internet")
    detector = default_detector(args.troops)
    per = Perception(lay, args.templates, official_dir=args.official, detector=detector)
    print("troops:", "trained detector (types known)" if not isinstance(detector, BadgeDetector)
          else "colour badges only (types unknown; models/troops.pt not found)")
    deck = load_deck(args.deck)
    deck_mtime = Path(args.deck).stat().st_mtime
    print("deck:", deck, "" if args.no_auto_deck else "(updates itself from what's in your hand)")
    missing = [n for n in deck if n not in per.matcher.templates and n not in per.hand_reader.official.names] \
        if per.hand_reader.official else [n for n in deck if n not in per.matcher.templates]
    if missing:
        print(f"warning: no card pictures for {missing}; the bot can't see those cards.")
    agent, brain_args = load_brain(args.brain)
    print(f"brain {args.brain}: decision {brain_args.get('decision_ms')} ms, {agent.net.kc.stop - agent.net.kc.start} KCs")
    debug = Path(args.debug_dir)
    debug.mkdir(parents=True, exist_ok=True)

    bot = Bot(agent, per, lay, deck, adb=None if args.dry_run else adb, learn=args.learn,
              auto_deck=not args.no_auto_deck, guard=not args.no_guard)
    last_seen, matches = 0.0, 0
    try:
        while True:
            t0 = time.time()
            img = adb.screencap()
            if not in_battle(img, lay):
                if bot.battle_start is not None and t0 - last_seen > 5:
                    length = last_seen - bot.battle_start
                    c_me, c_foe, result = bot.end_battle()
                    print(f"battle over after {length:.0f}s: crowns {c_me}-{c_foe} ({result}), {bot.played} cards played")
                    learned = bot.learned_deck()
                    if learned and sorted(learned) != sorted(bot.deck):
                        try:
                            write_deck(Path(args.deck).parent, Path(args.deck).stem, learned)
                            bot.set_deck(learned)
                            deck_mtime = Path(args.deck).stat().st_mtime
                            print(f"learned your deck from the hand, saved to {args.deck}: {learned}")
                        except ValueError as e:
                            print(f"saw a deck that doesn't check out, not saving it: {e}")
                    if args.learn:
                        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
                        torch.save(dict(agent=agent.state_dict(), args=brain_args), args.save)
                        print(f"saved {args.save}")
                    matches += 1
                    if args.matches and matches >= args.matches:
                        break
                time.sleep(1.0)
                continue
            if bot.battle_start is None:
                if Path(args.deck).stat().st_mtime != deck_mtime:      # edited with the deck editor
                    try:
                        bot.set_deck(load_deck(args.deck))
                        deck_mtime = Path(args.deck).stat().st_mtime
                        print("deck changed:", bot.deck)
                    except ValueError as e:
                        print(f"deck file has a problem, keeping the old deck: {e}")
                bot.start_battle(t0)
                print("battle started")
            last_seen = t0
            out = bot.step(img, t0)
            obs, mt = out["obs"], t0 - bot.battle_start
            print(f"[{mt:5.1f}s] elixir {obs.elixir:4.1f} hand {obs.hand} enemy {troop_summary(obs.units, 1)} "
                  f"mine {troop_summary(obs.units, 0)} -> {out['action']}", flush=True)
            if int(mt) % 5 == 0:
                cv2.imwrite(str(debug / f"frame_{int(mt):03d}.png"), draw_overlay(img, lay, obs, out["action"]))
            time.sleep(max(0.0, args.every - (time.time() - t0)))
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()
