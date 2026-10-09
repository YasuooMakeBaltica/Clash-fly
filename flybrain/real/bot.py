"""Play real Clash Royale with the fly brain, through LDPlayer and adb.

    python -m flybrain.real.bot --adb "C:\\LDPlayer\\LDPlayer9\\adb.exe" --dry-run
    python -m flybrain.real.bot --adb "C:\\LDPlayer\\LDPlayer9\\adb.exe"          # actually taps
    python -m flybrain.real.bot --adb ... --learn --save models/fly_live.pt         # keeps learning

Once per second: screenshot -> read elixir, hand, towers, troops -> rebuild
a simulator state -> the fly brain votes on card and lane (same inputs as in
training) -> the lane placement helper picks the spot -> adb taps the card
and the spot. Start a battle yourself (or leave the bot running between
battles); it waits while no battle is on screen.

Your deck must be the deck the brain was trained with: Knight, Archers,
Giant, Musketeer, Mini P.E.K.K.A, Goblins, Fireball, Arrows.

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

from ..envs.clash.cards import DECK
from ..envs.clash.env import masks
from ..envs.clash.strategy import View, place
from ..envs.clash.env import features
from .adb import Adb
from .layout import Layout
from .perception import Perception, in_battle
from .state import build_sim

warnings.filterwarnings("ignore", message="Sparse")
ROOT = Path(__file__).resolve().parents[2]


def load_brain(path: str, device: str = "cpu"):
    sys.path.insert(0, str(ROOT / "scripts"))
    from train_clash import make_agent

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
            cv2.putText(out, str(u.size), (cx + 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    if note:
        cv2.putText(out, note, (6, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return out


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

    def __init__(self, agent, perception: Perception, layout: Layout, adb=None, learn: bool = False):
        self.agent, self.per, self.lay, self.adb, self.learn = agent, perception, layout, adb, learn
        self.battle_start = None
        self.prev_score = None
        self.played = 0

    def start_battle(self, now: float) -> None:
        self.battle_start, self.prev_score, self.played = now, None, 0
        self.per.reset()
        self.agent.begin_episode(1)

    def end_battle(self) -> tuple[int, int, str]:
        _, _, c_me, c_foe = tower_score(self.per.towers.hp)
        result = "win" if c_me > c_foe else "loss" if c_foe > c_me else "draw"
        if self.learn:
            self.agent.reward(np.array([{"win": 1.0, "loss": -1.0}.get(result, 0.0)], np.float32))
        self.battle_start = None
        return c_me, c_foe, result

    def step(self, img: np.ndarray, now: float) -> dict:
        h, w = img.shape[:2]
        obs = self.per.read(img)
        view = View(build_sim(obs, now - self.battle_start), 0)
        card_mask, lane_mask = masks(view)
        a = self.agent.act(features(view)[None], [card_mask[None], lane_mask[None]], learn=self.learn)[0]
        if self.learn:
            score = tower_score(obs.towers)
            if self.prev_score is not None:
                r = 0.5 * ((score[0] - self.prev_score[0]) - (score[1] - self.prev_score[1])) \
                    + 0.3 * ((score[2] - self.prev_score[2]) - (score[3] - self.prev_score[3]))
                if r:
                    self.agent.reward(np.array([np.clip(r, -1, 1)], np.float32))
            self.prev_score = score
        out = dict(obs=obs, action="wait", card=None, lane=int(a[1]), slot_xy=None, target_xy=None)
        if a[0] > 0:
            card = int(a[0]) - 1
            name = DECK[card].name
            if name in obs.hand:
                x, y = place(view, card, int(a[1]))
                cx, cy = self.lay.hand_slots[obs.hand.index(name)].center()
                out.update(card=name, slot_xy=(int(cx * w), int(cy * h)), target_xy=self.lay.tile_to_px(x, y, w, h),
                           action=f"{name} {'left' if a[1] == 0 else 'right'} -> tile ({x:.1f},{y:.1f})")
                if self.adb is not None:
                    self.adb.play_card(out["slot_xy"], out["target_xy"])
                    self.played += 1
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adb", default="adb", help="path to adb (LDPlayer ships one)")
    ap.add_argument("--serial", help="device serial from 'adb devices' (e.g. emulator-5554)")
    ap.add_argument("--connect", help="adb connect to this host:port first (e.g. 127.0.0.1:5555)")
    ap.add_argument("--brain", default=str(ROOT / "models/fly_v2.pt"))
    ap.add_argument("--layout", default="layout.json", help="from 'calibrate pick' (defaults are estimates)")
    ap.add_argument("--templates", default="templates/cards", help="card pictures from 'calibrate templates'")
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
    per = Perception(lay, args.templates)
    if len(per.matcher.templates) < len(DECK):
        missing = [c.name for c in DECK if c.name not in per.matcher.templates]
        print(f"warning: no card pictures for {missing}; the bot can't see those cards. Run calibrate templates.")
    agent, brain_args = load_brain(args.brain)
    print(f"brain {args.brain}: decision {brain_args.get('decision_ms')} ms, {agent.net.kc.stop - agent.net.kc.start} KCs")
    debug = Path(args.debug_dir)
    debug.mkdir(parents=True, exist_ok=True)

    bot = Bot(agent, per, lay, adb=None if args.dry_run else adb, learn=args.learn)
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
                bot.start_battle(t0)
                print("battle started")
            last_seen = t0
            out = bot.step(img, t0)
            obs, mt = out["obs"], t0 - bot.battle_start
            print(f"[{mt:5.1f}s] elixir {obs.elixir:4.1f} hand {obs.hand} units "
                  f"{sum(u.owner == 1 for u in obs.units)} enemy / {sum(u.owner == 0 for u in obs.units)} mine "
                  f"-> {out['action']}", flush=True)
            if int(mt) % 5 == 0:
                cv2.imwrite(str(debug / f"frame_{int(mt):03d}.png"), draw_overlay(img, lay, obs, out["action"]))
            time.sleep(max(0.0, args.every - (time.time() - t0)))
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()
