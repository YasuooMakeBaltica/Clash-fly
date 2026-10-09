"""Find troops on the arena and tell what they are: a small CenterNet-style CNN.

Trained with scripts/train_troops.py on KataCR's Clash-Royale-Detection-Dataset
(MIT licence, github.com/wty-yy/Clash-Royale-Detection-Dataset): about 7,000
real battle frames where every troop is labelled with its type and side. The
dataset frames are 568x896 crops of the arena; :func:`arena_crop` cuts the
same region out of your screenshot using the calibrated layout, so the net
sees the same geometry it was trained on.

The net predicts, at 1/8 of the input resolution (one cell is about half a
tile), a heatmap per side (you / enemy) of where troops are, a
troop-type score per cell and a sub-cell offset. Peaks in the heatmap become
:class:`SeenUnit` with ``char`` set to the simulator character, so the
rebuilt game state knows a Hog Rider from a Minion (air, building targeting,
hit points ...).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layout import Box, Layout

# Where the arena (18x32 tiles) sits inside a 568x896 dataset frame, as
# fractions of the frame. Fitted by aligning dataset frames with a real
# LDPlayer 540x960 screenshot (ECC image registration on edges, 12 frames:
# screen x = 39.3 + 0.814 u, screen y = 14.8 + 0.834 v), then using the
# calibrated tile grid of the default layout.
FRAME_ARENA = Box(0.0036, 0.0860, 0.9948, 0.9723)
FRAME_SIZE = (568, 896)
INPUT_SIZE = (288, 448)          # (w, h) the net sees: the frame at half size, padded to a multiple of 16
SCALE = 0.5
STRIDE = 8

# Dataset class name -> simulator character (envs/royale/data/cards.json), None = not a troop we track.
KATA_TO_CHAR = {
    "skeleton": "Skeleton", "electro-spirit": "ElectroSpirit", "fire-spirit": "FireSpirits", "ice-spirit": "IceSpirits",
    "heal-spirit": "HealSpirit", "goblin": "Goblin", "spear-goblin": "SpearGoblin", "bomber": "Bomber", "bat": "Bat",
    "ice-golem": "IceGolemite", "barbarian": "Barbarian", "wall-breaker": "Wallbreaker", "archer": "Archer",
    "knight": "Knight", "minion": "Minion", "cannon": "Cannon", "skeleton-barrel": "SkeletonBalloon",
    "firecracker": "Firecracker", "royal-recruit": "Recruit", "tombstone": "Tombstone", "mega-minion": "MegaMinion",
    "dart-goblin": "BlowdartGoblin", "elixir-golem-big": "ElixirGolem4", "elixir-golem-mid": "ElixirGolem2",
    "elixir-golem-small": "ElixirGolem1", "guard": "SkeletonWarrior", "miner": "Miner", "princess": "Princess",
    "ice-wizard": "IceWizard", "royal-ghost": "Ghost", "bandit": "Assassin", "fisherman": "Fisherman",
    "skeleton-dragon": "SkeletonDragon", "mortar": "Mortar", "tesla": "Tesla", "mini-pekka": "MiniPekka",
    "musketeer": "Musketeer", "goblin-cage": "GoblinCage", "goblin-brawler": "GoblinBrawler", "valkyrie": "Valkyrie",
    "battle-ram": "BattleRam", "bomb-tower": "BombTower", "flying-machine": "DartBarrell", "hog-rider": "HogRider",
    "battle-healer": "BattleHealer", "furnace": "FirespiritHut", "zappy": "MiniZapMachine", "baby-dragon": "BabyDragon",
    "dark-prince": "DarkPrince", "hunter": "Hunter", "goblin-drill": "GoblinDrill", "electro-wizard": "ElectroWizard",
    "inferno-dragon": "InfernoDragon", "phoenix-big": "Phoenix", "phoenix-egg": "PhoenixEgg",
    "phoenix-small": "PhoenixNoRespawn", "magic-archer": "EliteArcher", "lumberjack": "RageBarbarian",
    "night-witch": "DarkWitch", "mother-witch": "WitchMother", "hog": "VoodooHog", "golden-knight": "GoldenKnight",
    "skeleton-king": "SkeletonKing", "mighty-miner": "MightyMiner", "rascal-boy": "RascalBoy",
    "rascal-girl": "RascalGirl", "giant": "Giant", "goblin-hut": "GoblinHut", "inferno-tower": "InfernoTower",
    "wizard": "Wizard", "royal-hog": "RoyalHog", "witch": "Witch", "balloon": "Balloon", "prince": "Prince",
    "electro-dragon": "ElectroDragon", "bowler": "Bowler", "executioner": "AxeMan", "cannon-cart": "MovingCannon",
    "ram-rider": "RamRider", "archer-queen": "ArcherQueen", "monk": "Monk", "royal-giant": "RoyalGiant",
    "elite-barbarian": "AngryBarbarian", "barbarian-hut": "BarbarianHut", "elixir-collector": "ElixirCollector",
    "giant-skeleton": "GiantSkeleton", "goblin-giant": "GoblinGiant", "x-bow": "Xbow", "sparky": "ZapMachine",
    "pekka": "Pekka", "electro-giant": "ElectroGiant", "mega-knight": "MegaKnight", "lava-hound": "LavaHound",
    "lava-pup": "LavaPups", "golem": "Golem", "golemite": "Golemite", "little-prince": None, "royal-guardian": None,
}
# Evolutions look different but play like their base troop.
for _name in list(KATA_TO_CHAR):
    KATA_TO_CHAR.setdefault(_name + "-evolution", KATA_TO_CHAR[_name])
CLASSES = sorted({c for c in KATA_TO_CHAR.values() if c})


def kata_class(name: str) -> str | None:
    if name.endswith("-evolution"):
        name = name[: -len("-evolution")]
    return KATA_TO_CHAR.get(name)


# --------------------------------------------------------------------- model
def _cbr(cin, cout, stride=1, dil=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, padding=dil, dilation=dil, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class _Res(nn.Module):
    def __init__(self, c, dil=1):
        super().__init__()
        self.a, self.b = _cbr(c, c, dil=dil), nn.Sequential(
            nn.Conv2d(c, c, 3, padding=dil, dilation=dil, bias=False), nn.BatchNorm2d(c))

    def forward(self, x):
        return F.relu(x + self.b(self.a(x)))


class TroopNet(nn.Module):
    """Input (B, 3, 448, 288) in 0..1 -> heat (B, 2, 56, 36), cls (B, K, 56, 36), offset (B, 2, 56, 36)."""

    def __init__(self, n_classes: int = len(CLASSES), width: int = 32):
        super().__init__()
        w = width
        self.s2 = nn.Sequential(_cbr(3, w // 2, 2), _cbr(w // 2, w, 2))                   # /4
        self.s3 = nn.Sequential(_cbr(w, 2 * w, 2), _Res(2 * w))                             # /8
        self.s4 = nn.Sequential(_cbr(2 * w, 3 * w, 2), _Res(3 * w), _Res(3 * w, dil=2))     # /16
        self.s5 = nn.Sequential(_cbr(3 * w, 4 * w, 2), _Res(4 * w), _Res(4 * w, dil=2))     # /32
        self.up5 = nn.Conv2d(4 * w, 3 * w, 1)
        self.up4 = nn.Conv2d(3 * w, 2 * w, 1)
        self.lat3 = nn.Conv2d(2 * w, 2 * w, 1)
        self.fuse = nn.Sequential(_cbr(2 * w, 2 * w), _Res(2 * w))
        self.heat = nn.Sequential(_cbr(2 * w, w), nn.Conv2d(w, 2, 1))
        self.cls = nn.Sequential(_cbr(2 * w, 2 * w), nn.Conv2d(2 * w, n_classes, 1))
        self.off = nn.Conv2d(2 * w, 2, 1)
        nn.init.constant_(self.heat[-1].bias, -4.6)          # start with "nothing here" (p = 0.01)

    def forward(self, x):
        x = x - 0.5
        f3 = self.s3(self.s2(x))
        f4 = self.s4(f3)
        f5 = self.s5(f4)
        f4 = f4 + F.interpolate(self.up5(f5), size=f4.shape[-2:], mode="nearest")
        f3 = self.lat3(f3) + F.interpolate(self.up4(f4), size=f3.shape[-2:], mode="nearest")
        f = self.fuse(f3)
        return self.heat(f), self.cls(f), self.off(f)


# ----------------------------------------------------------------- geometry
def frame_box(lay: Layout, w: int, h: int) -> tuple[float, float, float, float]:
    """Pixel box (x0, y0, x1, y1) of the screenshot that corresponds to a 568x896 dataset frame."""
    ax0, ay0, ax1, ay1 = lay.arena.x0 * w, lay.arena.y0 * h, lay.arena.x1 * w, lay.arena.y1 * h
    fw = (ax1 - ax0) / (FRAME_ARENA.x1 - FRAME_ARENA.x0)
    fh = (ay1 - ay0) / (FRAME_ARENA.y1 - FRAME_ARENA.y0)
    fx0, fy0 = ax0 - FRAME_ARENA.x0 * fw, ay0 - FRAME_ARENA.y0 * fh
    return fx0, fy0, fx0 + fw, fy0 + fh


def arena_crop(img: np.ndarray, lay: Layout) -> np.ndarray:
    """The dataset-frame region of a screenshot, resized to the net's input (BGR uint8, 448x288)."""
    h, w = img.shape[:2]
    fx0, fy0, fx1, fy1 = frame_box(lay, w, h)
    sx, sy = (fx1 - fx0) / FRAME_SIZE[0], (fy1 - fy0) / FRAME_SIZE[1]
    # affine map from net-input pixels to screenshot pixels
    m = np.array([[sx / SCALE, 0, fx0], [0, sy / SCALE, fy0]], np.float32)
    return cv2.warpAffine(img, m, INPUT_SIZE, flags=cv2.WARP_INVERSE_MAP | cv2.INTER_AREA,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def frame_to_tile(u: float, v: float) -> tuple[float, float]:
    """Dataset-frame pixel (568x896) -> simulator tile."""
    lay = Layout()
    lay.arena = FRAME_ARENA
    return lay.frac_to_tile(u / FRAME_SIZE[0], v / FRAME_SIZE[1])


def to_tensor(crops: np.ndarray) -> torch.Tensor:
    """(B, H, W, 3) BGR uint8 -> (B, 3, H, W) RGB float."""
    return torch.from_numpy(crops[..., ::-1].copy()).permute(0, 3, 1, 2).float() / 255.0


# ----------------------------------------------------------------- detector
class TroopDetector:
    """Drop-in for perception.BadgeDetector: ``detect(img, layout) -> list[SeenUnit]`` with troop types."""

    def __init__(self, weights: str | Path, threshold: float = 0.35, device: str = "cpu"):
        ckpt = torch.load(weights, map_location=device, weights_only=False)
        self.classes = ckpt["classes"]
        self.net = TroopNet(len(self.classes), ckpt.get("width", 32)).to(device).eval()
        self.net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ckpt["model"].items()})
        self.threshold, self.device = threshold, device
        self.ground = ckpt.get("ground", 0.0)       # extra shift (tiles) from the predicted point to the feet

    @torch.no_grad()
    def raw(self, crop: np.ndarray):
        heat, cls, off = self.net(to_tensor(crop[None]).to(self.device))
        return torch.sigmoid(heat)[0].cpu().numpy(), torch.softmax(cls, 1)[0].cpu().numpy(), off[0].cpu().numpy()

    def peaks(self, heat: np.ndarray, cls: np.ndarray, off: np.ndarray, threshold: float | None = None):
        """[(side, u, v, score, class index, class prob)] in dataset-frame pixels."""
        thr = self.threshold if threshold is None else threshold
        t = torch.from_numpy(heat)[None]
        keep = (t == F.max_pool2d(t, 3, 1, 1))[0].numpy() & (heat >= thr)
        out = []
        for side, gy, gx in zip(*np.nonzero(keep)):
            u = (gx + 0.5 + off[0, gy, gx]) * STRIDE / SCALE
            v = (gy + 0.5 + off[1, gy, gx]) * STRIDE / SCALE
            k = int(cls[:, gy, gx].argmax())
            out.append((int(side), float(u), float(v), float(heat[side, gy, gx]), k, float(cls[k, gy, gx])))
        return out

    def detect(self, img: np.ndarray, lay: Layout, extra_ignore=None):
        from .perception import SeenUnit

        heat, cls, off = self.raw(arena_crop(img, lay))
        units = []
        for side, u, v, score, k, p in self.peaks(heat, cls, off):
            tx, ty = frame_to_tile(u, v)
            if not (-0.5 <= tx <= 18.5 and -0.5 <= ty <= 32.5):
                continue
            units.append(SeenUnit(owner=side, x=float(np.clip(tx, 0, 18)), y=float(np.clip(ty - self.ground, 0, 32)),
                                  size=1, char=self.classes[k], score=score))
        return units
