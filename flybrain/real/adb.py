"""Talk to LDPlayer (or any Android emulator/device) through adb.

LDPlayer ships its own adb, e.g. ``C:\\LDPlayer\\LDPlayer9\\adb.exe``. The
first instance is usually ``emulator-5554`` (or ``127.0.0.1:5555``); run
``adb devices`` to see yours. Enable ADB in LDPlayer settings first.
"""

from __future__ import annotations

import re
import subprocess
import time

import cv2
import numpy as np


class Adb:
    def __init__(self, adb_path: str = "adb", serial: str | None = None, timeout: float = 10.0):
        self.adb_path, self.serial, self.timeout = adb_path, serial, timeout

    def _cmd(self, *args: str) -> list[str]:
        base = [self.adb_path] + (["-s", self.serial] if self.serial else [])
        return base + list(args)

    def run(self, *args: str) -> bytes:
        return subprocess.run(self._cmd(*args), capture_output=True, check=True, timeout=self.timeout).stdout

    def connect(self, host: str) -> str:
        return subprocess.run([self.adb_path, "connect", host], capture_output=True, text=True,
                              timeout=self.timeout).stdout.strip()

    def devices(self) -> list[str]:
        out = subprocess.run([self.adb_path, "devices"], capture_output=True, text=True, timeout=self.timeout).stdout
        return [ln.split()[0] for ln in out.splitlines()[1:] if ln.strip().endswith("device")]

    def screen_size(self) -> tuple[int, int]:
        out = self.run("shell", "wm", "size").decode()
        m = re.findall(r"(\d+)x(\d+)", out)
        w, h = map(int, m[-1])
        return w, h

    def screencap(self) -> np.ndarray:
        """Current screen as a BGR image."""
        png = self.run("exec-out", "screencap", "-p")
        img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("screencap returned no image; is the emulator running and ADB enabled?")
        return img

    def tap(self, x: int, y: int) -> None:
        self.run("shell", "input", "tap", str(int(x)), str(int(y)))

    def swipe(self, x0: int, y0: int, x1: int, y1: int, ms: int = 150) -> None:
        self.run("shell", "input", "swipe", str(int(x0)), str(int(y0)), str(int(x1)), str(int(y1)), str(ms))

    def play_card(self, slot_xy: tuple[int, int], target_xy: tuple[int, int], pause: float = 0.08) -> None:
        """Tap a hand slot, then tap where to drop it (how Clash Royale deploys by tap)."""
        self.tap(*slot_xy)
        time.sleep(pause)
        self.tap(*target_xy)
