"""Talk to LDPlayer (or any Android emulator/device) through adb.

LDPlayer ships its own adb, e.g. ``C:\\LDPlayer\\LDPlayer9\\adb.exe``. The
first instance is usually ``emulator-5554`` (or ``127.0.0.1:5555``); run
``adb devices`` to see yours. Enable ADB in LDPlayer settings first.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np


LDPLAYER_DIRS = ["LDPlayer\\LDPlayer9", "LDPlayer\\LDPlayer4.0", "LDPlayer9", "LDPlayer4.0", "LDPlayer",
                 "Program Files\\LDPlayer\\LDPlayer9", "Program Files\\ldplayer9box", "XuanZhi\\LDPlayer9"]


def _ldplayer_dirs() -> list[Path]:
    """Folders where LDPlayer may be: the running emulator, the registry, then common install folders."""
    out: list[Path] = []
    if os.name != "nt":
        return out
    try:                                                    # the running LDPlayer window's program folder
        r = subprocess.run(["powershell", "-NoProfile", "-Command",
                            "(Get-Process dnplayer,ldplayer,LdVBoxHeadless -ErrorAction SilentlyContinue).Path"],
                           capture_output=True, text=True, timeout=15)
        out += [Path(ln.strip()).parent for ln in r.stdout.splitlines() if ln.strip()]
    except Exception:
        pass
    try:
        import winreg

        for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
            for key in ("Software\\XuanZhi\\LDPlayer9", "Software\\XuanZhi\\LDPlayer", "Software\\Changzhi\\LDPlayer",
                        "Software\\WOW6432Node\\XuanZhi\\LDPlayer9"):
                for value in ("InstallDir", "InstallPath"):
                    try:
                        with winreg.OpenKey(hive, key) as k:
                            out.append(Path(winreg.QueryValueEx(k, value)[0]))
                    except OSError:
                        pass
    except ImportError:
        pass
    roots = [Path(f"{d}:\\") for d in "CDEFG"]
    roots += [Path(os.environ[v]) for v in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA") if v in os.environ]
    out += [r / d for r in roots for d in LDPLAYER_DIRS]
    for r in roots:                                         # any version, e.g. E:\\LDPlayer\\LDPlayer14
        try:
            out += sorted((r / "LDPlayer").glob("LDPlayer*"), reverse=True)
        except OSError:
            pass
    return out


def find_adb(adb_path: str = "adb") -> str:
    """The given adb if it exists, else LDPlayer's adb from the usual install folders, else adb on PATH."""
    p = adb_path.strip().strip('"').strip("'")
    if p and p != "adb" and Path(p).is_file():
        return p
    if p and p != "adb" and not p.startswith(("%", "$")):
        raise FileNotFoundError(f"adb not found at {p}. Find adb.exe in your LDPlayer folder and pass its path.")
    for d in _ldplayer_dirs():
        cand = d / "adb.exe"
        if cand.is_file():
            return str(cand)
    found = shutil.which("adb")
    if found:
        return found
    raise FileNotFoundError("couldn't find adb. Keep LDPlayer open and try again, or pass --adb with the full path to "
                            "adb.exe in your LDPlayer folder (right-click the LDPlayer shortcut > Open file location).")


class Adb:
    def __init__(self, adb_path: str = "adb", serial: str | None = None, timeout: float = 10.0):
        self.adb_path, self.serial, self.timeout = find_adb(adb_path), serial, timeout

    def _cmd(self, *args: str) -> list[str]:
        base = [self.adb_path] + (["-s", self.serial] if self.serial else [])
        return base + list(args)

    def run(self, *args: str) -> bytes:
        try:
            return subprocess.run(self._cmd(*args), capture_output=True, check=True, timeout=self.timeout).stdout
        except subprocess.CalledProcessError as e:
            msg = (e.stderr or b"").decode(errors="replace").strip()
            if "more than one" in msg:
                msg += f"\n-> pick one with --serial (from: {', '.join(self.devices())})"
            elif "no devices" in msg or "not found" in msg:
                msg += "\n-> is LDPlayer running with ADB debugging set to 'open local connection'?"
            raise RuntimeError(f"adb {' '.join(args)} failed: {msg}") from None

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
