"""Catch the ball: a ball falls one row per step, the paddle moves left/right.

Batched: ``batch`` independent games advance in lockstep, and every episode
lasts exactly ``height - 1`` steps, so all games end together.
"""

from __future__ import annotations

import numpy as np

LEFT, RIGHT = 0, 1


class CatchEnv:
    n_actions = 2

    def __init__(self, width: int = 8, height: int = 9, paddle_radius: int = 1,
                 encoding: str = "absolute", shaping: float = 0.0, seed: int | None = None):
        if encoding not in ("absolute", "relative"):
            raise ValueError(f"unknown encoding {encoding!r}")
        self.width, self.height = width, height
        self.paddle_radius = paddle_radius
        self.encoding = encoding
        self.shaping = shaping
        self.rng = np.random.default_rng(seed)

    @property
    def n_channels(self) -> int:
        """Number of input channels the agent sees (each drives a PN group)."""
        if self.encoding == "absolute":
            return 2 * self.width            # one-hot ball x, one-hot paddle x
        return 2 * self.width - 1            # one-hot (ball x - paddle x)

    def reset(self, batch: int = 1) -> np.ndarray:
        self.ball_x = self.rng.integers(0, self.width, size=batch)
        self.paddle_x = self.rng.integers(0, self.width, size=batch)
        self.ball_y = 0
        return self.features()

    def features(self) -> np.ndarray:
        """(batch, n_channels) binary input."""
        b = len(self.ball_x)
        f = np.zeros((b, self.n_channels), dtype=np.float32)
        rows = np.arange(b)
        if self.encoding == "absolute":
            f[rows, self.ball_x] = 1
            f[rows, self.width + self.paddle_x] = 1
        else:
            f[rows, self.ball_x - self.paddle_x + self.width - 1] = 1
        return f

    def step(self, action: np.ndarray):
        """Returns (features, reward, done). Reward is +1 catch / -1 miss at the end."""
        action = np.asarray(action)
        dist_before = np.abs(self.ball_x - self.paddle_x)
        self.paddle_x = np.clip(self.paddle_x + np.where(action == RIGHT, 1, -1), 0, self.width - 1)
        self.ball_y += 1
        done = self.ball_y >= self.height - 1
        if done:
            caught = np.abs(self.ball_x - self.paddle_x) <= self.paddle_radius
            reward = np.where(caught, 1.0, -1.0)
        else:
            closer = np.abs(self.ball_x - self.paddle_x) < dist_before
            reward = self.shaping * np.where(closer, 1.0, -1.0)
        return self.features(), reward.astype(np.float32), done

    def caught(self) -> np.ndarray:
        return np.abs(self.ball_x - self.paddle_x) <= self.paddle_radius
