import numpy as np

from flybrain.envs import CatchEnv
from flybrain.envs.catch import RIGHT


def play(env, policy, batch=500):
    env.reset(batch)
    done, steps = False, 0
    while not done:
        _, r, done = env.step(policy(env))
        steps += 1
    return env, r, steps


def test_greedy_policy_always_catches():
    env, r, steps = play(CatchEnv(seed=0), lambda e: (e.ball_x > e.paddle_x).astype(int))
    assert steps == env.height - 1
    assert np.all(r == 1)


def test_always_right_mostly_misses():
    env, r, _ = play(CatchEnv(seed=0), lambda e: np.full(len(e.ball_x), RIGHT))
    assert np.all(env.paddle_x == env.width - 1)
    assert set(np.unique(r)) == {-1.0, 1.0}
    assert (r == -1).mean() > 0.5


def test_features_one_hot():
    env = CatchEnv(width=5, encoding="absolute", seed=0)
    f = env.reset(7)
    assert f.shape == (7, 10) and np.all(f.sum(1) == 2)
    env = CatchEnv(width=5, encoding="relative", seed=0)
    f = env.reset(7)
    assert f.shape == (7, 9) and np.all(f.sum(1) == 1)
    assert np.all(f.argmax(1) == env.ball_x - env.paddle_x + 4)
