import numpy as np
import torch

from flybrain.agent import AgentConfig, FlyAgent
from flybrain.connectome import synthetic
from flybrain.envs import CatchEnv


def make_agent(env):
    conn = synthetic(n_pn=80, n_kc=400, n_mbon=16, n_dan=20, n_glomeruli=20, seed=2)
    return FlyAgent(conn, env.n_channels, cfg=AgentConfig(seed=0, decision_steps=30))


def test_channels_and_groups():
    env = CatchEnv(width=5)
    agent = make_agent(env)
    assert torch.all(agent.channel_pn.sum(0)[agent.net.sl["PN"]] == 1)  # each PN in one channel
    assert torch.all(agent.channel_pn.sum(1) > 0)
    assert torch.all(agent.groups.sum(0) == 1)  # each MBON votes for exactly one action
    sides = agent.net.conn.neurons.iloc[agent.net.mbon]["side"].to_numpy()
    assert np.all(sides[agent.groups[0].bool().numpy()] == "left")


def test_reward_drives_reward_dans():
    env = CatchEnv(width=5)
    agent = make_agent(env)
    da_r, da_p = agent.dopamine(np.array([1.0, -1.0, 0.0]))
    assert da_r[0] > 0.2 and da_p[0] < da_r[0]
    assert da_p[1] > 0.2 and da_r[1] < da_p[1]
    assert da_r[2] < 0.05 and da_p[2] < 0.05


def test_act_returns_valid_actions_and_tags():
    env = CatchEnv(width=5, seed=0)
    agent = make_agent(env)
    agent.begin_episode(4)
    a = agent.act(env.reset(4))
    assert a.shape == (4,) and set(a) <= {0, 1}
    assert agent.rule.chosen.sum() > 0
