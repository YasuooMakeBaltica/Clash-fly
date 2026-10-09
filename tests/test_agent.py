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
    assert torch.all(agent.groups[0].sum(0) == 1)  # each MBON votes for exactly one action
    sides = agent.net.conn.neurons.iloc[agent.net.mbon]["side"].to_numpy()
    assert np.all(sides[agent.groups[0][0].bool().numpy()] == "left")


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


def make_multi_head_agent():
    conn = synthetic(n_pn=80, n_kc=400, n_mbon=24, n_dan=20, n_glomeruli=20, seed=2)
    return FlyAgent(conn, 6, (5, 2), cfg=AgentConfig(seed=0, decision_steps=30, action_groups="random"))


def test_multi_head_groups_partition_mbons():
    agent = make_multi_head_agent()
    assert [g.shape[0] for g in agent.groups] == [5, 2]
    total = sum(g.sum(0) for g in agent.groups)
    assert torch.all(total == 1)  # every MBON in exactly one group of one head


def test_masked_options_never_chosen():
    agent = make_multi_head_agent()
    agent.cfg.epsilon = 0.5
    f = np.zeros((32, 6), dtype=np.float32)
    f[:, :2] = 1
    card_mask = np.zeros((32, 5), dtype=bool)
    card_mask[:, [0, 3]] = True
    a = agent.act(f, [card_mask, np.ones((32, 2), dtype=bool)], learn=False)
    assert a.shape == (32, 2) and set(a[:, 0]) <= {0, 3}


def test_teach_makes_brain_copy_teacher():
    agent = make_multi_head_agent()
    agent.cfg.epsilon = 0.0
    f = np.zeros((8, 6), dtype=np.float32)
    f[:, [1, 4]] = 1
    teacher = np.tile([2, 1], (8, 1))
    for _ in range(25):
        agent.act(f, learn=False)
        agent.teach(teacher)
    a = agent.act(f, learn=False)
    assert (a[:, 0] == 2).mean() >= 0.75


def test_state_dict_roundtrip():
    a1 = make_multi_head_agent()
    a1.net.W_kc_mbon.mul_(0.5)
    a2 = make_multi_head_agent()
    a2.load_state_dict(a1.state_dict())
    assert torch.equal(a1.net.W_kc_mbon, a2.net.W_kc_mbon)
