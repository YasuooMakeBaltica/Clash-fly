import pytest
import torch

from flybrain.plasticity import PlasticityConfig, ThreeFactorRule


def setup(mode, **kw):
    w0 = torch.ones(4, 3)          # 4 MBONs x 3 KCs
    w0[3, 2] = 0                   # one missing synapse
    mask = (w0 > 0).float()
    rule = ThreeFactorRule(w0.clone(), mask, PlasticityConfig(mode=mode, recovery=0.0, **kw))
    rule.reset(1)
    kc = torch.tensor([[2.0, 0.0, 1.0]])      # KCs 0 and 2 fired
    mbon = torch.ones(1, 4)
    chosen = torch.tensor([[1.0, 1.0, 0.0, 0.0]])   # MBONs 0,1 = chosen group
    other = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    rule.tag(kc, mbon, chosen, other)
    return rule, w0.clone()


def test_depression_reward_weakens_unchosen_only():
    rule, w = setup("depression")
    rule.apply(w, torch.tensor([1.0]), torch.tensor([0.0]))
    assert torch.all(w[:2] == 1)                       # chosen group untouched
    assert w[2, 0] < 1 and w[2, 2] < 1                 # active KCs onto other group weakened
    assert w[2, 1] == 1                                # silent KC untouched
    assert w[3, 2] == 0                                # missing synapse stays missing


def test_depression_punish_weakens_chosen():
    rule, w = setup("depression")
    rule.apply(w, torch.tensor([0.0]), torch.tensor([1.0]))
    assert w[0, 0] < 1 and torch.all(w[2:, :2] == 1)


def test_bidirectional_reward_strengthens_chosen():
    rule, w = setup("bidirectional")
    rule.apply(w, torch.tensor([1.0]), torch.tensor([0.0]))
    assert w[0, 0] > 1 and w[2, 0] == 1


def test_trace_decays_across_decisions():
    rule, _ = setup("depression", trace_decay=0.5)
    rule.tag(torch.zeros(1, 3), torch.zeros(1, 4), torch.zeros(1, 4), torch.zeros(1, 4))
    assert rule.chosen[0, 0, 0] == pytest.approx(0.5)


def test_weights_stay_bounded():
    rule, w = setup("bidirectional", lr=100.0, w_max=2.0)
    rule.apply(w, torch.tensor([1.0]), torch.tensor([0.0]))
    assert w.max() <= 2.0
    rule.apply(w, torch.tensor([0.0]), torch.tensor([50.0]))
    assert w.min() >= 0.0
