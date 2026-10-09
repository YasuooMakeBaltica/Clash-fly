import torch

from flybrain.connectome import synthetic
from flybrain.network import MushroomBody, NetworkConfig, calibrate


def small_net(**cfg):
    return MushroomBody(synthetic(n_pn=60, n_kc=300, n_mbon=8, n_dan=10, seed=1), NetworkConfig(**cfg))


def test_silent_without_input():
    net = small_net(noise=0.0)
    net.reset(2)
    assert net.run(torch.zeros(2, net.n), 50).sum() == 0


def test_pn_input_drives_kcs_and_mbons():
    torch.manual_seed(0)
    net = small_net()
    i = torch.zeros(1, net.n)
    i[0, net.sl["PN"]] = 0.15
    net.reset(1)
    cnt = net.run(i, 60)
    assert cnt[0, net.sl["PN"]].mean() > 3
    assert (cnt[0, net.kc] > 0).any()
    assert cnt[0, net.mbon].sum() > 0


def test_kc_to_mbon_is_plastic_matrix():
    net = small_net()
    pops = net.conn.neurons["pop"].to_numpy()
    e = net.conn.edges
    n_plastic = ((pops[e["pre"]] == "KC") & (pops[e["post"]] == "MBON")).sum()
    assert int(net.mask_kc_mbon.sum()) == n_plastic
    W = net.W.to_dense()
    assert W[net.mbon, net.kc].abs().sum() == 0  # not duplicated in the fixed matrix


def test_calibrate_hits_targets():
    torch.manual_seed(0)
    net = small_net()
    probe = torch.zeros(8, net.n)
    pn = net.sl["PN"]
    for b in range(8):
        probe[b, pn.start + torch.randperm(pn.stop - pn.start)[:15]] = 0.15
    out = calibrate(net, probe, 40, kc_active_frac=0.1, mbon_rate_hz=20)
    assert 0.05 < out["kc_active_frac"] < 0.2
    assert 8 < out["mbon_rate_hz"] < 40
