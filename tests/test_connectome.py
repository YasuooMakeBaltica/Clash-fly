import numpy as np
import pandas as pd

from flybrain.connectome import Connectome, load_codex, synthetic


def write_codex(d):
    """A tiny dataset in FlyWire Codex export format."""
    cls = pd.DataFrame({
        "root_id": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "super_class": ["central"] * 9,
        "class": ["ALPN", "Kenyon_Cell", "Kenyon_Cell", "MBON", "MBON", "DAN", "DAN", "", "LHLN"],
        "cell_type": ["DA1_lPN", "KCg-m", "KCab", "MBON01", "MBON02", "PAM01", "PPL101", "APL", "LHLN1"],
        "side": ["left", "left", "right", "left", "right", "left", "right", "left", "left"],
    })
    cls.to_csv(d / "classification.csv.gz", index=False)
    pd.DataFrame({
        "root_id": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "nt_type": ["ACH", "ACH", "ACH", "GLUT", "ACH", "DA", "DA", "GABA", "GABA"],
    }).to_csv(d / "neurons.csv.gz", index=False)
    pd.DataFrame({
        "pre_root_id":  [1, 1, 1, 2, 3, 4, 6, 8, 9, 2],
        "post_root_id": [2, 2, 3, 4, 5, 6, 2, 2, 2, 5],
        "neuropil":     ["CA_L", "CA_R", "CA_L", "MB_ML_L", "MB_ML_R", "SMP_L", "MB_ML_L", "MB_CA_L", "LH_L", "MB_ML_R"],
        "syn_count":    [4, 6, 8, 20, 30, 7, 9, 12, 50, 3],
        "nt_type":      ["ACH"] * 10,
    }).to_csv(d / "connections.csv.gz", index=False)


def test_load_codex_extracts_populations(tmp_path):
    write_codex(tmp_path)
    c = load_codex(tmp_path, min_syn=5)
    n = c.neurons
    assert list(n["pop"]) == ["PN", "KC", "KC", "MBON", "MBON", "DAN", "DAN", "APL"]
    assert 9 not in n["root_id"].values  # LHLN is not part of the subcircuit
    assert list(n.loc[n["pop"] == "DAN", "valence"]) == ["reward", "punish"]
    assert c.slice("KC") == slice(1, 3)

    rid = n["root_id"].to_numpy()
    e = c.edges.assign(pre=rid[c.edges["pre"]], post=rid[c.edges["post"]]).set_index(["pre", "post"])
    assert e.loc[(1, 2), "syn_count"] == 10      # split across neuropils, summed
    assert (2, 5) not in e.index                 # below min_syn
    assert e.loc[(4, 6), "sign"] == -1.0         # glutamate is inhibitory
    assert e.loc[(6, 2), "sign"] == 0.0          # dopamine is modulatory
    assert e.loc[(8, 2), "sign"] == -1.0         # APL is GABAergic


def test_save_load_roundtrip(tmp_path):
    c = synthetic(n_pn=40, n_kc=100, n_mbon=8, n_dan=10)
    c.save(tmp_path / "mb.npz")
    d = Connectome.load(tmp_path / "mb.npz")
    pd.testing.assert_frame_equal(c.edges, d.edges, check_dtype=False)
    assert list(c.neurons["pop"]) == list(d.neurons["pop"])


def test_synthetic_structure():
    c = synthetic(n_pn=40, n_kc=100, n_mbon=8, n_dan=10, pn_per_kc=6)
    pops = c.neurons["pop"].to_numpy()
    sides = c.neurons["side"].to_numpy()
    pre, post = c.edges["pre"].to_numpy(), c.edges["post"].to_numpy()
    pn_kc = (pops[pre] == "PN") & (pops[post] == "KC")
    assert np.all(np.bincount(post[pn_kc], minlength=c.n)[c.slice("KC")] == 6)
    assert np.all(sides[pre] == sides[post])  # ipsilateral only
