"""Load the mushroom-body subcircuit of the FlyWire connectome.

Two sources:

* ``load_codex(data_dir)`` reads the CSV exports from FlyWire Codex
  (https://codex.flywire.ai/api/download). Needed files:
    - ``classification.csv(.gz)``: root_id, super_class, class, sub_class,
      cell_type, side, ...
    - ``connections.csv(.gz)``: pre_root_id, post_root_id, neuropil,
      syn_count, nt_type
    - ``neurons.csv(.gz)`` (optional): root_id, nt_type, ... gives each
      neuron's predicted neurotransmitter.
* ``synthetic(...)`` builds a random circuit with roughly the same statistics,
  so everything can be developed and tested without the real data.

Both return a :class:`Connectome` whose neurons are sorted by population
(PN, KC, MBON, DAN, APL) so each population is a contiguous index range.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

POPULATIONS = ("PN", "KC", "MBON", "DAN", "APL")

# How each population is recognised in the Codex classification table.
# Each rule is (column, regex); a neuron matches if any rule matches.
DEFAULT_RULES: dict[str, list[tuple[str, str]]] = {
    "PN": [("class", r"^ALPN$")],
    "KC": [("class", r"^Kenyon_Cell$"), ("cell_type", r"^KC")],
    "MBON": [("class", r"^MBON$"), ("cell_type", r"^MBON")],
    "DAN": [("class", r"^DAN$"), ("cell_type", r"^(?:PAM|PPL1|PPL2)")],
    "APL": [("cell_type", r"^APL$")],
}

# Dopamine neuron cell types that signal reward vs punishment.
REWARD_DAN = r"^PAM"
PUNISH_DAN = r"^PPL1"

# Sign of each predicted neurotransmitter. In the fly, glutamate is mostly
# inhibitory (GluCl). Modulators (DA, SER, OCT) carry no fast current here.
NT_SIGN = {"ACH": 1.0, "GABA": -1.0, "GLUT": -1.0, "DA": 0.0, "SER": 0.0, "OCT": 0.0}


@dataclass
class Connectome:
    """Neurons and synapses of the extracted subcircuit.

    neurons: one row per neuron, index == neuron index. Columns: root_id, pop,
        side ("left"/"right"/other), cell_type, nt_type, valence (DANs only:
        "reward", "punish" or "").
    edges: pre, post (neuron indices), syn_count, sign.
    """

    neurons: pd.DataFrame
    edges: pd.DataFrame

    def slice(self, pop: str) -> slice:
        idx = np.flatnonzero(self.neurons["pop"].to_numpy() == pop)
        if len(idx) == 0:
            return slice(0, 0)
        return slice(int(idx[0]), int(idx[-1]) + 1)

    def count(self, pop: str) -> int:
        s = self.slice(pop)
        return s.stop - s.start

    @property
    def n(self) -> int:
        return len(self.neurons)

    def summary(self) -> str:
        lines = [f"{self.n} neurons, {len(self.edges)} connections"]
        for pop in POPULATIONS:
            sub = self.neurons[self.neurons["pop"] == pop]
            sides = sub["side"].value_counts().to_dict()
            extra = ""
            if pop == "DAN":
                extra = f", valence {sub['valence'].value_counts().to_dict()}"
            lines.append(f"  {pop:5s} {len(sub):6d}  sides {sides}{extra}")
        pops = self.neurons["pop"].to_numpy()
        pairs = (
            pd.DataFrame({"pre": pops[self.edges["pre"]], "post": pops[self.edges["post"]],
                          "syn": self.edges["syn_count"]})
            .groupby(["pre", "post"])["syn"].agg(["count", "sum"])
        )
        lines.append("  connections by population (count, synapses):")
        for (pre, post), row in pairs.iterrows():
            lines.append(f"    {pre:>5s} -> {post:<5s} {row['count']:8d} {row['sum']:9d}")
        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            **{f"n_{c}": self.neurons[c].to_numpy() for c in self.neurons.columns},
            **{f"e_{c}": self.edges[c].to_numpy() for c in self.edges.columns},
        )

    @classmethod
    def load(cls, path: str | Path) -> "Connectome":
        z = np.load(path, allow_pickle=True)
        neurons = pd.DataFrame({k[2:]: z[k] for k in z.files if k.startswith("n_")})
        edges = pd.DataFrame({k[2:]: z[k] for k in z.files if k.startswith("e_")})
        return cls(neurons=neurons, edges=edges)


def _find(data_dir: Path, stem: str) -> Path | None:
    for name in (f"{stem}.csv.gz", f"{stem}.csv"):
        if (data_dir / name).exists():
            return data_dir / name
    return None


def _classify(df: pd.DataFrame, rules: dict[str, list[tuple[str, str]]]) -> pd.Series:
    pop = pd.Series("", index=df.index, dtype=object)
    # Later populations win ties (e.g. APL is checked last).
    for name, name_rules in rules.items():
        hit = pd.Series(False, index=df.index)
        for col, pattern in name_rules:
            if col in df.columns:
                hit |= df[col].astype(str).str.contains(pattern, regex=True, na=False)
        pop[hit] = name
    return pop


def _finish(neurons: pd.DataFrame, edges: pd.DataFrame) -> Connectome:
    """Sort neurons by population, re-index edges, and add valence/sign."""
    order = {p: i for i, p in enumerate(POPULATIONS)}
    neurons = neurons.copy()
    neurons["_o"] = neurons["pop"].map(order)
    neurons = neurons.sort_values(["_o", "side", "cell_type", "root_id"], kind="stable")
    neurons = neurons.drop(columns="_o").reset_index(drop=True)
    for col in ("side", "cell_type", "nt_type"):
        neurons[col] = neurons[col].fillna("").astype(str)

    ct = neurons["cell_type"]
    valence = np.where(ct.str.contains(REWARD_DAN), "reward",
                       np.where(ct.str.contains(PUNISH_DAN), "punish", ""))
    neurons["valence"] = np.where(neurons["pop"] == "DAN", valence, "")

    index_of = pd.Series(np.arange(len(neurons)), index=neurons["root_id"].to_numpy())
    edges = edges.copy()
    edges["pre"] = index_of.reindex(edges["pre"].to_numpy()).to_numpy()
    edges["post"] = index_of.reindex(edges["post"].to_numpy()).to_numpy()
    edges = edges.dropna(subset=["pre", "post"])
    edges = edges.astype({"pre": np.int64, "post": np.int64, "syn_count": np.int64})
    # Merge connections split across neuropils into one edge per pair.
    edges = edges.groupby(["pre", "post"], as_index=False)["syn_count"].sum()
    nt = neurons["nt_type"].str.upper().to_numpy()[edges["pre"].to_numpy()]
    edges["sign"] = [NT_SIGN.get(t, 1.0) for t in nt]
    return Connectome(neurons=neurons, edges=edges.reset_index(drop=True))


def load_codex(
    data_dir: str | Path,
    rules: dict[str, list[tuple[str, str]]] | None = None,
    min_syn: int = 5,
    chunksize: int = 2_000_000,
) -> Connectome:
    """Extract PN/KC/MBON/DAN/APL neurons and their connections from Codex CSVs.

    ``min_syn`` drops connections with fewer synapses (FlyWire's usual
    threshold is 5).
    """
    data_dir = Path(data_dir)
    rules = rules or DEFAULT_RULES
    cls_path = _find(data_dir, "classification")
    con_path = _find(data_dir, "connections") or _find(data_dir, "connections_no_threshold")
    if cls_path is None or con_path is None:
        raise FileNotFoundError(
            f"Need classification.csv(.gz) and connections.csv(.gz) in {data_dir}. "
            "Download them from https://codex.flywire.ai/api/download"
        )

    cls = pd.read_csv(cls_path, dtype={"root_id": np.int64})
    cls["pop"] = _classify(cls, rules)
    neurons = cls[cls["pop"] != ""].copy()
    if "side" not in neurons.columns:
        neurons["side"] = ""
    if "cell_type" not in neurons.columns:
        neurons["cell_type"] = ""

    # Neurotransmitter per neuron: neurons.csv if present, else the most
    # common nt_type on that neuron's outgoing connections (filled below).
    nrn_path = _find(data_dir, "neurons")
    if nrn_path is not None:
        nt = pd.read_csv(nrn_path, usecols=["root_id", "nt_type"], dtype={"root_id": np.int64})
        neurons = neurons.merge(nt, on="root_id", how="left")
    else:
        neurons["nt_type"] = np.nan

    keep = set(neurons["root_id"].tolist())
    parts = []
    head = pd.read_csv(con_path, nrows=0).columns
    usecols = [c for c in ("pre_root_id", "post_root_id", "syn_count", "nt_type") if c in head]
    for chunk in pd.read_csv(con_path, usecols=usecols, chunksize=chunksize,
                             dtype={"pre_root_id": np.int64, "post_root_id": np.int64}):
        m = chunk["pre_root_id"].isin(keep) & chunk["post_root_id"].isin(keep)
        parts.append(chunk[m])
    con = pd.concat(parts, ignore_index=True)

    if neurons["nt_type"].isna().any() and "nt_type" in con.columns:
        top = con.groupby("pre_root_id")["nt_type"].agg(lambda s: s.mode().iat[0])
        missing = neurons["nt_type"].isna()
        neurons.loc[missing, "nt_type"] = neurons.loc[missing, "root_id"].map(top)

    con = con.groupby(["pre_root_id", "post_root_id"], as_index=False)["syn_count"].sum()
    con = con[con["syn_count"] >= min_syn]
    edges = con.rename(columns={"pre_root_id": "pre", "post_root_id": "post"})
    cols = ["root_id", "pop", "side", "cell_type", "nt_type"]
    return _finish(neurons[cols], edges)


def synthetic(
    n_pn: int = 340,
    n_kc: int = 2000,
    n_mbon: int = 48,
    n_dan: int = 120,
    pn_per_kc: int = 6,
    kc_to_mbon_p: float = 0.3,
    n_glomeruli: int = 50,
    seed: int = 0,
) -> Connectome:
    """A random mushroom body with FlyWire-like structure.

    Both hemispheres are built; connections are ipsilateral. PNs come in
    left/right pairs that share a glomerulus cell type, like the real data.
    MBON->DAN feedback and APL feedback inhibition are included.
    """
    rng = np.random.default_rng(seed)
    rows = []
    rid = iter(range(10**6, 10**8))

    def add(pop, n, cell_type, nt, side):
        for i in range(n):
            ct = cell_type(i) if callable(cell_type) else cell_type
            rows.append((next(rid), pop, side, ct, nt))

    for side in ("left", "right"):
        add("PN", n_pn // 2, lambda i: f"PN_G{i % n_glomeruli}", "ACH", side)
        add("KC", n_kc // 2, lambda i: ("KCg", "KCab", "KCapbp")[i % 3], "ACH", side)
        add("MBON", n_mbon // 2, lambda i: f"MBON{i:02d}", "ACH", side)  # nt reassigned below
        n_pam = int(n_dan // 2 * 0.9)
        add("DAN", n_pam, lambda i: f"PAM{i % 15:02d}", "DA", side)
        add("DAN", n_dan // 2 - n_pam, lambda i: f"PPL1{i % 6:02d}", "DA", side)
        add("APL", 1, "APL", "GABA", side)

    neurons = pd.DataFrame(rows, columns=["root_id", "pop", "side", "cell_type", "nt_type"])
    # MBONs: mostly cholinergic, some GABA and glutamate, as in the fly.
    mb = neurons["pop"] == "MBON"
    neurons.loc[mb, "nt_type"] = rng.choice(["ACH", "GABA", "GLUT"], size=mb.sum(), p=[0.5, 0.2, 0.3])

    edges = []
    for side in ("left", "right"):
        on = neurons["side"] == side
        ids = {p: neurons.loc[on & (neurons["pop"] == p), "root_id"].to_numpy() for p in POPULATIONS}
        for kc in ids["KC"]:  # PN -> KC, ~6 random claws
            for pn in rng.choice(ids["PN"], size=pn_per_kc, replace=False):
                edges.append((pn, kc, int(rng.integers(5, 30))))
        for mbon in ids["MBON"]:  # KC -> MBON
            kcs = ids["KC"][rng.random(len(ids["KC"])) < kc_to_mbon_p]
            edges += [(kc, mbon, int(rng.integers(5, 20))) for kc in kcs]
        for kc in ids["KC"]:  # KC <-> APL
            edges.append((kc, ids["APL"][0], int(rng.integers(5, 15))))
            edges.append((ids["APL"][0], kc, int(rng.integers(5, 15))))
        for dan in ids["DAN"]:  # MBON -> DAN feedback, DAN -> KC (modulatory)
            for mbon in rng.choice(ids["MBON"], size=3, replace=False):
                edges.append((mbon, dan, int(rng.integers(5, 15))))
            for kc in rng.choice(ids["KC"], size=20, replace=False):
                edges.append((dan, kc, int(rng.integers(5, 10))))
    edges = pd.DataFrame(edges, columns=["pre", "post", "syn_count"])
    return _finish(neurons, edges)


def load(source: str | Path, **kwargs) -> Connectome:
    """``"synthetic"``, a saved ``.npz``, or a directory of Codex CSVs."""
    if str(source) == "synthetic":
        return synthetic(**kwargs)
    path = Path(source)
    if path.suffix == ".npz":
        return Connectome.load(path)
    return load_codex(path, **kwargs)
