"""Leaky integrate-and-fire simulation of the mushroom body in PyTorch.

All synapses come from the connectome. KC->MBON synapses are kept in a dense
plastic matrix (they are what learns); everything else is a fixed sparse
matrix. State has a batch dimension so several game instances can run in
parallel on one GPU.

Units are dimensionless: the threshold is 1, the reset is 0, and each step is
``dt`` ms.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import torch

from .connectome import POPULATIONS, Connectome


@dataclass
class NetworkConfig:
    dt: float = 1.0          # ms per step
    tau_m: float = 20.0      # membrane time constant (ms)
    tau_syn: float = 5.0     # synaptic current time constant (ms)
    refractory: int = 2      # steps
    noise: float = 0.02      # std of Gaussian current noise on every neuron
    # Total input strength into each population pair, after normalising so
    # each postsynaptic neuron's summed |weight| from that population is 1.
    # Pairs not listed get ``default_gain``.
    gains: dict[tuple[str, str], float] = field(default_factory=lambda: {
        ("PN", "KC"): 0.8,
        ("KC", "MBON"): 2.0,
        ("KC", "APL"): 0.5,
        ("APL", "KC"): 0.0,   # APL handled by ``apl_gain`` below by default
        ("KC", "KC"): 0.0,    # KC-KC (axo-axonal) synapses ignored
        ("MBON", "DAN"): 0.1,
    })
    default_gain: float = 0.1
    # Global feedback inhibition onto KCs proportional to the fraction of KCs
    # that spiked last step. A simple stand-in for the APL neuron that keeps
    # the KC code sparse.
    apl_gain: float = 4.0


class MushroomBody:
    def __init__(self, conn: Connectome, cfg: NetworkConfig | None = None,
                 device: str | torch.device = "cpu", dtype=torch.float32):
        self.conn = conn
        self.cfg = cfg or NetworkConfig()
        self.device = torch.device(device)
        self.dtype = dtype
        self.n = conn.n
        self.sl = {p: conn.slice(p) for p in POPULATIONS}
        self.kc, self.mbon = self.sl["KC"], self.sl["MBON"]

        e = conn.edges
        pops = conn.neurons["pop"].to_numpy()
        pre, post = e["pre"].to_numpy(), e["post"].to_numpy()
        w = (e["syn_count"].to_numpy() * e["sign"].to_numpy()).astype(np.float64)

        # Normalise each (pre_pop -> post_pop) block per postsynaptic neuron.
        pre_pop, post_pop = pops[pre], pops[post]
        gain = np.empty_like(w)
        for (a, b) in set(zip(pre_pop, post_pop)):
            m = (pre_pop == a) & (post_pop == b)
            tot = np.bincount(post[m], weights=np.abs(w[m]), minlength=self.n)
            g = self.cfg.gains.get((a, b), self.cfg.default_gain)
            gain[m] = g / np.maximum(tot[post[m]], 1e-9)
        w = w * gain

        plastic = (pre_pop == "KC") & (post_pop == "MBON")
        fixed = ~plastic & (w != 0)
        self._idx = torch.tensor(np.stack([post[fixed], pre[fixed]]), dtype=torch.long)
        self._val = torch.tensor(w[fixed], dtype=dtype)
        self._block = np.char.add(np.char.add(pre_pop[fixed].astype(str), ">"),
                                  post_pop[fixed].astype(str))
        self._build_sparse()

        n_mbon = self.mbon.stop - self.mbon.start
        n_kc = self.kc.stop - self.kc.start
        w_p = torch.zeros(n_mbon, n_kc, dtype=dtype)
        w_p[post[plastic] - self.mbon.start, pre[plastic] - self.kc.start] = torch.tensor(
            w[plastic], dtype=dtype)
        self.W_kc_mbon = w_p.to(self.device)          # learned
        self.W_kc_mbon_init = self.W_kc_mbon.clone()  # baseline for recovery
        self.mask_kc_mbon = (self.W_kc_mbon != 0).to(dtype)  # only real synapses

        self.alpha = math.exp(-self.cfg.dt / self.cfg.tau_m)
        self.beta = math.exp(-self.cfg.dt / self.cfg.tau_syn)
        self.reset(1)

    def _build_sparse(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # sparse CSR "beta" notices
            W = torch.sparse_coo_tensor(self._idx, self._val, (self.n, self.n)).coalesce()
            self.W = W.to_sparse_csr().to(self.device)

    def scale_block(self, pre: str, post: str, factor: float) -> None:
        """Multiply all weights from population ``pre`` to ``post`` by ``factor``."""
        if (pre, post) == ("KC", "MBON"):
            self.W_kc_mbon *= factor
            self.W_kc_mbon_init *= factor
            return
        m = torch.from_numpy(self._block == f"{pre}>{post}")
        self._val[m] *= factor
        self._build_sparse()

    # ------------------------------------------------------------------ state
    def reset(self, batch: int) -> None:
        z = lambda: torch.zeros(batch, self.n, device=self.device, dtype=self.dtype)
        self.v, self.i_syn, self.spikes = z(), z(), z()
        self.refr = torch.zeros(batch, self.n, device=self.device, dtype=torch.int32)

    @property
    def batch(self) -> int:
        return self.v.shape[0]

    # --------------------------------------------------------------- dynamics
    @torch.no_grad()
    def step(self, i_ext: torch.Tensor) -> torch.Tensor:
        """Advance one step. ``i_ext``: (batch, n) external current."""
        s = self.spikes
        rec = (self.W @ s.T).T
        rec[:, self.mbon] += s[:, self.kc] @ self.W_kc_mbon.T
        self.i_syn = self.beta * self.i_syn + rec
        if self.cfg.apl_gain:
            self.i_syn[:, self.kc] -= self.cfg.apl_gain * s[:, self.kc].mean(1, keepdim=True)

        i_tot = self.i_syn + i_ext
        if self.cfg.noise:
            i_tot = i_tot + self.cfg.noise * torch.randn_like(i_tot)
        active = self.refr <= 0
        self.v = torch.where(active, self.alpha * self.v + i_tot, self.v)
        spk = self.v >= 1.0
        self.v = torch.where(spk, torch.zeros_like(self.v), self.v)
        self.refr = torch.where(spk, torch.full_like(self.refr, self.cfg.refractory), self.refr - 1)
        self.spikes = spk.to(self.dtype)
        return self.spikes

    @torch.no_grad()
    def run(self, i_ext: torch.Tensor, steps: int) -> torch.Tensor:
        """Run ``steps`` steps with constant input; return spike counts (batch, n)."""
        counts = torch.zeros_like(self.v)
        for _ in range(steps):
            counts += self.step(i_ext)
        return counts

    def rates_hz(self, counts: torch.Tensor, steps: int) -> torch.Tensor:
        return counts * (1000.0 / (steps * self.cfg.dt))


@torch.no_grad()
def calibrate(net: MushroomBody, probe: torch.Tensor, steps: int,
              kc_active_frac: float = 0.1, mbon_rate_hz: float = 20.0,
              iters: int = 12, verbose: bool = False) -> dict[str, float]:
    """Rescale PN->KC and KC->MBON weights to hit target activity.

    ``probe`` is a (batch, n) external current representative of real inputs
    (e.g. encoded game states). PN->KC is tuned so ``kc_active_frac`` of KCs
    spike at least once in ``steps``; then KC->MBON so the mean MBON rate is
    ``mbon_rate_hz``. Uses bisection on a log scale.
    """
    pops = net.conn.neurons["pop"].to_numpy()

    def measure(seed=0):
        torch.manual_seed(seed)
        net.reset(probe.shape[0])
        cnt = net.run(probe, steps)
        kc = (cnt[:, net.kc] > 0).float().mean().item()
        mb = net.rates_hz(cnt[:, net.mbon], steps).mean().item()
        return kc, mb

    out = {}
    for pre, post, idx, target in (("PN", "KC", 0, kc_active_frac), ("KC", "MBON", 1, mbon_rate_hz)):
        if not ((pops == pre).any() and (pops == post).any()):
            continue
        lo, hi, total = 1 / 16, 16.0, 1.0
        for _ in range(iters):
            mid = math.sqrt(lo * hi)
            net.scale_block(pre, post, mid / total)
            total = mid
            val = measure()[idx]
            if verbose:
                print(f"  calibrate {pre}->{post} x{total:.3f}: {val:.3f} (target {target})")
            if val < target:
                lo = mid
            else:
                hi = mid
        out[f"{pre}->{post}"] = total
    kc, mb = measure(seed=1)
    out.update(kc_active_frac=kc, mbon_rate_hz=mb)
    return out
