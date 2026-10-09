"""Three-factor dopamine-gated plasticity on KC->MBON synapses.

Factor 1, presynaptic: which KCs fired during a decision.
Factor 2, action credit: which MBON group(s) that decision went to.
Factor 3, dopamine: reward and punishment DAN firing when the outcome arrives.

Factors 1 and 2 are stored in an eligibility trace that decays every
decision, so a reward at the end of an episode still reaches earlier choices.

Two modes:

* ``"depression"`` (fly-like). Dopamine only weakens KC->MBON synapses, as in
  the real mushroom body. Reward weakens the KC synapses onto the MBON
  groups that were *not* chosen; punishment weakens those onto the chosen
  group. Weights relax slowly back toward their initial values.
* ``"bidirectional"``. Reward strengthens the chosen group's synapses and
  punishment weakens them. Simpler, not what the fly does.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PlasticityConfig:
    mode: str = "depression"      # or "bidirectional"
    lr: float = 0.05              # fraction of a synapse's initial weight per unit dopamine x trace
    trace_decay: float = 0.8      # per decision
    recovery: float = 0.002       # per update, pull toward initial weight
    w_max: float = 3.0            # upper bound, in units of initial weight
    # "pre_action": trace = KC spikes x [MBON in chosen group]
    # "pre_post":   trace = KC spikes x MBON spikes (Hebbian tag)
    eligibility: str = "pre_action"
    # Synaptic scaling: after each update, rescale every MBON's KC inputs so
    # their total stays at its initial value. Keeps MBONs firing (and votes
    # above spike noise) while preserving which synapses were weakened.
    scaling: bool = False


class ThreeFactorRule:
    def __init__(self, w_init: torch.Tensor, mask: torch.Tensor, cfg: PlasticityConfig | None = None):
        self.cfg = cfg or PlasticityConfig()
        if self.cfg.mode not in ("depression", "bidirectional"):
            raise ValueError(f"unknown mode {self.cfg.mode!r}")
        if self.cfg.eligibility not in ("pre_action", "pre_post"):
            raise ValueError(f"unknown eligibility {self.cfg.eligibility!r}")
        self.w_init = w_init
        self.mask = mask
        self.chosen = None    # (batch, n_mbon, n_kc) trace for chosen group
        self.unchosen = None  # (batch, n_mbon, n_kc) trace for the other groups

    def reset(self, batch: int) -> None:
        """Clear eligibility traces (call at the start of each episode)."""
        shape = (batch, *self.w_init.shape)
        self.chosen = torch.zeros(shape, device=self.w_init.device, dtype=self.w_init.dtype)
        self.unchosen = torch.zeros_like(self.chosen)

    @torch.no_grad()
    def tag(self, kc: torch.Tensor, mbon: torch.Tensor, chosen: torch.Tensor,
            other: torch.Tensor) -> None:
        """Record one decision.

        kc: (batch, n_kc) KC spike counts. mbon: (batch, n_mbon) MBON spike
        counts. chosen / other: (batch, n_mbon) 0/1 masks of the MBONs in the
        chosen group and in the groups that were not chosen.
        """
        d = self.cfg.trace_decay
        self.chosen.mul_(d)
        self.unchosen.mul_(d)
        pre = (kc > 0).to(kc.dtype)[:, None, :]  # did the KC fire at all
        if self.cfg.eligibility == "pre_post":
            post = mbon / mbon.amax(1, keepdim=True).clamp_min(1)
            chosen, other = chosen * post, other * post
        self.chosen.add_(chosen[:, :, None] * pre)
        self.unchosen.add_(other[:, :, None] * pre)

    @torch.no_grad()
    def apply(self, w: torch.Tensor, da_reward: torch.Tensor, da_punish: torch.Tensor) -> torch.Tensor:
        """Update weights in place given per-env dopamine levels (batch,)."""
        c = self.cfg
        r = da_reward[:, None, None]
        p = da_punish[:, None, None]
        if c.mode == "bidirectional":
            dw = ((r - p) * self.chosen).mean(0)
        else:
            dw = -(r * self.unchosen + p * self.chosen).mean(0)
        w.add_(c.lr * dw * self.w_init)
        w.add_(c.recovery * (self.w_init - w))
        w.copy_(torch.minimum(w.clamp_min(0), c.w_max * self.w_init))
        w.mul_(self.mask)
        if c.scaling:
            total = w.sum(1, keepdim=True)
            w.mul_(self.w_init.sum(1, keepdim=True) / total.clamp_min(1e-12))
        return w
