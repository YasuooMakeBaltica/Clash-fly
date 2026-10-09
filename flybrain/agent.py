"""A fly-brain agent: game state -> PNs -> KCs -> MBON groups vote on an action.

Reward is delivered by driving reward (PAM) or punishment (PPL1) dopamine
neurons; their simulated firing rates gate the KC->MBON plasticity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .connectome import Connectome
from .network import MushroomBody, NetworkConfig, calibrate
from .plasticity import PlasticityConfig, ThreeFactorRule


@dataclass
class AgentConfig:
    decision_steps: int = 40      # ms of simulation per decision
    dopamine_steps: int = 30      # ms of simulation to deliver reward
    pn_current: float = 0.15      # current into PNs of an active channel (~95 Hz)
    dan_current: float = 0.3      # current into DANs per unit |reward|
    dan_ref_hz: float = 100.0     # DAN rate that counts as dopamine level 1
    mbon_noise: float = 0.05      # extra current noise on MBONs (exploration)
    epsilon: float = 0.05         # chance of a random action
    action_groups: str = "side"   # "side" (left/right hemisphere) or "random"
    # Reward prediction error: dopamine reports reward minus a running average
    # of past rewards (the fly gets this from MBON->DAN feedback). This is the
    # averaging rate; 0 disables it and dopamine reports raw reward.
    rpe_rate: float = 0.0
    calibrate: bool = True
    kc_active_frac: float = 0.1
    mbon_rate_hz: float = 20.0
    seed: int = 0


class FlyAgent:
    def __init__(self, conn: Connectome, n_channels: int, n_actions: int = 2,
                 net_cfg: NetworkConfig | None = None, plast_cfg: PlasticityConfig | None = None,
                 cfg: AgentConfig | None = None, device: str = "cpu"):
        self.cfg = cfg or AgentConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        self.device = torch.device(device)
        self.net = MushroomBody(conn, net_cfg, device=device)
        self.n_channels, self.n_actions = n_channels, n_actions
        nrn = conn.neurons

        self.channel_pn = self._assign_channels(nrn, n_channels)   # (C, N)
        self.groups = self._assign_groups(nrn, n_actions)           # (A, n_mbon)
        dan = (nrn["pop"] == "DAN").to_numpy()
        self.reward_dans = torch.tensor(dan & (nrn["valence"] == "reward").to_numpy(), device=self.device)
        self.punish_dans = torch.tensor(dan & (nrn["valence"] == "punish").to_numpy(), device=self.device)
        self.mbon_noise_mask = torch.zeros(self.net.n, device=self.device)
        self.mbon_noise_mask[self.net.mbon] = 1

        self.expected_reward = 0.0
        self.calibration = None
        if self.cfg.calibrate:
            self.calibration = self.calibrate()
        self.rule = ThreeFactorRule(self.net.W_kc_mbon_init, self.net.mask_kc_mbon, plast_cfg)

    # ---------------------------------------------------------------- wiring
    def _assign_channels(self, nrn, n_channels: int) -> torch.Tensor:
        """Map each input channel to a set of PN cell types (glomeruli).

        PNs of the same type exist in both hemispheres, so every channel
        drives both sides of the brain, like an odour would.
        """
        pn = np.flatnonzero((nrn["pop"] == "PN").to_numpy())
        types = nrn["cell_type"].to_numpy()[pn]
        uniq = np.unique(types)
        if len(uniq) < n_channels or (len(uniq) == 1 and uniq[0] == ""):
            types = self.rng.integers(0, n_channels, size=len(pn)).astype(str)
            uniq = np.unique(types)
        uniq = self.rng.permutation(uniq)
        channel_of_type = {t: i % n_channels for i, t in enumerate(uniq)}
        m = torch.zeros(n_channels, self.net.n, device=self.device)
        for idx, t in zip(pn, types):
            m[channel_of_type[t], idx] = 1
        return m

    def _assign_groups(self, nrn, n_actions: int) -> torch.Tensor:
        """Split MBONs into one voting group per action."""
        mb = nrn.iloc[self.net.mbon]
        g = torch.zeros(n_actions, len(mb), device=self.device)
        if self.cfg.action_groups == "side" and n_actions == 2:
            side = mb["side"].to_numpy()
            g[0, torch.tensor(side == "left")] = 1
            g[1, torch.tensor(side == "right")] = 1
        elif self.cfg.action_groups in ("side", "random"):
            perm = self.rng.permutation(len(mb))
            for a in range(n_actions):
                g[a, torch.tensor(perm[a::n_actions])] = 1
        else:
            raise ValueError(f"unknown action_groups {self.cfg.action_groups!r}")
        return g

    def encode(self, features: np.ndarray) -> torch.Tensor:
        f = torch.as_tensor(features, device=self.device, dtype=self.net.dtype)
        return self.cfg.pn_current * (f @ self.channel_pn).clamp(max=1)

    def calibrate(self, n_probe: int = 16, verbose: bool = False) -> dict:
        probe = np.zeros((n_probe, self.n_channels), dtype=np.float32)
        k = max(1, self.n_channels // 8)  # a sparse random pattern per probe
        for row in probe:
            row[self.rng.choice(self.n_channels, size=k, replace=False)] = 1
        return calibrate(self.net, self.encode(probe), self.cfg.decision_steps,
                         kc_active_frac=self.cfg.kc_active_frac,
                         mbon_rate_hz=self.cfg.mbon_rate_hz, verbose=verbose)

    # ------------------------------------------------------------- behaviour
    def begin_episode(self, batch: int) -> None:
        self.rule.reset(batch)

    @torch.no_grad()
    def act(self, features: np.ndarray, learn: bool = True) -> np.ndarray:
        """Run the network on the current state and return one action per game."""
        i_ext = self.encode(features)
        b = i_ext.shape[0]
        self.net.reset(b)
        counts = torch.zeros_like(i_ext)
        for _ in range(self.cfg.decision_steps):
            noise = self.cfg.mbon_noise * torch.randn_like(i_ext) * self.mbon_noise_mask
            counts += self.net.step(i_ext + noise)
        kc, mbon = counts[:, self.net.kc], counts[:, self.net.mbon]
        votes = (mbon @ self.groups.T) / self.groups.sum(1)          # mean count per group
        votes = votes + 1e-3 * torch.rand_like(votes)                  # break ties randomly
        action = votes.argmax(1).cpu().numpy()
        explore = self.rng.random(b) < self.cfg.epsilon
        action[explore] = self.rng.integers(0, self.n_actions, size=explore.sum())
        self.last = dict(kc=kc, mbon=mbon, votes=votes)
        if learn:
            a = torch.as_tensor(action, device=self.device)
            chosen = self.groups[a]
            other = self.groups.sum(0, keepdim=True) - chosen
            self.rule.tag(kc, mbon, chosen, other)
        return action

    @torch.no_grad()
    def dopamine(self, reward: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Drive reward/punish DANs and return their dopamine levels (batch,)."""
        r = torch.as_tensor(reward, device=self.device, dtype=self.net.dtype)
        if not (self.reward_dans.any() and self.punish_dans.any()):
            return r.clamp_min(0), (-r).clamp_min(0)  # no DANs: use reward directly
        i_ext = torch.zeros(len(r), self.net.n, device=self.device, dtype=self.net.dtype)
        i_ext += self.cfg.dan_current * r.clamp_min(0)[:, None] * self.reward_dans
        i_ext += self.cfg.dan_current * (-r).clamp_min(0)[:, None] * self.punish_dans
        self.net.reset(len(r))
        counts = self.net.run(i_ext, self.cfg.dopamine_steps)
        hz = self.net.rates_hz(counts, self.cfg.dopamine_steps) / self.cfg.dan_ref_hz
        return hz[:, self.reward_dans].mean(1), hz[:, self.punish_dans].mean(1)

    def reward(self, reward: np.ndarray) -> None:
        if not np.any(reward):
            return
        if self.cfg.rpe_rate:
            surprise = reward - self.expected_reward
            self.expected_reward += self.cfg.rpe_rate * float(np.mean(reward) - self.expected_reward)
            reward = surprise
        da_r, da_p = self.dopamine(reward)
        self.rule.apply(self.net.W_kc_mbon, da_r, da_p)
