"""A fly-brain agent: game state -> PNs -> KCs -> MBON groups vote on an action.

Reward is delivered by driving reward (PAM) or punishment (PPL1) dopamine
neurons; their simulated firing rates gate the KC->MBON plasticity.

An agent can have several action heads (e.g. which card, and which lane).
Each head gets its own share of the MBONs, split into one group per option;
every head votes independently from the same Kenyon cell code.
"""

from __future__ import annotations

from collections.abc import Sequence
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
    def __init__(self, conn: Connectome, n_channels: int, n_actions: int | Sequence[int] = 2,
                 net_cfg: NetworkConfig | None = None, plast_cfg: PlasticityConfig | None = None,
                 cfg: AgentConfig | None = None, device: str = "cpu",
                 probe: np.ndarray | None = None):
        """``n_actions``: options per head, e.g. 2, or (9, 2) for card x lane.
        ``probe``: example feature vectors to calibrate firing rates on
        (random sparse patterns if omitted)."""
        self.cfg = cfg or AgentConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        self.device = torch.device(device)
        self.net = MushroomBody(conn, net_cfg, device=device)
        self.n_channels = n_channels
        self.heads = [int(n_actions)] if np.isscalar(n_actions) else [int(n) for n in n_actions]
        nrn = conn.neurons

        self.channel_pn = self._assign_channels(nrn, n_channels)   # (C, N)
        self.groups = self._assign_groups(nrn, self.heads)          # per head: (options, n_mbon)
        dan = (nrn["pop"] == "DAN").to_numpy()
        self.reward_dans = torch.tensor(dan & (nrn["valence"] == "reward").to_numpy(), device=self.device)
        self.punish_dans = torch.tensor(dan & (nrn["valence"] == "punish").to_numpy(), device=self.device)
        self.mbon_noise_mask = torch.zeros(self.net.n, device=self.device)
        self.mbon_noise_mask[self.net.mbon] = 1

        self.expected_reward = 0.0
        self.calibration = None
        if self.cfg.calibrate:
            self.calibration = self.calibrate(probe)
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

    def _assign_groups(self, nrn, heads: list[int]) -> list[torch.Tensor]:
        """Split MBONs into one voting group per option of each head.

        With a single two-option head and ``action_groups="side"``, left
        hemisphere MBONs vote for option 0 and right ones for option 1.
        Otherwise MBONs are shared out at random, in proportion to each
        head's number of options.
        """
        mb = nrn.iloc[self.net.mbon]
        n = len(mb)
        if self.cfg.action_groups not in ("side", "random"):
            raise ValueError(f"unknown action_groups {self.cfg.action_groups!r}")
        if n < sum(heads):
            raise ValueError(f"{n} MBONs is too few for {sum(heads)} action options")
        if self.cfg.action_groups == "side" and heads == [2]:
            side = mb["side"].to_numpy()
            g = torch.zeros(2, n, device=self.device)
            g[0, torch.tensor(side == "left")] = 1
            g[1, torch.tensor(side == "right")] = 1
            return [g]
        perm = self.rng.permutation(n)
        share = np.floor(np.array(heads) / sum(heads) * n).astype(int)
        share[np.argmax(heads)] += n - share.sum()
        out, start = [], 0
        for k, m in zip(heads, share):
            g = torch.zeros(k, n, device=self.device)
            idx = perm[start:start + m]
            for a in range(k):
                g[a, torch.tensor(idx[a::k])] = 1
            out.append(g)
            start += m
        return out

    def encode(self, features: np.ndarray) -> torch.Tensor:
        f = torch.as_tensor(features, device=self.device, dtype=self.net.dtype)
        return self.cfg.pn_current * (f @ self.channel_pn).clamp(max=1)

    def calibrate(self, probe: np.ndarray | None = None, n_probe: int = 16, verbose: bool = False) -> dict:
        if probe is None:
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
    def act(self, features: np.ndarray, mask: Sequence[np.ndarray] | None = None,
            learn: bool = True) -> np.ndarray:
        """Run the network on the current state and return actions.

        ``mask``: per head, a (batch, options) boolean array of allowed
        options (e.g. cards you can afford). Returns (batch,) for a single
        head, else (batch, heads).
        """
        i_ext = self.encode(features)
        b = i_ext.shape[0]
        self.net.reset(b)
        counts = torch.zeros_like(i_ext)
        for _ in range(self.cfg.decision_steps):
            noise = self.cfg.mbon_noise * torch.randn_like(i_ext) * self.mbon_noise_mask
            counts += self.net.step(i_ext + noise)
        kc, mbon = counts[:, self.net.kc], counts[:, self.net.mbon]

        masks = self._masks(mask, b)
        actions = np.zeros((b, len(self.heads)), dtype=np.int64)
        all_votes = []
        for h, g in enumerate(self.groups):
            votes = (mbon @ g.T) / g.sum(1)                       # mean spikes per group
            votes = votes + 1e-3 * torch.rand_like(votes)         # break ties randomly
            votes = votes.masked_fill(~masks[h], -1.0)
            a = votes.argmax(1).cpu().numpy()
            explore = self.rng.random(b) < self.cfg.epsilon
            for i in np.flatnonzero(explore):
                a[i] = self.rng.choice(np.flatnonzero(masks[h][i].cpu().numpy()))
            actions[:, h] = a
            all_votes.append(votes)
        self.last = dict(kc=kc, mbon=mbon, votes=all_votes, masks=masks)
        if learn:
            self._tag(actions)
        return actions[:, 0] if len(self.heads) == 1 else actions

    def _masks(self, mask, b: int) -> list[torch.Tensor]:
        if mask is None:
            return [torch.ones(b, k, dtype=torch.bool, device=self.device) for k in self.heads]
        return [torch.as_tensor(np.asarray(m), dtype=torch.bool, device=self.device) for m in mask]

    def _tag(self, actions: np.ndarray, active: np.ndarray | None = None) -> None:
        """Mark the chosen groups (and the allowed alternatives) in the eligibility trace.

        ``active``: optional (batch, heads) boolean; heads marked False are
        left out (e.g. the lane when the action is "wait").
        """
        actions = np.asarray(actions).reshape(len(actions), -1)
        chosen = torch.zeros_like(self.last["mbon"])
        other = torch.zeros_like(chosen)
        for h, g in enumerate(self.groups):
            on = 1.0 if active is None else torch.as_tensor(
                np.asarray(active)[:, h], device=self.device, dtype=g.dtype)[:, None]
            c = g[torch.as_tensor(actions[:, h], device=self.device)]
            chosen += on * c
            other += on * (self.last["masks"][h].to(g.dtype) @ g - c)
        self.rule.tag(self.last["kc"], self.last["mbon"], chosen, other)

    def teach(self, teacher_actions: np.ndarray, active: np.ndarray | None = None,
              strength: float = 1.0) -> None:
        """Learn to copy a teacher on the state from the last :meth:`act` call.

        Like a fly pairing an odour with sugar: the teacher's choice is tagged
        and immediately followed by reward dopamine. Clears eligibility traces.
        """
        b = len(teacher_actions)
        self.rule.reset(b)
        self._tag(teacher_actions, active)
        da_r, da_p = self.dopamine(np.full(b, strength, dtype=np.float32))
        self.rule.apply(self.net.W_kc_mbon, da_r, da_p)

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

    # ------------------------------------------------------------ save/load
    def state_dict(self) -> dict:
        """Everything learned or calibrated (the wiring comes from the connectome)."""
        return dict(
            W_kc_mbon=self.net.W_kc_mbon.cpu(), W_kc_mbon_init=self.net.W_kc_mbon_init.cpu(),
            fixed_values=self.net._val.cpu(), channel_pn=self.channel_pn.cpu(),
            groups=[g.cpu() for g in self.groups], heads=self.heads,
            expected_reward=self.expected_reward, calibration=self.calibration,
        )

    def load_state_dict(self, state: dict) -> None:
        if list(state["heads"]) != self.heads:
            raise ValueError(f"checkpoint heads {state['heads']} != agent heads {self.heads}")
        self.net.W_kc_mbon.copy_(state["W_kc_mbon"].to(self.device))
        self.net.W_kc_mbon_init.copy_(state["W_kc_mbon_init"].to(self.device))
        self.net._val.copy_(state["fixed_values"])
        self.net._build_sparse()
        self.channel_pn = state["channel_pn"].to(self.device)
        self.groups = [g.to(self.device) for g in state["groups"]]
        self.expected_reward = state["expected_reward"]
        self.calibration = state["calibration"]
