"""Export a trained Clash brain for the browser game (web/clash_core.js).

    python scripts/export_brain_js.py --brain runs/clash/brain.pt --out web/brain.json

Only what a decision needs is exported: PN, KC and MBON neurons, their
synapses (fixed and learned), channel->PN wiring and the MBON voting groups.
Arrays are base64 so the file stays small.
"""

import argparse
import base64
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore", message="Sparse")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from train_clash import make_agent  # noqa: E402


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def export(agent) -> dict:
    net = agent.net
    n_pn = net.sl["PN"].stop - net.sl["PN"].start
    n_kc = net.kc.stop - net.kc.start
    n_mb = net.mbon.stop - net.mbon.start
    assert net.sl["PN"].start == 0 and net.kc.start == n_pn and net.mbon.start == n_pn + n_kc
    n = n_pn + n_kc + n_mb

    post, pre = net._idx.numpy()
    w = net._val.numpy()
    keep = (pre < n) & (post < n)
    W = net.W_kc_mbon.cpu().numpy()
    m, k = np.nonzero(W)
    pre = np.concatenate([pre[keep], k + net.kc.start])
    post = np.concatenate([post[keep], m + net.mbon.start])
    w = np.concatenate([w[keep], W[m, k]]).astype(np.float32)
    order = np.lexsort((post, pre))
    pre, post, w = pre[order], post[order], w[order]
    ptr = np.zeros(n + 1, dtype=np.int32)
    np.add.at(ptr, pre + 1, 1)
    ptr = np.cumsum(ptr).astype(np.int32)

    chan = agent.channel_pn.cpu().numpy()[:, :n_pn] > 0
    chan_ptr = np.concatenate([[0], np.cumsum(chan.sum(1))]).tolist()
    chan_pn = np.concatenate([np.flatnonzero(row) for row in chan]).tolist()

    def group_of(g):
        g = g.cpu().numpy()
        return np.where(g.sum(0) > 0, g.argmax(0), -1).tolist(), g.sum(1).astype(int).tolist()

    card_group, card_n = group_of(agent.groups[0])
    lane_group, lane_n = group_of(agent.groups[1])
    cfg, ncfg = agent.cfg, net.cfg
    return dict(
        n_pn=int(n_pn), n_kc=int(n_kc), n_mbon=int(n_mb),
        alpha=net.alpha, beta=net.beta, refractory=ncfg.refractory, noise=ncfg.noise, apl_gain=ncfg.apl_gain,
        pn_current=cfg.pn_current, mbon_noise=cfg.mbon_noise, decision_steps=cfg.decision_steps,
        syn_ptr=b64(ptr), syn_post=b64(post.astype(np.uint16)), syn_w=b64(w),
        chan_ptr=chan_ptr, chan_pn=chan_pn,
        card_group=card_group, card_n=card_n, lane_group=lane_group, lane_n=lane_n,
    )


def load_agent(brain_path: str, device: str = "cpu"):
    ckpt = torch.load(brain_path, weights_only=False)
    args = argparse.Namespace(**ckpt["args"])
    args.device = device
    if "n_mbon" in ckpt["args"]:          # full card pool brain (train_royale.py)
        from train_royale import make_agent as make_royale_agent
        agent = make_royale_agent(args, probe=None)
        agent.load_state_dict(ckpt["agent"])
        return agent
    agent = make_agent(args, probe=None)
    agent.load_state_dict(ckpt["agent"])
    return agent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--brain", required=True)
    ap.add_argument("--out", default="web/brain.json")
    args = ap.parse_args()
    data = export(load_agent(args.brain))
    Path(args.out).write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {args.out} ({Path(args.out).stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
