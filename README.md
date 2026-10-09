# Clash-fly

Why not let a fly play Clash Royale, what could go wrong.

An agent whose "brain" is the fruit-fly mushroom body, wired from the FlyWire
connectome and simulated as spiking neurons in PyTorch. It learns from reward
through dopamine-gated plasticity, the way flies learn which odours are good
or bad. It is tested on a toy game first; Clash Royale comes later.

## How it works

```
game state ──> PNs ──> KCs ──> MBONs ──> vote: left / right
 (channels)   (input)  (sparse  (output     (most spikes wins)
                        code)    groups)
                          │        ▲
                          └─ KC→MBON synapses: the only plastic ones
                                   ▲
              reward ──> DANs ─────┘  dopamine × eligibility trace
```

* **Wiring is from the connectome, strengths are not.** FlyWire gives who
  connects to whom, synapse counts and predicted neurotransmitters (which set
  each synapse's sign). Weights are synapse counts normalised per neuron, then
  auto-calibrated so about 10% of KCs fire and MBONs fire at about 20 Hz.
* **Input:** each game feature (e.g. "ball in column 3") is a channel driving
  a set of PN glomerulus types in both hemispheres, like an odour.
* **Output:** MBONs are split into one group per action. By default, left
  hemisphere MBONs vote "left" and right hemisphere MBONs vote "right".
* **Learning (three-factor rule):** KCs that fired during a decision leave an
  eligibility trace on their synapses to the MBON group that was chosen (and
  on those to the groups that weren't). The trace decays each decision. When
  reward or punishment arrives, reward DANs (PAM) or punishment DANs (PPL1)
  are driven in the simulation and their firing rate is the dopamine signal.
  * `depression` mode (default, fly-like): dopamine only weakens synapses.
    Reward weakens KC synapses onto the actions *not* taken; punishment
    weakens those onto the action taken. Weights relax back toward baseline.
  * `bidirectional` mode: reward strengthens the chosen action, punishment
    weakens it.

## Setup

```bash
pip install -r requirements.txt   # pick the CUDA torch build for GPU
pytest                            # ~1 min on CPU
```

### FlyWire data

Everything runs on a **synthetic** mushroom body until you have the real data.
To use FlyWire:

1. Log in at https://codex.flywire.ai/api/download (free account; accept the
   data terms).
2. Download `classification.csv.gz`, `connections.csv.gz` and
   `neurons.csv.gz` into `data/flywire/`.
3. Extract the mushroom body once (reads the big connections file in chunks):

```bash
python scripts/extract_populations.py --data data/flywire --out data/mushroom_body.npz
```

Then pass `--data data/mushroom_body.npz` to the other scripts. If a
population comes out empty or wrong, check the column values against
`DEFAULT_RULES` in `flybrain/connectome.py`.

## Milestones

| # | Milestone | Command |
|---|---|---|
| 1 | Load connectome, extract PN/KC/MBON/DAN/APL | `python scripts/extract_populations.py` |
| 2 | Spiking sim shows sensible firing | `python scripts/test_sim.py` |
| 3–4 | Catch-the-ball + dopamine plasticity | `python scripts/train_catch.py` |
| 5 | Clash Royale perception (LDPlayer) | not started; needs a local session |

`train_catch.py` writes `runs/catch/learning_curve.{csv,png}` and
`weights.pt`. Useful flags: `--device cuda`, `--batch 64` (parallel games),
`--mode bidirectional`, `--encoding relative`, `--shaping 0.3`.

## Layout

```
flybrain/
  connectome.py   Codex CSV loader, synthetic generator, population extraction
  network.py      batched LIF simulation (sparse fixed + dense plastic weights), calibration
  plasticity.py   three-factor dopamine rule with eligibility traces
  agent.py        state -> PN encoding, MBON-group voting, reward -> DANs
  envs/catch.py   batched catch-the-ball
scripts/          milestone scripts
tests/
```

## Caveats

* Most of what the agent learns lives in the plastic KC→MBON synapses, not in
  anything the fly learned.
* Supercell's terms of service prohibit automation; the account used may be
  banned. Use a fresh alt account.
