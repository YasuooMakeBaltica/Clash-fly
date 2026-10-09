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
| 5 | Clash Royale simulator + coaching | `python scripts/train_clash.py` |
| 6 | Real game: read the screen, play through adb | `python -m flybrain.real.bot` (calibrate first) |

`train_catch.py` writes `runs/catch/learning_curve.{csv,png}` and
`weights.pt`. Useful flags: `--device cuda`, `--batch 64` (parallel games),
`--mode bidirectional`, `--encoding relative`, `--shaping 0.3`.

## Clash Royale (simulated)

Before touching the real game, the brain learns in a simplified simulator
(`flybrain/envs/clash/`): the 18×32 arena with river and bridges, princess
and king towers, elixir (double in the last minute), an 8-card starter deck
cycling through a 4-card hand, troops that walk their lane, lock onto the
nearest target and fight, delayed area spells, crowns and a 3-minute match.
Stats are approximate. No collisions, air units or overtime.

**Coach.** `strategy.py` holds a scripted coach that plays basic strategy:
finish towers with spells, defend first with the right counter (tank →
Mini P.E.K.K.A, swarm → Arrows, melee → Goblins, ranged → Knight), make
positive spell trades, counter-push with survivors, punish a tank dropped at
the back by attacking the other lane, and never sit at full elixir. In the
simulator it beats a random bot ~99% and a basic bot ~95% of the time.

**What the brain sees and does.** 62 situation channels (elixir level, cards
ready, threat size and type per lane, own pushes, tower health, spell value)
drive PN groups. MBONs are split into two voting sets: which card (or wait)
and which lane. Placement inside the lane uses the same helper as the coach.

**Training** (`scripts/train_clash.py`):
1. *Coaching:* in each situation the coach's move is paired with reward
   dopamine, so KC→MBON synapses for other moves weaken. Moves actually played
   shift from the coach's to the brain's own as coaching goes on.
2. *Practice:* the brain plays alone against random → basic → coach bots,
   moving up after winning 60% of recent matches, learning from tower damage,
   crowns and wins.

`scripts/record_clash.py --brain runs/clash/brain.pt` records a match
(positions, elixir, the brain's votes and what the coach would have done) for
the replay viewer.

## Playing the real game (LDPlayer)

`flybrain/real/` reads the LDPlayer screen and plays through adb:

1. **Screenshots and taps** via adb (`adb.py`). LDPlayer ships its own adb
   (e.g. `C:\LDPlayer\LDPlayer9\adb.exe`); enable ADB in LDPlayer's
   settings and set the resolution to 540×960 (portrait).
2. **Screen reading** (`perception.py`): elixir from the pink bar, your hand
   by matching card pictures, tower HP from the HP bars (tracked over time,
   destroyed towers detected), troops from their red/blue level badges.
   Troop *types* are not recognised yet; that needs a trained detector.
3. **Same inputs as in training** (`state.py`): what's on screen is turned
   back into a simulator state, so the trained brain gets the exact 62
   situation channels it learned on, and cards are placed with the same lane
   helper.
4. **Decision and tap** (`bot.py`) once per second: tap the card slot, then
   the drop spot.

The deck must be the training deck: Knight, Archers, Giant, Musketeer,
Mini P.E.K.K.A, Goblins, Fireball, Arrows. The trained brain is in
`models/fly_v2.pt`.

**Setup on your PC** (screen positions in the code are estimates until you
calibrate):

```bash
pip install -r requirements-real.txt
set ADB="C:\LDPlayer\LDPlayer9\adb.exe"
# in a battle:
python -m flybrain.real.calibrate screenshot --adb %ADB% --out shot.png
python -m flybrain.real.calibrate check --image shot.png      # look at shot_check.png
python -m flybrain.real.calibrate pick --image shot.png       # drag boxes if they're off
python -m flybrain.real.calibrate templates --image shot.png --cards "Knight,Archers,Giant,Musketeer" --next Goblins
#   ...repeat templates with other screenshots until all 8 cards are saved
python -m flybrain.real.bot --adb %ADB% --dry-run             # reads + decides, no taps
python -m flybrain.real.bot --adb %ADB%                       # plays
python -m flybrain.real.bot --adb %ADB% --learn               # keeps learning from tower damage
```

The bot saves annotated screenshots to `runs/real/` every 5 s so you can see
what it read. Supercell's terms of service prohibit automation: use an alt
account.

## Layout

```
flybrain/
  connectome.py   Codex CSV loader, synthetic generator, population extraction
  network.py      batched LIF simulation (sparse fixed + dense plastic weights), calibration
  plasticity.py   three-factor dopamine rule with eligibility traces
  agent.py        state -> PN encoding, MBON-group voting, reward -> DANs
  envs/catch.py   batched catch-the-ball
  envs/clash/     Clash Royale simulator, coach and bots, brain environment
  real/           real game: adb, screen reading, calibration, bot loop
web/              browser game (Beat the Fly): JS port of the sim and brain
models/           trained brains
scripts/          milestone scripts
tests/
```

## Caveats

* Most of what the agent learns lives in the plastic KC→MBON synapses, not in
  anything the fly learned.
* Supercell's terms of service prohibit automation; the account used may be
  banned. Use a fresh alt account.
