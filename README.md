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

## Full card pool (109 cards)

`flybrain/envs/royale/` is the full game: every ladder card, built from
RoyaleAPI's datamine of the game files (`scripts/build_card_db.py` converts
it to `flybrain/envs/royale/data/cards.json` at tournament level 11).

* **Simulator** (`sim.py`): air and ground units, collisions, bridges and
  river jumps, splash, piercing and chain attacks, charge and dash, ramping
  damage, shields, spawners, death damage/spawns, timed bombs, building
  decay, freeze/stun/slow/rage/poison/heal/invisibility, area spells,
  Tornado, Clone, Mirror, Graveyard, Lightning, rolling spells, spell
  travel, tunnelling (Miner, Goblin Drill), Elixir Collector, champions and
  their abilities, double/triple elixir, overtime and tiebreak. Not in:
  card evolutions and event-only cards (no stats in the data).
* **Roles** (`db.py`): every card has one role, e.g. win condition, tank
  killer, splash air, small spell. The brain decides *which role* to play
  and in *which lane*; a helper picks the card in hand with that role. What
  it learns therefore carries over to any deck.
* **Your deck**: edit `decks/fly.txt` (8 cards, loose names like "pekka",
  "log" or "e-wiz" are fine). Training mixes your deck with random sensible
  decks so the brain learns every card.
* **Training**: `python scripts/train_royale.py` records the coach playing
  with many decks, teaches its choices to the brain, then tests it.
  `--split` gives the brain separate "play now?", role and lane outputs;
  `--init <brain> --dagger-rounds N` continues training on the brain's own
  games, labelled by the coach (DAgger). `scripts/compare_brains.py` plays
  brains on identical matches, pure and with the coach guard.
* **Coach** (`strategy.py`): its settings live in `COACH_DEFAULTS` and
  `PLACE_DEFAULTS`. Test a change with `scripts/duel_coaches.py`, which plays
  every random deck pairing twice with the coaches swapped, so deck and seat
  luck cancel and only the strategy difference shows (identical coaches
  score exactly 0). Tuned this way: defensive buildings 6 tiles from the
  river, spells only for trades worth 1.5 elixir more than they cost,
  elixir held until 9.8 (+5 to +6 points per game against the earlier coach).
* **Lookahead** (`lookahead.py`): before it defends, the coach tries every
  card in hand that could help at a few spots, and waiting, each in a copy of
  the match played 8 seconds ahead, and keeps the option that leaves the best
  position (tower health, troops left standing on both sides by elixir worth,
  elixir in hand). It beats the rule-only coach by +0.43 to +0.48 per game
  in duplicate format (about 71% wins to 28% losses). Spell trades are played
  out against waiting too (+0.12 more). The real bot and the
  browser game use it; `--no-lookahead` turns it off in the real bot, and
  `duel_coaches.py --a look` tests it.

## Clash Royale (simulated, starter deck)

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

`flybrain/real/` reads the LDPlayer screen and plays through adb, with the
full-card-pool brain (`models/fly_royale.pt`) and whatever deck you use:

1. **Screenshots and taps** via adb (`adb.py`, finds LDPlayer's adb by itself).
2. **Screen reading** (`perception.py`), calibrated on a real LDPlayer 540×960
   battle screenshot:
   - elixir from the pink bar;
   - your hand by matching each slot against the official card art of all
     109 cards (`cards.py`, downloaded once from RoyaleAPI's assets;
     greyed-out cards work too), and your 8-card deck learned from the hand;
   - tower HP from the HP bars (tracked over time, destroyed towers detected);
   - **troops with their type** (`troops.py`): a small CNN finds every troop,
     its side and what it is (96 troop types), trained on about 7,000 real
     labelled battle frames (see below). Without its weights the bot falls
     back to looking for red/blue level badges (positions only).
3. **Same inputs as in training** (`state.py`): the screen is turned back
   into a full-game simulator state (real troop types, so air/ground, tanks
   and building targeters are known), so the brain gets the 89 situation
   channels it learned on. The opponent's elixir is counted from the troops
   they deploy (`tracking.py`).
4. **Decision** (`bot.py`) once per second: the fly brain picks a card role
   and lane, and the **coach guard** (`envs/royale/guard.py`) steps in where
   the brain is weak. It defends and finishes towers, takes good chances the
   brain lets pass, cancels spells with nothing to hit, and saves elixir for
   your win condition. To defend, it first plays its options out a few
   seconds ahead in copies of the rebuilt game state (`lookahead.py`). Troops
   the bot just played stay in the state until the detector sees them, so it
   doesn't defend twice. `--no-guard` plays with the pure brain. Champions'
   ability button is tapped when enemies come at you.

### Troop detector training

`python scripts/train_troops.py --data <dataset>/images/part2` trains the
detector (and `scripts/train_troop_types.py` the close-up type classifier)
on KataCR's [Clash-Royale-Detection-Dataset](https://github.com/wty-yy/Clash-Royale-Detection-Dataset)
(MIT licence, see THIRD_PARTY_NOTICES.md): 568×896 arena crops of real
matches with every troop labelled. Whole recording sessions are held out to
measure accuracy on matches the net never saw. The dataset frames are
aligned to the LDPlayer screen by image registration (`troops.FRAME_ARENA`).

### Setup on Windows

1. Install **Python 3.11+** from python.org (tick "Add python.exe to PATH")
   and **Git** from git-scm.com.
2. Get the code and install packages (in a terminal):

   ```bat
   git clone -b claude/read-learn-twin-dyb48p https://github.com/YasuooMakeBaltica/Clash-fly
   cd Clash-fly
   pip install -r requirements-real.txt
   ```

3. **LDPlayer settings**: Settings → Advanced → resolution **540×960**
   (mobile/portrait). Settings → Other settings → **ADB debugging: open
   local connection**. Restart LDPlayer. Finish the Clash Royale tutorial by
   hand on your alt account.
4. **Your deck updates itself**: the bot recognises every card in your
   hand and works out your 8-card deck as the cards cycle. Once it has seen
   all 8 it uses them, and after the battle it saves them to
   `decks/fly.txt`. So just change decks in the game. If you want to set it
   by hand anyway (e.g. before the first battle), double-click
   `edit_deck.bat` (or `python -m flybrain.deck_editor`), tap 8 cards and
   press Save. `--no-auto-deck` turns the automatic update off.
5. **Calibrate** (start a Training Camp or friendly battle first):

   In PowerShell (the default Windows terminal). `--adb` is optional: the
   bot looks for LDPlayer's adb.exe in the usual folders by itself.

   ```powershell
   $ADB = "C:\LDPlayer\LDPlayer9\adb.exe"
   & $ADB devices
   python -m flybrain.real.calibrate screenshot --adb $ADB --out shot.png
   python -m flybrain.real.calibrate check --image shot.png
   ```

   (In the old Command Prompt, `cmd`, use `set ADB=C:\LDPlayer\LDPlayer9\adb.exe`
   and `%ADB%` instead of `$ADB`.)

   The screen positions were measured on a real LDPlayer 540×960 battle, so
   they should already fit. Open `shot_check.png`. If the boxes don't sit on the arena, elixir bar,
   cards and tower HP bars, run `python -m flybrain.real.calibrate pick
   --image shot.png` and drag them. The card names in the hand boxes should
   be right; only if one shows `?`, save its picture from your screen
   (the 4 hand cards left to right, then the small next card):

   ```bat
   python -m flybrain.real.calibrate templates --image shot.png --cards "Hog Rider,Musketeer,Cannon,Ice Golem" --next "Skeletons"
   ```
6. **Run**:

   ```powershell
   python -m flybrain.real.bot --adb $ADB --dry-run     # reads + decides, no taps
   python -m flybrain.real.bot --adb $ADB               # plays
   python -m flybrain.real.bot --adb $ADB --learn       # keeps learning from tower damage
   ```

   Start battles yourself; the bot waits between them. Each battle gets a
   folder in `runs/real/` with an annotated screenshot every 5 s and
   `log.jsonl`: one line per decision with what it saw (hand, elixir,
   towers, troops with types, the opponent's estimated elixir), what it
   played and who chose it (the fly, or the coach and why).

Supercell's terms of service prohibit automation: use an alt account.

## Layout

```
flybrain/
  connectome.py   Codex CSV loader, synthetic generator, population extraction
  network.py      batched LIF simulation (sparse fixed + dense plastic weights), calibration
  plasticity.py   three-factor dopamine rule with eligibility traces
  agent.py        state -> PN encoding, MBON-group voting, reward -> DANs
  envs/catch.py   batched catch-the-ball
  envs/clash/     first simulator (starter deck only); used by the browser game
  envs/royale/    full simulator: card database, roles, decks, coach (+ lookahead), guard, brain environment
  real/           real game: adb, screen reading, troop detector, elixir counting, bot loop
decks/fly.txt     the fly brain's deck (edit me)
web/              browser game (Beat the Fly): JS port of the sim and brain
models/           trained brains, troop detector and type classifier
scripts/          milestone scripts
tests/
```

## Caveats

* Most of what the agent learns lives in the plastic KC→MBON synapses, not in
  anything the fly learned.
* Supercell's terms of service prohibit automation; the account used may be
  banned. Use a fresh alt account.
