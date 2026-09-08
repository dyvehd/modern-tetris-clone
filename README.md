# Modern Tetris Clone

A Tetris Guideline clone aimed at *Jstris*-style rules (TETR.IO / Puyo Puyo
Tetris as secondary references), built with **pygame-ce**.

The architecture is split for later **Tetris AI development**:

```
┌────────────────────────────┐
│  tetris.engine (pure stdlib)│  ← deterministic, fixed-timestep (60 Hz),
│  board / SRS / scoring /    │    zero pygame imports, bitmask playfield,
│  game / env                 │    Numba/Cython-portable hot paths
└──────────┬─────────────────┘
           │ tick(actions, held)
┌──────────┴─────────────────┐
│  tetris.input               │  ← DAS/ARR/SDF keyboard logic (also pygame-free)
└──────────┬─────────────────┘
┌──────────┴─────────────────┐
│  tetris.render + tetris.app │  ← pygame-ce visualization on top
└────────────────────────────┘
```

The engine never imports pygame (enforced by a unit test). The AI works
against `tetris.engine.env.TetrisEnv` — see [AI environment](#ai-environment).

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[game,dev]"   # or: uv pip install ...
.venv/bin/python -m tetris          # menu → pick a mode → Enter
.venv/bin/pytest                    # 210 correctness tests
```

Modes: **Marathon** (Guideline curve gravity, 150 lines), **Sprint 40 Lines**
(0.02G), **Cheese 10 / 18 / 100 / ∞** (dig races, see below), **Zen**
(endless 0.02G, `Ctrl+Z` undo), **Zen 0G** (endless, no gravity — pieces
stay where you move them; they only lock via lock delay against the stack or
a hard drop), **VS Sandbox** (garbage trainer: sends 4 rows every 15 s). The
endless modes (Zen, Zen 0G, VS Sandbox) support **undo**: `Ctrl+Z` steps the
last placement back — the placed piece returns to your hands and the board,
queue, hold and score revert with it. The history survives restarts:
`R` then `Ctrl+Z` reaches into the game you just left, and `Ctrl+Z` on the
game-over screen steps back out of a top-out (TETR.IO zen behaviour).
The zen modes are also **mouse-editable sandboxes** (see below).

### Cheese race (dig mode)

The field starts on — and is topped back up to — a stack of garbage rows
with one hole each. **Messiness** (TETR.IO's term, default **100%**) is the
per-row chance that the hole moves to a different column: at 100% — Jstris
default cheese — two adjacent rows never share a hole, so nothing lines up
for an I piece; at 0% every hole sits in the same column (clean, four-tris
style). The **goal counts only dug cheese** — clearing lines of your own
stack never advances the counter. The **refill trigger** is Jstris's by
default: a downstack combo keeps the field reduced, and the cheese only
comes back (all at once, up to the goal cap) when a placement clears
nothing — clearing player junk without touching cheese keeps the combo,
so it keeps the field reduced too. With a goal the refill is
capped at the cheese lines still needed either way — the cheese only runs
out as you approach the goal and the last dig finishes the race (Jstris
goals 10/18/100 plus endless). The stack is **9 rows like Jstris**; put
`[rules] cheese_rows = 10` in `settings.toml` for the four-tris/Techmino
height, `[rules] cheese_messiness = 50` to loosen/tighten the holes, and
`[rules] cheese_refill_on_clear = true` for TETR.IO's instant refill
(topped back up after every placement, clears included).

### Zen sandbox editors (four-tris inspired)

In the zen modes (Zen, Zen 0G, VS Sandbox) the playfield and the queue are
directly editable with the mouse:

- **Left click / drag** paints gray blocks; **right click** or
  **Shift + left click** (draggable too) erases any cell. Painting works on
  the visible field and the buffer strip shown above it, and is interpolated
  so fast drags leave no gaps. Every stroke is one `Ctrl+Z` undo step.
- **4-cell drag = tetromino**: dragging across exactly 4 cells that form a
  connected shape colors them with that piece's color (a straight line
  becomes a cyan I, an L-shape a J or an L by chirality, ...). Any connected
  4-cell shape is one of the 7 tetrominoes, so the match is always unique.
  A 5th cell reverts the four to gray — the coloring only happens for
  *exactly* 4. Like four-tris's AutoColor, separate clicks that complete a
  4-cell gray component also color it; oversized regions never recolor.
  Edited lines behave exactly like placed ones (they clear normally).
- **Queue editor**: left click the next-pieces preview to open a dialog.
  Type a sequence of piece letters (`IJLOSTZ`, any length — prefilled with
  the current queue; empty = back to random bags) and a **7-bag offset**
  (0–6, default 0): how many pieces of the current bag were already dealt
  before your sequence, i.e. where the bag boundaries fall. `TAB` switches
  fields, `ENTER` applies, `ESC` cancels. The queue then deals your sequence
  followed by fresh shuffled 7-bags, and the preview draws **bag
  separators** at the boundaries (offset 2 with a 7-piece sequence shows
  the first separator after 5 pieces). The game freezes while the dialog is
  open, and a queue edit is also one `Ctrl+Z` step.

### Controls (Jstris-like defaults)

| Action | Keys |
| --- | --- |
| Move left / right | ← / → |
| Soft drop | ↓ |
| Hard drop | Space |
| Rotate CW / CCW / 180 | ↑ or X / Z or Ctrl / A |
| Hold | C or Shift |
| Pause / restart / menu | Esc or P / R / Q |
| Undo last action (Zen modes) | Ctrl + Z |
| Paint / erase board cells (Zen modes) | left mouse / right or Shift+left |
| Edit the piece queue (Zen modes) | left click the next preview |
| Open settings | S (main menu) or S / O (pause) |
| Screenshot | F12 |

### In-game settings (S in the menu, S/O in pause)

- **Handling**: DAS (0–333 ms), ARR (0–100 ms, 0 = instant to wall), soft
  drop speed (1–40x, past 40x = ∞ instant). `LEFT/RIGHT` adjusts in steps of
  5 (hold `Shift` for ±1); changes apply **live** to the running game.
- **Key remapping**: select any action, press `Enter`, then press the key you
  want (modifiers like `lshift`/`lctrl` work; `Backspace` clears a binding;
  `Esc` cancels). Rebindings are active immediately, including mid-game.
- `R` resets handling + keys to defaults; leaving the screen auto-saves to
  `settings.toml` (repo root) — hand-edited sections like `[rules]` are
  preserved on save.

All keybinds, DAS/ARR/SDF, gravity, lock delay, garbage behaviour etc. can also be
overridden in a `settings.toml` (repo root or `~/.config/modern-tetris-clone/`):

```toml
[rules]
lock_delay_ms = 500
soft_drop_factor = 20      # math.inf is allowed (instant soft drop)

[input]
das_ms = 133
arr_ms = 0
```

### Input log (mis-drop debugging)

Every session writes `logs/input_<date>_<time>.log` recording each key
press/release (by bound action, so remaps are reflected) and every piece
you take control of, with millisecond timestamps:

```
    12.345 T piece
    12.401 Left (key down)
    12.455 Left (key up)
    12.501 Harddrop (key down)
    12.518 L piece          # next piece spawns the tick the drop resolves
    12.610 Harddrop (key up)
```

The first line of each game records mode, RNG seed (the exact piece
sequence is replayable with it) and your handling; pauses, restarts,
handling changes and top-outs are marked too. A mis-drop can then be
traced to either your own input or a missing/duplicated event — e.g. a
`key down` with no `key up` means the release never reached the game.

Cost is one buffered-free `O_APPEND` write per event (a page-cache
syscall, ~µs) and one identity comparison per logic tick; disable with
`[debug] log_input = false` in `settings.toml` (or point it elsewhere
with `input_log_dir = "..."`). No file is created until a game starts.

## Implemented rules (Guideline / Jstris)

| Rule | Implementation | Default |
| --- | --- | --- |
| SRS wall kicks | Verbatim JLSTZ + I tables from [harddrop wiki](https://harddrop.com/wiki/SRS), pinned by tests (wiki convention +y-up, converted internally) | — |
| 180° rotation | SRS+ kick table (osk's published TETR.IO "180 kick table"; the same table Jstris' 180 option uses per the now-removed wiki article). Universal — all pieces, I included. N→S/S→N off-the-floor/ceiling kicks, E⇄W only horizontal "unhook"/"zipper" kicks and 2-up "tall jumps", never downward. 180s count as rotations for T-spin detection (no mini-upgrade kicks on 180 transitions) | — |
| 7-bag randomizer | Standard bag; deterministic per seed | — |
| Spawn | Rows 21–22 from the bottom, fully above the visible field ("later games" behaviour per the SRS wiki); I on the lower row, JLSTZ lean left, O central; block-out + lock-out top-outs | — |
| Ghost piece | Hard-drop shadow | on |
| Hold | Once per piece, resets piece state | on |
| IRS / IHS | **Off** (TGM-only; Jstris/TETR.IO/Nullpomino/Techmino don't rotate or hold a piece because the key was still held at spawn — verified against real games). Available as opt-in `irs_enabled`/`ihs_enabled` flags | off |
| Initial DAS | Held direction carries across spawns (charges while pieces lock; 0 ARR spawns at the wall) | on |
| Lock delay | 500 ms, move reset, **15-move budget**, new lowest row resets the budget, hard drop locks instantly | 500 ms / 15 |
| DAS / ARR | 167 ms / 33 ms (TETR.IO defaults); ARR 0 = instant to wall; opposite-direction press re-arms DAS | 167 / 33 |
| SDF (soft drop factor) | 5× gravity while held | 5 |
| Gravity | Fixed G or the Tetris Worlds marathon curve `(0.8 − 0.007(l−1))^(l−1)` s/row | per mode |
| Line clear delay | **0 ms** (Jstris/TETR.IO style); ARE 0 | 0 |
| T-spin detection | 3-corner rule (walls count), last successful movement must be a rotation; **mini** unless both front corners filled; the TST kick (5th test on 0⇒R / 2⇒L) upgrades to full. Gravity falls and successful shifts cancel; hard drops do **not** | — |
| Scoring | Jstris: 100/300/500/800; T-spin 400/800/1200/1600; mini 100/200; mini TSD counts as TSD; PC +3000; B2B ×1.5 (score); combo +50×(combo−1); soft drop 1/cell, hard drop 2/cell | — |
| Attack (garbage sent) | Jstris table: single 0, double 1, triple 2, quad 4, TSS 2, TSD 4, TST 6, mini TSS 0; B2B +1 (mini TSS keeps the chain but no bonus); PC = 10; combo table 0/0/1/1/1/2/2/3/3/4/4/4/5 (5 from the 13th consecutive clear) | — |
| B2B chain | Quads + any line-clearing T-spin are difficult; non-difficult *clears* break it; non-clearing locks don't touch it | — |
| Garbage | 500 ms delay, cancels against your outgoing attacks FIFO, cap 8 rows per rise, rises at spawn, single hole per row, overflow = top out | see config |

Known intentional deviations / approximations (flagged for testing):

- **Jstris buffer-zone quirks are not replicated.** Jstris uses a single
  buffer row that is not solid and erases rows locked above it (the 20TSD
  line-clip exploit). We use the Guideline's solid 20 hidden rows instead,
  like TETR.IO.
- Combo counting: the engine's `combo` is "consecutive clearing locks",
  starting at 1. The Jstris combo table is applied as
  `COMBO_ATTACK[combo - 1]` — i.e. the **3rd** consecutive clear is the first
  to add garbage. Jstris's own display convention ("REN x") may differ by one;
  *please verify in game*.
- B2B stacking on a Perfect Clear adds +1 attack (PC otherwise replaces the
  clear's attack). Jstris's exact PC/B2B interaction is unverified.
- IRS/IHS exist only in the TGM series; they are off here (opt-in flags),
  matching Jstris/TETR.IO/Nullpomino/Techmino. Initial DAS, which all of
  those games do have, is on (it lives in the input controller).

## AI environment

`tetris.engine.env` gives a deterministic, gym-style environment:

```python
from tetris.engine.env import TetrisEnv, Action

env = TetrisEnv(seed=42)                 # or GameConfig(...) for rule tweaks
obs = env.reset()

while not obs.over:
    placements = env.reachable_placements()   # [(x, rot, y), ...] hard-drop spots
    # ... pick one / one action per tick:
    obs, info = env.step(Action.HARD_DROP)    # 1 logic tick = 1/60 s
    env.add_garbage(4)                        # simulate an opponent

print(obs.score, obs.lines, obs.attack_sent)
```

- `Observation.rows` is the playfield as 40 integer row bitmasks (row 0 =
  top, bit 0 = leftmost column) — plain data, ready for Numba/Cython or
  tensorization.
- `env.clone()` deep-copies the full simulation (RNG included) for tree
  search; same seed ⇒ identical games (tested).
- `reachable_placements()` BFS over (x, y, rot) with shifts/rotations and
  returns all hard-drop landings of the current piece — the standard
  interface for placement-selection AIs. It intentionally does *not* model
  lock-delay move resets or T-spin paths yet; use `step()` for those.
- `info["events"]` carries per-tick clear/tspin/garbage events for reward
  shaping.

## AI layer (movement engine)

`tetris.ai` is the bot's mobility core, built for a 0G planning model —
following the architecture shared by MisaMino, Cold Clear, Zetris and
cobra-movegen (see `tmp/` reference clones): **placement enumeration**
decoupled from **target-to-inputs navigation**.

- `enumerate_placements(rows, piece)` — exhaustive BFS over movement states
  from spawn (shifts, SRS rotations with kicks, optional 180, sonic drops)
  → every reachable resting placement, each with its engine-accurate spin
  verdict (`none` / `mini` / `full`, Jstris rules). Spin-in T-spin slots
  (TSD/TST) are found with their classification — a drop-column
  enumeration can never see them.
- `find_path(rows, piece, placement)` → shortest `Action` input sequence
  (ending `HARD_DROP`) that steers the piece from spawn into the target;
  replaying it through `Game.tick` locks exactly the predicted cells with
  the predicted spin (asserted as a test oracle over every placement).
- Movement model: **infinite soft drop only** (sonic drop) — competitive
  standard; placements that require stopping mid-fall are deliberately
  unreachable, and every real spin entry survives the cut (pre-rotation
  rest → kick → hard drop). A T-spin always ends rotation → hard drop.
- No finite-SDF timing, no lock-delay move budgets, no gravity clocks:
  0G reachability. Timing filters for real-time modes come later, on top.
- Performance (CPython): ~2 ms per (board, piece) enumeration, ~0.5 ms per
  path — the numbers that size a future beam search.

```python
from tetris.ai import enumerate_placements, find_path

placements = enumerate_placements(game.rows, game.active.type)
best = ...                                  # the future search layer picks
path = find_path(game.rows, game.active.type, best)
for action in path:                         # then drive the real game
    game.tick([action])
```

`tetris.engine.env.reachable_placements()` (the older hard-drop-only
interface) remains for simple agents; `tetris.ai` supersedes it with spin
classification and spin-in reachability.

## AI cheese-race harness

`tetris.ai.cheese` wraps the engine into the episode protocol the AI stack is
built and measured on — **the position-prediction / navigation decoupling
made literal**: an agent predicts *placements* (where the current — or held —
piece should lock); the harness navigates (movegen + pathfinder → inputs).

- `CheeseEnv` — episode parameters: level (cheese lines to dig), stack height
  (Jstris keeps 9), messiness, refill semantics (Jstris vs TETR.IO), hold,
  previews, piece cap. 0G + infinite SDF: pieces-per-dug-line is the only
  measured skill.
- `run_episode(agent, env, seed)` — one seeded episode. Two application
  modes: `navigate=True` converts each placement to its shortest input
  sequence and replays it through the real engine (full fidelity, per-piece
  input log); `navigate=False` teleports + hard-drops (2 ticks/piece — same
  cheese outcomes, the fast path for big batches). Illegal agent decisions
  raise `InvalidDecision`; a piece cap turns unfinished episodes into
  `capped` results, never hangs.
- `run_batch(...)` → `BatchResult` — ≥1,000-episode batches with win rate,
  mean pieces-to-clear over wins, 95% CI, pieces/line, failure breakdown.
- `Obs` — the frozen snapshot an agent sees (rows, active, hold, previews,
  cheese counters); `Decision(placement, hold)` — what it answers.

Baseline agents (`tetris.ai.agents`): `RandomAgent` (floor) and
`GreedyDigAgent` (greedy line clears, hold when strictly better). At level 1
the greedy agent is optimal wherever 1 piece can win (99% of seeds); the
remaining ~1% — hole 0 with S/O only, hole 9 with Z/O only — genuinely need
2+ pieces (a vertical S's pointy bottom cell sits on its *left* column, so
column 0 is out of reach; Z mirrors this at column 9; O never digs alone).
The pinned edge-seed tests document greedy's 1-ply myopia cost there: the
headroom for search (Deliverable 2).

```python
from tetris.ai import CheeseEnv, GreedyDigAgent, run_batch

env = CheeseEnv(level=10)   # dig 10 lines, Jstris stack/messiness
print(run_batch(GreedyDigAgent(), env, 1000).summary())
```

## AI search baselines

`tetris.ai.search` + `tetris.ai.eval` — the search baselines every learner is
measured against (Deliverable 2 of the cheese-race plan).

- **Eval** (`eval.py`): linear board eval in the Dellacherie lineage, adapted
  for downstacking — holes (heavily weighted: they block the shaft), covered
  cells above holes (depth), aggregate height, bumpiness, row/column
  transitions, well depth, and empty cells of a *partially*-filled bottom row
  (the dig target; a fully-dug board is exempt). Weights are hand-set now;
  rung-1 learning tunes exactly these numbers.
- **`OnePlyAgent`**: best single placement (active or hold) by eval + line
  bonus + win bonus.
- **`BeamAgent`** (`width` × `depth`): beam over the preview queue (depth is
  capped by what the agent can see), hold as a branch at every ply with the
  engine's can_hold semantics, win terminals score a big bonus minus a
  per-piece decay — so among winning plans the *shortest* is preferred.
- **Quiescence rule** (the key correctness property): a placement that
  clears nothing is a beam leaf at *every* ply. Under Jstris refill
  semantics the cheese tops back up after a combo break with RNG hole
  positions the search cannot know, so no future win may be credited
  through a no-clear move. This is what let the beam lose the pinned
  level-1 edge seeds by committing junk on the promise of a next-piece dig
  that the refill destroyed — fixed, the seeds now solve at the true
  2-piece minimum.

Measured (Jstris cheese, stack 9, messiness 100%, 100 episodes/level —
`examples/cheese_baselines.py`):

| level | greedy win% | greedy pieces/line | 1ply | beam20x4 |
|------:|:-----------:|:------------------:|:----:|:--------:|
| 1     | 100         | 1.03               | 1.01 | 1.01 |
| 2     | 100         | 1.39               | 1.20 | **1.16** |
| 3     | 96          | 3.40               | 1.76 | **1.58** |
| 5     | 66          | 6.06               | **3.46** | 3.59 |
| 10    | 40          | 9.60               | 5.12 | **4.86** |

The eval-driven agents never top out (greedy: 40% win at level 10) and
roughly halve pieces-per-line — the reference numbers for the curriculum
ladder. Run `examples/cheese_baselines.py` for the live table.

## AI weight tuning (rung-1 learning)

`tetris.ai.tuning` — cross-entropy method over the downstack eval weights,
the historical quality jump of heuristic Tetris (Dellacherie → BCTS) applied
to the cheese objective: candidates play real seeded episodes through the
*search agent itself*, and the cost is mean pieces-to-clear with failures
charged the full piece cap — survival is a hard constraint, never a
tradeable.

- **Cost** = capped mean pieces across the training levels (``levels=(3,5,10)``).
  Single-level training measurably overfits: weights tuned at level 10
  alone won level 10 (4.185 vs 4.322 pieces/line) but *regressed* levels 3-5.
- **Search**: Gaussian sampling around the elite mean, incumbent always
  evaluated (no-regression guarantee), per-weight clip bounds, smoothing
  blend, fork-pool parallelism — a whole generation evaluates in one
  wall-clock generation time (~90 s for 24 candidates × 40 episodes × 3
  levels on a 16-core laptop).
- **Determinism**: identical config → identical trajectory (seeded sampler,
  fixed episode seeds).
- Tuned models land in `models/*.json` with full history;
  `examples/validate_weights.py` compares a tuned agent to the hand-set
  baseline across the curriculum ladder.

```python
from tetris.ai.tuning import TuningConfig, tune

best, history = tune(TuningConfig(
    agent="beam20x4", levels=(3, 5, 10),
    episodes=40, generations=12, candidates=24,
))
```

## AI curriculum learner (Deliverable 4, GPU-first)

`tetris.ai.policy` + `tetris.ai.curriculum` — the learned placement policy,
torch-native from day one: the same code trains on the local RTX 3050 and,
unchanged (`--device cuda`), on the RTX Pro 6000 server.

- **Architecture**: *score-per-candidate* MLP (the fusion bot's policy
  shape): the state (downscaled board, column heights, piece one-hots,
  cheese counters) is paired with each candidate placement (4x4 pattern,
  position, rotation, hold flag) and scored; softmax over candidates picks
  the decision. Porting to a bigger net or PPO replaces one module.
- **Training**: REINFORCE with moving-average baseline and entropy bonus.
  Rollouts run on CPU through the harness (engine-bound regardless of
  device); every decision of every episode in an iteration is concatenated
  into **one batched forward/backward on the GPU** — the per-decision
  softmax is segmented by candidate group (`scatter_reduce` logsumexp),
  one Adam step per iteration. Reward: −1 per piece + potential-based
  shaping (alpha per cheese line dug — telescope-sums away, so the optimal
  policy is unchanged) + terminal win bonus.
- **Curriculum gates** (the user's rule, made statistical): advance on
  **strict** improvement — learner mean + 95% CI below the reference mean
  over a fresh gate batch — or on a **tie**: matching the baseline within
  the batches' combined noise with no worse win rate (necessary at levels
  where the reference plays the theoretical optimum, like level 1's mean
  1.00; measured: a perfect learner was blocked forever by the strict rule
  alone). With retention on, every easier level is re-gated on advancement;
  a level that blocks twice stops the ladder.
- **Search-oracle distillation** (the fusion-bot recipe, now the
  curriculum's warm-start): measured on this repo, cold-start REINFORCE
  cannot clear even the level-1 gate at sane budgets — sampled win rate
  climbs (~30%) but the argmax stays near-random (~3%), each (piece x
  hole-column) state seeing ~one winning example per iteration. Distilling
  the beam search's decisions as candidate-index labels on the policy's
  own encodings fixes it: 62k decisions x 300 epochs (chunked GPU batches
  — a full batch OOMs a 4 GB 3050) reaches 96% teacher-decision accuracy
  and plays level 1 at the 1.00-piece optimum, level 2 at 100% wins /
  2.44 pieces, level 3 at ~96% wins. RL then fine-tunes from there.
- Checkpoints per passed level: `models/curriculum/cheese_policy_L{N}.json`.

```bash
.venv/py.sh examples/cheese_learner.py --max-level 10 --distill-episodes 4000
```

- **DAgger hard-state mining** (`tetris.ai.dagger`): behavioral cloning
  leaves a compounding-error gap — the policy's rare missteps land on board
  states the teacher never visited, and ~1% of level-2 episodes top out
  that the beam wins easily (measured: 99% win but 2.71 mean pieces vs the
  beam's 2.18). DAgger closes the distribution shift: roll the *current*
  policy, label every visited state with the *teacher's* move (mixture
  play, beta decaying 0.8→0), re-distill. Measured on the RTX 3050:
  7 rounds × 400 episodes → 2.51 mean pieces at the same 99% win rate.
  The remaining gap to the beam is scale (rounds, episodes, net size) —
  the RTX Pro 6000 workload.

## Project layout

```
src/tetris/
  engine/          # pure, dependency-free simulation
    constants.py   # field, piece data, SRS kick tables, colors
    rng.py         # 7-bag
    board.py       # bitmask playfield ops
    scoring.py     # Jstris score/attack/B2B/combo tables
    game.py        # state machine: rotation, lock delay, gravity, hold,
                   # T-spin detection, garbage, spawns/top-out
    env.py         # TetrisEnv + placement search
  ai/              # bot mobility core (0G planning model)
    movegen.py     # exhaustive reachable-placement BFS + spin classes
    pathfinder.py  # placement -> shortest Action input sequence
    cheese.py      # cheese-race episode harness (env, obs, agent protocol)
    agents.py      # baselines: random, greedy dig
    eval.py        # downstack linear eval (Dellacherie-lineage features)
    search.py      # 1-ply + beam agents (hold-as-branch, quiescence leaves)
    tuning.py      # cross-entropy weight tuning (rung-1 learning)
  input/           # DAS/ARR controller (no pygame)
  render/          # pygame-ce renderer
  app.py           # 60 Hz fixed-timestep game loop, menus
  config.py        # defaults + settings.toml override
tests/             # 232 tests pinning all of the above
```

## Verification checklist (for pro-player review)

Things that most need human eyes on a real keyboard:

1. **Spawn visibility** — pieces appear in the buffer rows above the skyline
   (Guideline rows 21–22) and drop into view, like TETR.IO's buffer rendering.
2. **Lock delay feel** — 500 ms with 15 move-resets; sliding off a ledge
   restarts on the next lowest row.
3. **DAS/ARR** — 167/33 defaults; ARR 0 teleports to the wall; re-pressing
   the other direction resets DAS (matches TETR.IO default, no DCD yet).
4. **T-spins** — mini vs full on front corners; TST kicks upgrade to full;
   hard-dropped spins count; spins after a gravity fall do not.
5. **180 rotation** — SRS+ kicks (osk's table): T on the floor N→S pops
   **up 1** ("off-the-floor"); horizontal 180s kick sideways ("unhook") or
   jump **up 2**, never down. If Jstris's 180 option ever feels different,
   tell me — its wiki documentation was removed.
6. **Combo attack start** — +1 garbage from the 3rd consecutive clear.
7. **Garbage** — delay/cancel/cap/rise defaults are approximations of VS
   rules; the trainer mode exercises them.
8. **Cheese race** — compare the stack size and refill rhythm against Jstris
   cheese (9 rows, topped up after every placement that clears nothing, combo
   junk clears included, shrinking near the goal).
9. **Next preview scale** — pieces drawn at board-cell size in ~3-row slots,
   queue top aligned with the visible board top (Jstris measurements).
10. **Sandbox editors** — compare the 4-cell auto-coloring and the queue
    editor against four-tris (stroke behavior, the 5th-cell revert, the bag
    offset's effect on the preview separators).
