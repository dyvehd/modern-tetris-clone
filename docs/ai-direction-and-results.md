# Cheese-Race AI: Direction and Preliminary Results

Status: work in progress. Written for an expert review of the
direction; every number here was measured on this repo (local RTX 3050
unless marked *server* — NVIDIA RTX PRO 6000, 96 GB, 20-core Molab
instance).

Companion documents: [onboarding-guide.md](onboarding-guide.md) (code
tour) and [cheese-ai-learning.md](cheese-ai-learning.md) (plain-language
training concepts).

## The task

A cheese race: the field starts with a 9-row garbage stack (one hole
per row, 100% messiness — Jstris cheese rules), and the objective is to
clear `level` cheese lines with **the fewest pieces**. 0G, infinite SDF,
no ARE: placement quality is the only measured skill. The curriculum
starts at 1 line and scales to 10; the learner advances only when it
beats — or statistically ties — a search reference on fresh seeds.

## What was considered (and the decision)

### 1. What the agent controls

- **Inputs-per-tick agents** (frame-perfect button presses): rejected.
  Navigating real inputs is the *rendering* problem; the skill we want
  to measure is placement quality. The harness protocol: agents decide
  **placements**; a shared movegen+pathfinder layer produces inputs
  (verified mode) or teleports the piece (2-tick fast path, asserted
  outcome-identical). One seam, every agent — random, search, learned —
  uses it.

### 2. The objective

- **PVP-vs-bot / garbage battles**: deferred. Cheese race is a clean,
  fully observable, single-agent objective whose optimum we can *verify*
  (level 1's is 1.00 pieces — the beam plays it).
- **Level-1 ground truth was extracted from the movegen, not guessed**:
  I/J/L/T dig any hole column; S cannot open column 0, Z cannot open
  column 9, O never digs alone. One-piece wins exist on ~99.0% of seeds
  (500-seed sweep: 496 × 1-piece).

### 3. Reference / teacher

- Beam search (width 20, depth 4) with a hand-tuned eval, plus 1-ply and
  a lexicographic greedy. The beam is the teacher *and* the gate
  opponent. Its own pieces-per-line: ~4.86-5.12 across levels, 100% win
  at low levels.
- A quiescence rule was required for search correctness: under Jstris
  refill, a no-clear placement cannot be credited any future win (the
  cheese tops back up with unknowable hole positions after a combo
  break). Fixing this changed beam level-1 behavior from occasionally
  3-piece to the true 2-piece edge-case minimum.

### 4. Policy architecture

- **Score-per-candidate MLP** (the fusion-bot shape): the net scores
  each (board-state, candidate-placement) pair; softmax over candidates
  is the move distribution. State: full-resolution 20×10 occupancy +
  column heights + piece one-hots + cheese counters (235 dims).
  Candidate: 4×4 pattern + position + rotation + hold flag (39 dims).
  Full cell resolution is load-bearing — a pooled 5×10 view literally
  cannot see a one-cell cheese hole (measured).
- **Torch-native from day one, GPU-first** (rollouts CPU / updates GPU
  batched): the same code runs on the 4 GB laptop and the 96 GB server;
  the server got bigger batches and a bigger net with zero porting.

### 5. Training algorithm

- **Cold-start REINFORCE**: fails at level 1 at sane budgets (measured:
  sampled win climbs ~17%→39% but greedy stays ~3-7%; each (piece ×
  hole-column) config sees ~1 winning example per iteration). Kept as
  a fine-tune stage, not an initializer.
- **Distillation from the beam** (behavioral cloning): works. 62k
  teacher decisions → 96.4% teacher-decision accuracy, level 1 at the
  1.00 optimum, level 2 at 100% win / 2.44 pieces (teacher: 2.32).
- **DAgger** (policy-visited states, teacher labels, mixture play β
  0.8→0): fixes behavioral cloning's compounding topouts — level 2
  mean 2.71 → 2.51 at 99% win.
- **CEM eval-weight tuning**: machinery works (deterministic,
  no-regression), but tuned weights did not transfer to fresh seeds —
  kept in the repo as a negative result.
- **Curriculum + statistical gates**: the user's rule ("advance only
  when it consistently beats the baseline, option to retain easier
  levels"), formalized as strict CI-separated improvement OR a
  statistical tie (mean within combined noise AND win rate within the
  pooled two-proportion band), with retention re-gating and a
  twice-blocked stop.

### 6. Parallelism

- Data collection is ~90% of wall time (pure-Python rollouts), so the
  server phase added a forkserver process pool: fork (not of the
  torch parent — deadlock lottery, measured), per-episode tasks
  (chunksize=1 for load balance, 3.6x→5.6x on 15 cores), teacher
  shipped by name (lambdas don't pickle), policy shipped once per
  worker via pool initializer. Parallel output asserted byte-identical
  to sequential.

## Current direction

Climb the 1→10 ladder with: mixed-level distillation warm-start
(`distill_level_window=3` — level-L-only data is structurally starved
because a 1-piece episode carries one decision), REINFORCE fine-tune,
then safeguarded DAgger rounds (teacher replay mixed in, ¼-lr
fine-tune, no-regression probe with automatic weight restore). A 256×3
net (202k params). Gates at 300 episodes per side on seed bands
disjoint from training by construction.

## Preliminary results

### Baselines (500 seeds/level, fresh-seed validation)

| agent | L1 win | L2 win | L10 win | L10 pieces/line |
|---|---|---|---|---|
| greedy (hand heuristic) | 99% | ~97% | ~40% | 9.60 |
| 1-ply search | 100% | 100% | ~100% | ~5.12 |
| beam20x4 (reference) | 100% | 100% | ~100% | ~4.86 |

### Learner progressions (each row a real run on this repo)

| run | setup | outcome |
|---|---|---|
| v5 laptop | 128×2, 62k mixed decisions, distill | L1 optimum 1.00; L2 100%/2.44; ladder stalled at L2 (97% vs 100% win) |
| v5 laptop | + 7 manual DAgger rounds | L2 2.71→2.51 at 99% win |
| server run 1 | 256×3, 8k L1-only distill episodes | twice-blocked at L1: 80% gate win, topouts at 33+ pieces — data starvation (8k decisions for 202k params) |
| server run 2-3 | 24k episodes, 3-level window (67k decisions, 78-84% acc) | distill+REINFORCE reached 99%/1.03 gate-range; DAgger then wrecked the net to 84%/2.14 — junk-board-state forgetting (missed holes spiral into 30-40-piece failures, so policy-rollout data is ~8:1 junk states) |
| server run 4 | + DAgger safeguards (replay, ¼ lr, probe-restore) | probe worked (restored on regression); gate blocked on 99% vs 100% win at n=300 — exact-equality demanded perfection on noise |
| server run 5 | + statistical win-rate gates (pooled two-proportion) | **L1 PASSED** (100% win / 1.13 vs beam 1.01, statistical tie; DAgger converged, probe improved it); L2 in progress at time of writing |

### The failure→fix ledger (what a reviewer should weigh)

Every design change in the learner stack came from a measured failure,
not a hunch: pooled encoding hid the hole → full resolution; L1-only
distill starved the net → mixed-level window; DAgger forgot clean-board
play → replay + fine-tune lr + no-regression probe; exact-equality tie
rule blocked on noise → two-proportion tests; poll SIGINT killed a
detached runner → `start_new_session`; a 4 GB GPU OOM'd → chunked
training (now a server-tunable chunk size).

## Open questions we'd value review on

1. **DAgger economics at higher levels.** At level 2+, policy rollouts
   visit long junk-free games, so the junk-state ratio drops — but the
   probe n=100 may still be too weak to catch mean regressions (a
   1.03→1.22 mean slip passed it once). Is a gate-strength probe (300
   episodes) worth its cost inside the ladder?
2. **Distillation accuracy ceiling.** ~76-84% teacher-move accuracy on
   67k-134k decisions, not the 96.4% the laptop's 62k got on its
   distribution — plausibly the *mixed-level* window makes the task
   harder (the teacher's own choices are less predictable on junk
   boards?). Worth measuring per-level accuracy.
3. **Gate stringency at scale.** The tie rule passes a learner whose
   mean is within combined noise AND whose win rate is within the
   proportion band. At n=300 the win-rate band is ~±2-3 points — a
   learner 1-2 points below the reference still advances. Is that the
   right trade between ladder progress and mastery?
4. **Search as the permanent teacher.** The beam is ~4.86-5.12
   pieces/line at L10 — far from the human pro level (sub-4). The
   learner can only approach its teacher. Options: deeper/wider beam
   (compute), or accept the beam as a *runway* and switch to
   self-play/REINFORCE fine-tuning once past it (the harness supports
   both).
5. **Engine-first correctness.** The triangle tests pin
   movegen×pathfinder×engine agreement. Anything we should add for
   Guideline-rule edge cases a pro reviewer would probe (spin
   upgrades, 180 kicks, cheese refill parity)?

## Reproducing

```bash
.venv/bin/pytest                       # 274 tests
.venv/py.sh examples/cheese_baselines.py          # baseline table
.venv/py.sh examples/cheese_learner.py --hidden 256 --layers 3 \
    --distill-episodes 24000 --distill-window 3 --dagger-rounds 3 ...
# see examples/server_run.txt for the full server command
```
