# Cheese-Race AI: Direction and Preliminary Results

Status: work in progress. Written for an expert review of the
direction; every number here was measured on this repo (local RTX 3050
unless marked *server* — NVIDIA RTX PRO 6000, 96 GB, 20-core Molab
instance).

Companion documents: [onboarding-guide.md](onboarding-guide.md) (code
tour) and [cheese-ai-learning.md](cheese-ai-learning.md) (plain-language
training concepts).

## The task

A cheese race: the field starts with `min(9, level)` garbage rows (one
hole per row, 100% messiness — Jstris cheese rules; the 9-row cap is
reached only at level 9+, so levels 1–8 are small open-board puzzles,
not full downstacking under a stack), and the objective is to clear
`level` cheese lines with **the fewest pieces**. 0G, infinite SDF, no
ARE: placement quality is the only measured skill. The curriculum
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
  (level 1's expected optimum is ~1.0095 — a 1/105 fraction of seeds
  needs 2 pieces when S/O meet hole-column 0 or Z/O meet hole-column 9;
  the beam plays that optimum).
- **Level-1 ground truth was extracted from the movegen, not guessed**:
  I/J/L/T dig any hole column; S cannot open column 0, Z cannot open
  column 9, O never digs alone. One-piece wins exist on ~99.0% of seeds
  (500-seed sweep: 496 × 1-piece — consistent with the 1/105
  expectation).

### 3. Reference / teacher

- Beam search (width 20, depth 4) with a hand-tuned eval, plus 1-ply and
  a lexicographic greedy. The beam is the teacher *and* the gate
  opponent. Its measured pieces-per-line: ~4.86-5.12 across levels,
  100% win at low levels. *(Review finding: this ceiling is an artifact
  of the quiescence rule below — a corrected 20×4 beam plays L10 at
  2.33 pieces/line, 80×6 at 2.08. The original beam also agrees with
  plain 1-ply on 97% of decisions, so it was never much of a search
  teacher.)*
- ~~A quiescence rule was required for search correctness~~ **retracted
  by review**: the engine caps cheese at `min(9, goal − dug)`, so **no
  refill ever fires at levels 1–9** (and at most one row at L10) — the
  "unknowable post-combo board" the rule guarded against does not exist
  in this harness. The rule actually removed every setup move from the
  teacher's lookahead. Kept historically: the rule did fix a real
  phantom-win bug (level-1 plans crediting wins through no-clear
  moves), but at 10× the cost.

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
| server run 5 | + statistical win-rate gates (pooled two-proportion) | **L1 PASSED** (100% win / 1.13 vs beam 1.01, statistical tie; DAgger converged, probe improved it); L2 twice-blocked (2.71/92%, then 3.33/99% vs 2.34) |
| server run 6 | + optimizer-step fix (chunk 10k × 300 epochs ≈ 3,900-4,200 Adam steps vs run 5's ~360), resumed from run-5 L1 net | **L2 PASSED** (100% win / 2.35 vs 2.34 — tie; probe-confirmed DAgger gain 93%/2.20→100%/2.33). **L3 PASSED on attempt 2** (99% win / 5.42 vs beam 4.76 — tie *within noise but 0.66 worse in mean*: the wide L3 variance makes the tie band generous; exactly the pattern review 1 §4 criticizes). **L4 PASSED** (99% / 10.92 vs 9.70 — tie) **but retention REGRESSED at L2 (2.99 vs 2.34) and L3 (5.75 vs 4.76), and the ladder advanced anyway** — review 1's "retention is logged, not enforced" finding confirmed live: the policy forgets earlier levels as it climbs. L5 in progress at time of writing |

### The failure→fix ledger (what a reviewer should weigh)

Every design change in the learner stack came from a measured failure,
not a hunch: pooled encoding hid the hole → full resolution; L1-only
distill starved the net → mixed-level window; DAgger forgot clean-board
play → replay + fine-tune lr + no-regression probe; exact-equality tie
rule blocked on noise → two-proportion tests; poll SIGINT killed a
detached runner → `start_new_session`; a 4 GB GPU OOM'd → chunked
training (now a server-tunable chunk size).

## Open questions we'd value review on

*(Both external reviews have since answered most of these — see the
validation section below. Kept for the record, with pointers.)*

1. **DAgger probe strength.** Answered by both reviews: the probe's real
   defect is shared seeds with the gate, not its size — give it a
   development manifest of its own.
2. **Distillation accuracy ceiling.** Answered by measurement (expert
   2, reproduced): the dominant levers are the afterstate input
   encoding and held-out (not training) accuracy; the current
   encoding demonstrably overfits.
3. **Gate stringency at scale.** Answered: use a pre-declared
   non-inferiority margin on paired per-seed differences; a
   failure-to-reject is not evidence of mastery.
4. **Search as the permanent teacher.** Answered: the beam's ~4.9
   pieces/line was an artifact of its quiescence rule, not a property
   of beam search — a corrected 20×4 beam plays L10 at 2.33, and
   80×6 reaches 2.08. The direction is now a cost-to-go value network
   on afterstates feeding the corrected search (expert iteration).
5. **Engine-first correctness.** Partially answered: add search-transition
   tests vs `Game`, full-afterstate checks after clears, goal-resolution
   ordering at the final lock, refill-boundary assertions (no refill
   below L10; one row at most at L10).

## Validation status (external reviews, 2026-09-09)

Two independent expert reviews were validated by re-running every
measurement; all reproduce:

- **[cheese-ai-review.md](cheese-ai-review.md)** (+ its witness script,
  14/14 observations reproduce) — the defect ledger: beam hold/queue
  transition bug; quiescence rule justified by a refill that never
  fires below L10 (invalidating the "~4.86 pieces/line" framing);
  encoding collisions; zero-gradient entropy; gate statistics; retention
  not enforced; goal-before-spawn ordering; and more.
- **[cheese-ai-advisor-review.md](cheese-ai-advisor-review.md)** (+ its
  repro package, all numbers re-verified) — the measured weighting:
  the teacher cutoff fix alone is worth ~20 pieces per L10 game
  (43.61 → 23.31 at 20×4, 91/100 paired seeds better); the current beam
  agrees with 1-ply on 97% of decisions (distillation has been cloning
  a linear eval); the afterstate encoding raises held-out agreement
  56% → 72% with 12× fewer hole-class errors; the x-clipping collision
  touches <1% of labels (real, but not the bottleneck).

**Consequence for these results**: run 6's numbers are the historical
baseline of the *old* stack (old teacher, old encoding, old gate
statistics). Its training accuracies are training-set numbers — held-out
accuracy was never measured. Per both reviews, do not scale this stack
further; the fix order is teacher transitions + non-clearing expansion
first, afterstate encoding + 5 previews second, seed/gate/entropy
hygiene third, then a cost-to-go value network on afterstates feeding
the corrected search.

## Reproducing

```bash
.venv/bin/pytest                       # 275 tests
.venv/py.sh examples/cheese_baselines.py          # baseline table
.venv/py.sh examples/cheese_learner.py --hidden 256 --layers 3 \
    --distill-episodes 24000 --distill-window 3 --dagger-rounds 3 ...
# see examples/server_run.txt for the full server command
```
