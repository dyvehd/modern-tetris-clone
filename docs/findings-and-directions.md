# Findings and Directions — the Complete Record

Status: post-run-8 (2026-09-10). This document is the consolidated
narrative of **every direction this project has taken, what was found,
and where the evidence points next**. It is written for a reviewer who
wants the whole arc without re-reading the commit history. Numbers come
from runs on this repo (local RTX 3050 unless marked *server* — RTX
PRO 6000, 20-core Molab). The per-run ledger with exact numbers lives in
[ai-direction-and-results.md](ai-direction-and-results.md); this
document is the story and the synthesis, the glossary is in
[glossary.md](glossary.md), and the external reviews' own words in
[expert-recommendations.md](expert-recommendations.md).

---

## 1. The one-paragraph summary

A Guideline-correct pure-Python Tetris engine was built first, then a
placement-prediction AI seam on it (agents decide *placements*;
navigation is a shared layer), then a cheese-race learner distilled
from a beam-search teacher through a gated level curriculum. Two
independent expert reviews (both validated to reproduce 100%) found the
teacher was throwing away half its strength (a cutoff rule justified by
a refill that never fires, plus wrong hold transitions — it agreed with
1-ply on 97% of moves) and the student's input encoding collided and
overfit. The v7 branch fixed all of it: the corrected teacher digs
level 10 in 24.2 pieces instead of 43.6, the student now encodes
afterstates with all 5 previews, and the gates enforce reliability,
retention, and fresh retry data. On the corrected stack, the ladder
passes L1 and L2, stops honestly at L3 (student 5.00-5.70 vs teacher
3.68 at 98-100% win) — and the four L3 data points triangulate a
**structural imitation ceiling**: near-optimal L1/L2 play is reactive
(visible in the current board), L3+ play needs the teacher's planning
through hold and preview queue, which per-state cloning cannot express.
The reviews', the encoding's, and the run evidence's converged
direction — **a cost-to-go value network on afterstates trained from
and plugged into the corrected search (expert iteration)** — was then
built and run end-to-end (run V1, 2026-09-10): the pipeline works
(twin-headed V, 5.4M-row collection, V-in-beam, locked manifest), the
net **halves search runtime at equal strength and adds ~1.6 pieces of
strength at equal beam budget** when blended with the linear eval —
but single-visit MC labels calibrate *across boards* without
discriminating *within* one board, so a pure-V beam still collapses
between search horizons and the strict "V@10×3 ≥ linear@40×5"
acceptance is not met. Round 2 is defined by measurement: multi-visit
expected-cost-to-go labels, n-step bootstrapped targets, or both.

## 2. Directions taken, in order, and what each produced

### 2.1 Engine-first (v1 foundation) — taken, validated

**Decision**: build the pure engine (`tetris.engine`) before any AI:
deterministic fixed-60 Hz, zero rendering imports (test-enforced),
bitmask boards, full SRS+180 kick tables, hold, cheese refill, and the
 pygame-ce renderer strictly on top.

**Finding**: this was the project's best early decision. Both reviews
independently confirmed the engine and movement layer sound (review 2
read them in full and found one ordering defect in an edge case —
goal-check-after-spawn — noted, not yet fixed). Every later "is the AI
allowed to do that" question resolved to "ask the engine," and the
triangle test (movegen × pathfinder × real `Game` replay) held through
every stack generation.

### 2.2 Placement protocol, not input control — taken, validated

**Decision**: agents decide *placements*; a shared movegen+pathfinder
layer navigates. `navigate=True` replays real inputs through
`Game.tick`; `navigate=False` teleports+hard-drops (2 ticks/piece) and
is asserted identical on every cheese-relevant outcome.

**Finding**: navigation turned out to be a solved, testable, shared
subproblem, and "placement quality is the only measured skill" became
true by construction (0G, infinite SDF, no ARE). The fast path is what
makes 300-episode gates and 129k-decision collections affordable at
all. Cost: the movement model excludes finite-drop midair stops — a
scoped claim both reviews asked us to state explicitly (now stated in
the glossary and module docs).

### 2.3 Search baselines as the first AI — taken, superseded in v7

**Decision**: hand-written `OnePlyAgent` and `BeamAgent` with a
Dellacherie-lineage linear eval; the beam becomes teacher and gate
reference.

**Finding (the big one)**: the original beam carried two defects —
(1) a "quiescence" rule making every no-clear placement a leaf,
justified by "Jstris refill makes the post-combo board unknowable";
(2) wrong hold/queue transitions (empty-hold children kept the whole
queue; `can_hold` never reset). Both reviews measured the refill
justification as simply false below L10: the engine caps cheese at
`min(9, goal − dug)`, so **no refill ever fires at levels 1-9 and at
most one row at L10** — the rule was deleting every setup move from
the teacher's lookahead. The corrected beam (v7) plays L10 at 24.2
pieces (2.42/line) where the old one took 43.6, and the old beam
agreed with plain 1-ply on **97% of decisions** — distillation had been
cloning a nine-feature linear eval. The old "~4.86 pieces/line is the
beam's ceiling" claim is retracted as an artifact.

Also from this era: CEM eval-weight tuning was built, measured
deterministic, and **did not transfer to fresh seeds** — kept as a
negative result; weight hand-tuning was closed as a direction.

### 2.4 Curriculum learner with distillation warm-start — taken, validated, then re-baselined

**Decision**: cold-start REINFORCE was measured to fail at L1 (sampled
win ~30-39%, greedy ~3%), so every level starts with behavioral cloning
from the beam, then REINFORCE fine-tune, then DAgger rounds, under a
statistical gate with retention and a twice-blocked stop.

**Findings**, each discovered by a failed run and fixed in the next
(see the ledger in ai-direction-and-results.md §results):

- **Data starvation** (run 1): 8k L1-only episodes = 8k decisions for a
  202k-param net → 80% gate win with topouts. Fix: mixed-level
  distillation window (L..L+2).
- **DAgger catastrophic forgetting** (runs 2-3): policy rollouts visit
  ~8:1 junk states; retraining on them wrecked clean-board play
  (distilled 99% → 84.5%). Fix: teacher replay mixed into every pass,
  ¼ learning rate, fresh optimizer.
- **Exact-equality gates** (run 4): demanding win-rate *equality*
  blocked on 99%-vs-100% noise (p≈0.25). Fix: pooled two-proportion
  statistical test.
- **Optimizer-step starvation** (run 5→6): 20k-decision chunks × 60
  epochs = ~360 Adam steps vs the laptop's 9,600. Fix: 10k × 300
  (~4,200 steps) — L2 and L3 passed after this alone. (A correction we
  had to take twice: my own "26×" arithmetic was wrong; review 1's
  ~10.8× was right, and the final-partial-chunk formula is
  `epochs × ceil(decisions/chunk)`.)
- **Retention not enforced** (run 6, live confirmation of review 1):
  L2 and L3 regressed during L4 training and the ladder advanced
  anyway. Fix (v7): retention failure blocks advancement.
- **Retry-data staleness** (run 7b post-mortem): distill seeds were
  deterministic per level, so a blocked attempt re-collected
  *byte-identical* data and re-memorized it. Fix (run 8): an attempts
  counter advancing the seed bands.

The old stack's terminal state (run 6): L1✓ L2✓ L3✓ L4✓ on ties,
with a widening efficiency gap to the teacher per level (+0.01 →
+0.66 → +1.22 → +2.06 pieces) and the retention regressions above —
now understood as artifacts of a weak teacher being imitated
compounding with an encoding that could not express what the teacher
did use.

### 2.5 Policy architecture: score-per-candidate MLP — kept; input changed in v7

**Decision**: the fusion-bot shape — the net scores each (context,
candidate) row; softmax over candidates is the move distribution;
GPU-first torch-native from day one (the 4 GB laptop and the 96 GB
server run identical code).

**Finding**: the architecture survived every review (no reviewer
proposed replacing it; review 1 suggested a shared-board-encoder +
candidate-head variant, review 2 went further — keep the shape but make
the candidate input the **afterstate** and train it as a value). The
*input* did not survive: the old [state | 4×4-pattern | clipped-x]
encoding (274 dims) collided (distinct placements, identical inputs)
and overfit (train 91% while held-out fell to 56%). The v7 afterstate
encoding (256 dims: board-after-lock+clear + outcome + hold flag per
candidate; 5 previews + hold + counters shared) measured 72% held-out
with hole-class errors 11.5% → 0.9% in the review's matched experiment,
and run-7b/8 logs now expose train-vs-held-out directly. The input-dim
guard refuses every old checkpoint by design (their labels are
invalidated).

### 2.6 Server scale-out — taken, validated, twice killed by sandbox expiry

**Decision**: move collection to Molab (forkserver pool, 19 workers),
with per-episode tasks (`chunksize=1`) after measuring round-robin
lists at 3.6× vs episodes at 5.6×, teacher-by-name pickling, payload-
per-worker via pool initializer, and byte-identical parallel/sequential
assertions.

**Findings**: collection scales (~13.5 eps/s); plain `fork` of a torch
parent deadlocks (OpenMP threads — forkserver mandatory); the
scratchpad is time-limited (long steps must be detached); `ps` lies
under gVisor (`/proc` scans instead); a poll SIGINT can kill a
Popen-child runner (own session per launch); and **instances die
mid-run** — twice (HTTP 410), each time costing only wall time because
durable artifacts live in git. The storage-policy lesson is now
institutionalized: `scripts/pull_checkpoint.sh` pulls checkpoints
through the notebook console (gzip+base85, ≤700 KB chunks, dual md5)
with zero credentials on the server.

### 2.7 The v7 repair (branch `ai/v7-teacher-repair`) — taken, fully validated

The reviews' fix order, implemented as three commits with tests:

1. **Corrected teacher**: engine-exact `_hold_transitions` (a pure,
   tested function — review 1's "make the transition a tested function
   instead of duplicating it across branches"), refill-exact leaf rule,
   horizon-consistent comparison. Measured: L10 43.61 → 24.20 at 20×4
   (review 2's 100-seed number: 23.31), L5 14.70 → 9.20, L2's
   counterexample seed-35 4 → 2 pieces. Regression tests replay hold
   branches through real engine clones.
2. **Afterstate encoding** (see 2.5) + held-out split + strict eval
   metric + the checkpoint guard.
3. **Gate hygiene**: win-rate check on every gate path; retention
   enforced; probe on its own band; on-graph entropy (it had zero
   gradient since the first run — review 1's coef-0-vs-100 witness
   inverted and pinned); greedy-eval trace leak fixed.

**Result**: 287 tests green locally and on the server. The stack is
honest: every number it reports since v7 is either held-out or
gameplay-measured.

### 2.8 Runs 7b and 8 on the corrected stack — taken; the ceiling found

**Run 7b** (fresh v7 stack, 181 min): L1 ✓ (1.01/100%, tie at the
~1.0095 theoretical optimum), L2 ✓ (2.49/100% vs the corrected
beam's 2.16 — the teacher got better and the student matched within
noise; retention L1 confirmed under the new enforcement), L3
twice-blocked (5.70, 6.49 vs 3.68). The corrected teacher raised the
L3 bar from the old stack's 4.76 to 3.68 — the same fix that made the
teacher worth copying made it harder to copy.

**Run 8** (resumed from L2; 3× distillation data at L3: 129k decisions;
fresh-retry seeds, 225 min): L3 blocked twice again (5.40, then 5.00
at 98% win). Every intervention measurably helped — fresh seeds:
5.70 → 5.40; 3× volume: → 5.00; held-out peak 65% → 71.9%; the
no-regression probe fired live (rolled back a 100%→96% DAgger
regression) — but the gap stopped shrinking at ~1.3 pieces (36%).

**The synthesis** (this is the document's central finding):

- L1/L2 pass because their near-optimal play is *reactive* — the right
  placement is a function of the current board.
- L3+ requires *planning*: the teacher's edge at L3 (3.68 vs the
  student's 5.00+) comes from sequences through hold and the preview
  queue that a per-state policy imitating decisions cannot represent,
  no matter the data volume. The memorization numbers agree (100%
  train vs 65-72% held-out, still overfitting late).
- Imitation also has a *bar* problem: the gate demands matching a
  teacher that is itself improving — run 7b's L2 passed at 2.49 vs a
  baseline that had moved from 2.34 (old teacher) to 2.16 (corrected).
- Conclusion: **more distillation is not the next lever.** The value-
  network direction is.

### 2.9 Value-network round 1 (run V1) — taken; the ceiling explained, the bar not yet met

**Decision**: implement the reviews' convergence point end-to-end —
`V(afterstate)` trained on Monte-Carlo cost-to-go labels from
corrected-beam rollouts, plugged into the beam as the leaf evaluator
(`tetris.ai.value`, `tetris.ai.valuebeam`, `examples/value_experiment.py`;
commit cde4d9d + the `eval_blend` knob 5e5d104).

**Findings** (5.4M candidate rows across L1/L3/L5/L10 plus a capped-1ply
L10 failure source with 65% genuine failure labels; acceptance manifest
on the locked 800M seed band):

- **Label exactness holds at scale**: L1's 280k rows regress to mean
  q = 1.0049 — the empirical optimum (1 + 1/105); the fail head reads
  ~0 on clean roots and the calibration by level is near-true (root
  means 0.97/3.75/8.28/25.41 vs true 1.01/~3.5/~8.3/~21.5+capped).
- **The structural failure is within-board discrimination**: the
  teacher's chosen move ranks ~16/26 by q̂ with a spread of ~2 pieces
  across *all* candidates of one board — single-visit MC returns carry
  almost no signal about *which* placement is better, only about *which
  board* is better. Consequence, measured: a **pure-V beam collapses
  when no win is inside the search horizon** (L10: 0% win, 11-piece
  topout; L5: 1% win) — between wins the beam random-walks on noise.
- **Blending the linear eval back in (blend .5) restores reliability**
  everywhere (100% win at every level) and beats the pure-linear beam
  *of the same size*: 26.87 vs 28.50 (10×3) and 24.17 vs 24.60 (20×4)
  at L10 — the net adds ~1.6-1.8 pieces of strength at matched search
  budget and halves the runtime of the 40×5 reference (4.82 vs 18.03
  s/episode). At 5×2 linear still wins: too little search for the
  calibration to pay.
- **The strict acceptance fails**: beam 10×3+V (blend .5) does not
  reach beam 40×5+linear (L10 25.89 vs 21.47; paired 17/70/13). The
  net is a *same-budget amplifier*, not yet a *budget compressor*.

**Conclusion**: round 1 established the pipeline (data, twin heads,
V-in-beam, locked manifest, runtime measurement) and produced the
diagnosis the reviews anticipated: single-visit MC labels calibrate
but don't discriminate. Round 2's levers, in evidence order:
(1) **multi-visit labels** — expected cost-to-go from k rollouts per
state (the teacher's 20×4 replays of a state are cheap on the server);
(2) **n-step bootstrapped targets** (review 2's rollout-cost cut, now
a quality lever: pieces spent + V at the horizon);
(3) deeper plies through the V-amplified beam (the blend's win at
equal budget means V can *widen* effective depth, if discrimination
improves).

### 2.10 Value-network round 2 (run V2) — taken; the label fix works, the distribution shift doesn't

**Decision**: fix the *label*, not the architecture — swap the teacher
from corrected-beam to **blockfish** (the dedicated cheese B* bot,
integrated via the cheese-trainer backend layer, merged 5884c1f), whose
per-candidate search ratings and forced-sibling continuations provide
exactly the within-board signal V1's MC labels lacked. The shim's
`BFCandidate.rating` (lower = better; ~10 units ≈ 1 piece — its
`piece_penalty`; terminal traces are exact piece counts) labels every
candidate; `run_episode_from` (4594893) plays exact counterfactual
continuations from a `Game.clone()` — the played line gets its realized
return, the top siblings get *measured* play-out costs, unvisited
candidates get a capped rating-gap prior. Collection: 974k rows in
10 min (blockfish never loses — zero failure labels, a known gap).
blockfish itself was measured on the locked manifest as the **second
acceptance bar**: L1 1.01 / L3 3.50 / L5 7.24 / **L10 17.13, 100% win,
1.7 s/ep** — 20% better than lin40x5 (21.47) at 1/10 the time.

**Findings**:

- **The label fix delivered what it promised**: held-out q-MAE **1.57**
  (V1: 6.64), within-board pair concordance **0.60** (chance 0.50),
  and the teacher's move is V2's argmin 16-30% of decisions by level
  (V1: ~0). The net demonstrably learned *which placement is better on
  this board* where labels exist.
- **Pure V still collapses** (L3 4%, L5 0%, L10 0% win) and blend .5
  gives L10 **24.81** — *no better than V1's 25.89 against the same
  bar*, both bars unmet (21.47 lin, 17.13 bf). Reliability comes from
  the blend; strength did not move.
- **The argmin diagnostic names the residual failure**: across 215 L10
  teacher decisions, in 89 of them V2 *confidently* prefers a different
  move by >0.5 pieces "cheaper" than the teacher's winning move (mean
  gap −0.48, p10 −1.13). The ranking is right on most pairs
  (concordance 0.60) but its worst confusions are catastrophic, and a
  beam compounds argmin errors across ~17 decisions per episode.
- **Root cause is now distribution shift, not granularity**: the
  labels came from blockfish's own *winning* lines — positions where
  its B* values are trustworthy. The V-beam, playing its own (worse)
  lines, visits positions the label distribution never covered, and
  the net's extrapolation there is confidently wrong.

**Conclusion**: round 2 exhausted the *label-quality* lever at fixed
data distribution. Round 3's lever is the one that fixed the policy
net in run 5: **on-policy relabeling** — play the V-beam, then label
its *visited* states with blockfish continuations (DAgger for V).
The infrastructure already exists: `run_episode_from` + the collector
take the player and the labeler as independent agents, so "collect
V-beam games, label with blockfish" is a parameter change, not new
code. A second, cheaper guard: keep blend > 0 as the *reliability*
component (it is what carries win rate) and treat pure-V as the
research signal, not the shipping config.

## 3. Directions considered and rejected (with reasons)

| direction | status | reason |
|---|---|---|
| Input-per-tick AI agents | rejected at design | measures navigation rendering, not placement skill |
| Cold-start REINFORCE as initializer | rejected by measurement | fails L1 at sane budgets (greedy ~3% win) |
| CEM/tuned eval weights | closed, negative result | machinery fine; tuned weights did not transfer to fresh seeds |
| PVP / garbage battles as the objective | deferred | cheese race is verifiable single-player; keep as the scale test |
| Scaling the old stack (more data/net on v6) | rejected by both reviews | the teacher was the bottleneck (97% = 1-ply) and the encoding overfit; run 7b/8 confirm: volume closes only a third of the gap |
| Wider/deeper beams on the *broken* transition model | rejected (review 1) | the model was wrong, not the width |
| Rewriting the engine in a "faster" language | not taken | the engine is the rule authority; compilation belongs at the movegen/eval seam (a recommended, open direction) |
| Adversarial self-play | rejected (review 1) | single-player policy improvement does not need it |
| Mirroring *actions* for augmentation | rejected (both reviews) | SRS kick asymmetry breaks naive reflection; mirror *afterstates* for `V` instead |

## 4. Open directions (ranked by evidence and convergence)

1. **Value-network round 3: on-policy relabeling (DAgger for V)** —
   rounds 1-2 localized the remaining gap precisely. V1 (single-visit
   MC returns) fixed across-board calibration; V2 (blockfish contrast
   labels: per-candidate B* ratings + measured sibling continuations)
   fixed within-board discrimination where labels exist (concordance
   0.60, q-MAE 1.57) — but the labels cover only blockfish's own
   winning lines, and the V-beam's off-distribution visits are where
   it collapses (L10 pure-V 0% win; blend .5 stuck at 24.81 vs the
   21.47/17.13 bars). Round 3 plays the V-beam and labels *its*
   visited states with blockfish continuations — the same
   distribution-shift fix that broke the policy net's ceiling in
   run 5. The collector already separates player from labeler, so this
   is a parameter change. Acceptance: both bars (lin40x5 21.47 and
   blockfish 17.13 at L10 on the locked 800M manifest), pure-V
   reliability ≥ 90% win at every level.
2. **Compiled movegen/eval + batched V-in-beam** — the compute
   enabler: review 2 measured 10-50× collection throughput available
   (numba or a Cobra adapter) and one-forward-pass-per-ply `V`
   batching. Without it, each expert-iteration round at 20×4 costs
   ~20 s/episode/core. Round 1's V-in-beam already batches per ply.
3. **Three seed manifests + non-inferiority gates** — the remaining
   statistical hygiene: train/development/locked-final manifests with
   saved per-seed files; pre-declared margins instead of
   "difference below its own noise"; paired per-seed tests. (Round 1
   already runs paired per-seed comparisons on a locked band.)
4. **Stack-height curriculum axis** — decouple goal from board
   fullness (`cheese_rows` already exists; `_cheese_target`'s cap is
   what binds them — verify the Techmino-style rule against live
   Jstris first, as reviews could not). Makes levels 1-8 measure the
   real skill (digging under a full stack) from the first rung.
5. **Bounded exact solver for L2-L4** — certificates (proven optima)
   and certified regret; both reviews recommend it, review 2 as a
   measurement tool rather than a training source. L1's expected
   optimum is already derived (1 + 1/105 ≈ 1.0095).
6. **Smaller hygiene**: engine goal-check-before-spawn ordering; the
   styles-grid dependency in dug counting; split bag/garbage RNG
   streams; DAgger per-task torch generators; replay budget in
   decisions (not episodes); realized-β logging; reward
   win-vs-failure contract; RL reward-to-go with a value baseline
   (only if REINFORCE returns to the loop).

## 5. What we would tell a reviewer to check first

- The **retractions are marked**: the quiescence rule's justification,
  the "9-row stack" description, the "~4.86 pieces/line beam ceiling,"
  and "275 tests" (now 287). ai-direction-and-results.md carries them
  with history rather than silent edits.
- The **v7 numbers reproduce**: seed-35 (2-piece win, both navigate
  modes), L10 24.2 (paired), held-out 71.9% peak (run 8 log), the
  entropy-gradient inversion, retention blocking — each is a test or a
  committed run record.
- The **negative results are kept**: CEM tuning, cold-start REINFORCE,
  the old-stack runs 1-6 (as the historical baseline), and the
  twice-blocked stops (7b, 8) — the ladder refusing to advance is the
  system working.
- The **honest current state**: L1 ✓, L2 ✓, L3 blocked at 5.00 vs 3.68
  (98% win). The ceiling is diagnosed (imitation of planning by a
  reactive policy), and the next direction is chosen by evidence, not
  fashion.

## 6. Document map

- [glossary.md](glossary.md) — every term, once, with the v7-era
  definitions.
- [ai-direction-and-results.md](ai-direction-and-results.md) — the
  original direction document with the full run ledger (kept
  current; this document synthesizes it).
- [expert-recommendations.md](expert-recommendations.md) — the two
  reviews compiled, with validation status and the merged roadmap.
- [cheese-ai-review.md](cheese-ai-review.md) /
  [cheese-ai-advisor-review.md](cheese-ai-advisor-review.md) — the
  reviews themselves, with repro packages.
- [cheese-ai-learning.md](cheese-ai-learning.md) — the plain-language
  training concepts (pre-v7 in places; the glossary supersedes where
  they differ).
- [onboarding-guide.md](onboarding-guide.md) — the code tour and
  setup.
