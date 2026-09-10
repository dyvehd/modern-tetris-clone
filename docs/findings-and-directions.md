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
The evidence, the reviews, and the encoding all converge on the same
next direction: **a cost-to-go value network on afterstates, trained
from and plugged into the corrected search — expert iteration**.

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

1. **Cost-to-go value network on afterstates (expert iteration)** — the
   convergence point of both reviews, the run-7b/8 diagnosis, and the
   v7 encoding's design. Train `V(afterstate)` on remaining-pieces
   labels from corrected-beam rollouts; use `V` as the beam's leaf
   evaluator; iterate. Removes both the imitation ceiling (the search
   can improve on `V`) and the label-arbitrariness problem (near-ties
   get near-equal targets). Acceptance (review 2): beam-10×3+V ≥
   beam-40×5+linear-eval on a locked manifest.
2. **Compiled movegen/eval + batched V-in-beam** — the compute
   enabler: review 2 measured 10-50× collection throughput available
   (numba or a Cobra adapter) and one-forward-pass-per-ply `V`
   batching. Without it, each expert-iteration round at 20×4 costs
   ~20 s/episode/core.
3. **Three seed manifests + non-inferiority gates** — the remaining
   statistical hygiene: train/development/locked-final manifests with
   saved per-seed files; pre-declared margins instead of
   "difference below its own noise"; paired per-seed tests.
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
