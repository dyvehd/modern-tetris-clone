# Project Glossary — Modern Tetris Clone / Cheese-Race AI

Every term and piece of jargon used in this repository's code, docs,
tests, run logs, and discussions, defined once. Written for a new
contributor *and* for the experts who review us — so a term like
"afterstate" or "tie rule" means exactly one thing everywhere.

Organization: game rules → engine internals → harness protocol → search
→ learning → evaluation/gates → operations/run-logs → review jargon.
Terms in **bold** inside definitions have their own entries. Cross-file
conventions (what a word means in *this* repo, even where the wider
literature differs) are called out explicitly.

---

## 1. The game and the task

**Tetris** — the game; this repo clones the modern competitive rule set.

**Guideline** (the Tetris Guideline) — the semi-standard specification
modern clones converge on: SRS rotation, 7-bag randomizer, hold, ghosts,
modern lock delay. "Guideline-correct" is this project's highest
correctness bar.

**Jstris** — the ruleset this clone treats as primary (cheese-race
refill semantics, 5 previews, no ARE in cheese). Where Jstris and
TETR.IO differ (e.g. refill-on-clear), the engine follows Jstris.

**TETR.IO S1 / PPT** — secondary targets (PPT = Tetris Puyo Puyo
Champions). Kept as engine config knobs, not separate code paths.

**Cheese race** — the AI's task: a game mode where the board starts with
garbage rows ("cheese") and the player must clear them; the winner is
whoever clears all cheese with the **fewest pieces**. In this repo:
single-player, seeded, deterministic.

**Level** (task difficulty) — the number of cheese lines to clear.
Level 1 = clear 1 line; level 10 = clear 10. **Not** the engine's
falling-speed level; in the AI context "level" always means the cheese
goal.

**Garbage / cheese row** — a row with exactly one empty cell. The board
starts with `min(9, level)` of them (the cap is reached only at level
9+; levels 1–8 are small open-board puzzles — a review-corrected
statement; older docs wrongly said "a 9-row stack").

**Hole** (cheese context) — the single empty cell of a garbage row.
"Digging" means aligning piece cells under/into holes to complete rows.

**Messiness** — garbage-parameter: the probability (here 100%) that a
newly risen garbage row's hole *moves* relative to the last one. At
100%, consecutive holes never align (a review-verified property used in
an optimality lower bound).

**Piece / tetromino** — the seven shapes: I, O, T, S, Z, J, L
(`PieceType` enum; `I=1` index order in `PIECES`).

**SRS** (Super Rotation System) — the standard rotation/kick tables.
This engine: standard SRS I-kicks, a TETR.IO-style six-test 180° table
(a deliberate hybrid — reviews asked us to name this "rule profile"
explicitly rather than claim dual parity).

**Preview / queue** — upcoming pieces. The harness shows 5 (Jstris
parity). `queue[0]` is *next*. A search node's "queue" includes the
active piece in front (`queue_rest`).

**Hold** — the hold slot. Engine rules mirrored by the search (v7):
plain placement consumes the active piece and **resets hold to usable**
at the next spawn; hold with a filled slot swaps; hold with an *empty*
slot consumes `queue[0]` into hold and spawns `queue[1]`.

**can_hold** — engine flag: hold usable *right now* (folds in whether
hold is enabled at all). One hold per spawn.

**Refill** (cheese) — rising new garbage rows. The engine target is
`min(cheese_rows, goal − dug)` (Techmino-style; the one Jstris-detail
the reviews could not pin to a live source). Consequence (review-verified,
load-bearing): **no refill ever fires at levels 1–9, at most one row at
level 10** — the whole board after any lock is *known* in advance almost
everywhere. This fact killed the old "quiescence" rule (§7).

**Topout / blockout / lockout** — loss conditions: stack reaching the
top, a new piece having no room to spawn, a piece locking entirely above
the skyline. `_game_over` applies the first triggered.

**0G / SDF / ARE / lock delay** — the harness's timing knobs: zero
gravity, infinite soft-drop factor (SDF), zero ARE, huge lock delay —
by design, *placement quality is the only measured skill; timing is a
non-factor*.

**DAS / ARR** — engine input-timing parameters (delayed auto-shift rate
/ auto-repeat rate) — irrelevant to the AI protocol but part of the
human game.

## 2. Engine internals

**Engine** — `tetris.engine`: the pure-Python, deterministic, 60 Hz,
zero-rendering-dependency game state machine. **The rule authority**:
when any AI code disagrees with the engine about legality, the engine
is right and the AI code is buggy.

**Determinism** — same seed + same input sequence ⇒ identical game.
Every test and every seed-band argument depends on it.

**`Game` / `Game.tick(actions)`** — the state machine and its single
entry point: one fixed 60 Hz tick, optionally consuming a set of
`Action`s (inputs).

**Board / rows** — the playfield as a list of 40 ints (20 visible rows
+ 20 buffer), each int a 10-bit **bitmask** row (bit x = column x).
`FIELD_H=40`, `FIELD_W=10`; the visible field is the bottom 20.

**`styles` grid** — the per-cell rendering/style layer. *Load-bearing*
(reviews): dug counting reads it (a cleared row is cheese iff any cell
carries style −2). An open architectural complaint: the rule authority
depending on a rendering structure.

**`SevenBag`** — the randomizer: each bag of 7 deals all seven pieces
shuffled (per-bag uniform; the *old docstring's* "every window of 7"
claim was false — corrected). Bag and garbage share one RNG stream (a
known caveat: changing the level changes the piece stream for a seed).

**Spin / T-spin** — locking a piece after a rotation kick that moved it
(its 3-corner rule). The movegen carries spin verdicts; spin class is
cosmetic in cheese (no spin bonus exists in the task) but must be
*reported* correctly for parity tests.

**Ghost** — the hard-drop preview of a piece's landing position.

**Cheese counters** — `cheese_on_board` (garbage rows currently on the
field), `cheese_dug` (cheese lines cleared so far), `goal_lines`
(the level target; the win condition is `dug ≥ goal`).

## 3. The harness protocol (AI seam)

**Harness** — `tetris.ai.cheese`: the episode layer between engine and
agents. Its contract is the project's central design decision.

**Placement** — a reachable resting position of a piece:
`(piece, rot, x, y, spin_class, cells)`. `x`/`y` are bounding-box
coordinates and may be negative (a box overhangs the field edge while
its cells stay in bounds — the source of the old encoding collision).

**Agents decide placements, never inputs** — the position-prediction /
navigation decoupling: an agent's `decide(obs) -> Decision` picks a
placement; a shared navigation layer turns it into inputs. Every agent
(random, search, learned) shares it.

**`Obs` (observation)** — the frozen snapshot an agent may know:
rows, active piece, hold, `can_hold`, queue (5 previews),
cheese counters, goal, `pieces_placed`, `allow_180`, and (v7) `stack`
— the env's cheese cap, needed by the search's refill-exact leaf rule.

**`Decision`** — `(placement, hold_flag)`: lock this placement, holding
first if flagged (the placement must then be for the piece hold brings
out).

**`candidate_moves(obs)`** — every legal decision at an observation:
each reachable placement of the active piece plus each of the
hold-brought piece. **The one action-space definition**, shared by
search, policy, and all data collection — a v7 test pins that the beam's
root branches are exactly this list.

**navigate=True / navigate=False** — the two application modes. True:
pathfinder converts the placement to a shortest input sequence,
replayed through `Game.tick` (full fidelity; produces per-piece input
logs). False (the fast path): teleport the piece to its resting
position and hard-drop (2 ticks/piece) — asserted *identical on every
cheese-relevant outcome* (dug, refill, win; only cosmetic spin labels
skip). All data collection and evaluation uses the fast path.

**`run_episode` / `run_batch`** — one seeded episode / N sequential
seeded episodes; `BatchResult` carries per-seed pieces and reasons
(wins counted by reason `"cleared"`).

**Episode / rollout** — one full game, fresh board to win (reason
`cleared`), topout, or piece cap (400 placements, reason `capped`).
"Rollout" = an episode played *to collect training data*.

**InvalidDecision** — the harness exception for an illegal agent
output (unreachable placement, wrong hold piece) — errors are never
silently swallowed.

**Movegen** (`tetris.ai.movegen`) — BFS over the movement graph
yielding every reachable resting placement, with spin verdicts.
Movement model: 0G + infinite SDF (finite-drop midair stops are
deliberately excluded — claims of "exhaustive legal placements" are
scoped to this model; reviews flagged the wording).

**Pathfinder** (`tetris.ai.pathfinder`) — shortest input sequence
spawn → target placement.

**The triangle test** — the cross-validation heart of the correctness
story: movegen's placement × pathfinder's path × a real `Game.tick`
replay must agree exactly (cells, lines, spin class) — a three-way
soundness check on fixtures. Reviews noted its limits (shared
`successors` between movegen and pathfinder can hide a common missing
edge; clear/spin branches compare labels, not full afterstates) — v7
added the search-side analogue: hold branches replayed through engine
clones.

## 4. Search

**Search agents** — `OnePlyAgent` (1-ply), `BeamAgent` (beam): hand-
written placement scorers; the "hand-written AI" baseline layer.

**Eval** (`tetris.ai.eval`) — the linear board-scoring function:
`EvalWeights` (holes, covered, height, bumpiness, row/col transitions,
wells, lowest-row holes, lines, win) × features. Dellacherie-lineage,
hand-set. Deliberately *not* tuned by search (CEM tuning was measured
not to transfer).

**1-ply / `OnePlyAgent`** — evaluates each candidate once, picks the
best. Essentially the eval's argmax.

**Beam search / `BeamAgent(width, depth)`** — keeps the best `width`
plans per ply, searches `depth` pieces deep over the visible queue.
`beam20x4` = width 20, depth 4 — the default teacher/reference.

**The v7 teacher repair** (branch `ai/v7-teacher-repair`) — the v7
rewrite of `BeamAgent` implementing both reviews' corrections:
(1) engine-exact hold/queue transitions via `_hold_transitions`
(empty hold consumes `queue[0]`; `can_hold` resets after every lock);
(2) the refill-exact leaf rule; (3) horizon-consistent comparison.
Measured: L10 43.61 → 24.20 pieces (20×4), L5 → 9.20, seed-35 → 2-piece
win. `docs/advisor-review-repro/beam2.py` is the validated reference
implementation.

**Refill-exact leaf rule** (v7) — a placement is a search *leaf* only
when the engine would actually refill after it: a no-clear lock with
cheese below `min(stack, goal − dug)`. Since no refill ever fires below
L10, setup moves are fully planned; at L10 the leaf boundary is the one
genuinely unknowable position.

**Horizon-consistent comparison** (v7) — plans are compared at a common
horizon: wins ordered by fewest pieces first (a win is a win; the
shortest is best); otherwise by eval + lines-weight × cheese dug *along
the plan*, so stopping depths are comparable. The old code mixed
depth-1 and depth-4 leaf scores.

**Quiescence rule** (retracted, historical) — the *old* v6-and-earlier
rule: "a no-clear placement is a leaf because refill makes the future
unknowable." Both reviews proved the justification false below L10 (no
refill exists there) — the rule was actually deleting every setup move
from the teacher's lookahead. Kept in docs as a retraction with history:
it did fix a real phantom-win bug, at ~10× the cost.

**`lock_and_count`** — the pure lock-and-count helper used by search
and encodings: merge placement onto a row copy, return (rows, lines,
dug). Its documented approximation: a cleared row counts as cheese iff
it lies in the bottom `cheese_left` region (exact while the region is
pure; mixed rows match the engine's contains-garbage style test).

**Node / `_Node` / frontier / leaves** — beam internals: a plan-prefix
node (rows, hold, can_hold, queue_rest, dug, cheese_left, score, first
move, pieces, refill flag); the frontier is the expandable set; leaves
are refill-bounded or terminal.

**`queue_rest`** — a node's remaining piece stream: the active piece
followed by the visible previews (`(obs.active,) + obs.queue` at the
root, sliding forward with each placement).

**Phantom continuation** — the old bug's signature: plans built on a
piece that will not exist (empty-hold children planned `queue[0]`
twice). A review term now pinned by a regression test.

**Transposition table** — a *recommended* (unbuilt) beam dedup keyed on
(rows, hold, queue index, can_hold): hold/no-hold orderings reaching
identical boards. From the advisor review.

**Chance node** — a *recommended* (unbuilt) search device for the L10
refill boundary: at 100% messiness the new hole is uniform over the 9
non-last columns; average (or sample 3 of) the children. Cold Clear's
"speculation" mechanism applied to garbage.

## 5. Learning

**Policy / `PolicyNet` / `PolicyAgent`** — the learned placement-choosing
network. *Terminology rule: "policy", never "learner", in code and
docs* (the older docs' "learner" means the same thing). A
**score-per-candidate MLP** (the fusion-bot shape): one score per
(board-context, candidate) row; softmax over candidates is the move
distribution; tanh hidden layers, scalar head, glorot init.

**Score-per-candidate** — the architecture: the net never sees a board
alone; it scores each candidate *in context* of it. Softmax over the
candidates of one decision = the policy's move probabilities.

**Afterstate** (v7, the current encoding) — the board *after* a
candidate placement locks and clears, plus that lock's outcome (lines,
cheese dug, win flag) and hold flag. Each candidate row encodes its own
afterstate; the shared context row encodes all 5 previews, hold, active,
and cheese counters. INPUT_DIM 256 (v6 and earlier: 274, from a
4×4-pattern + clipped-x parametrization). Why (advisor review, matched
experiment): held-out agreement 56% → 72%, hole-class errors 11.5% →
0.9%; the old encoding overfit (train 91% while held-out fell). Also
kills the x-clipping collision by construction and is exactly the input
a **value function** needs.

**Encoding collision** (fixed, historical) — the old encoding clipped
a placement's bounding-box x into 0..9, so distinct placements (e.g.
vertical-I x = −2/−1/0) produced *identical* inputs. Real but
second-order: measurements showed it touched <1% of teacher labels and
changed the argmax 0.0% of the time (the advisor's re-weighting of
review 1's P1).

**Held-out split** (v7) — distillation reserves the last 10% of
collected decisions for evaluation only; the ladder logs train and
held-out accuracy side by side. Exists because run 6's "96% accuracy"
was training-set accuracy; the true held-out number was unknown and
lower.

**Strict (tie-unsatisfying) accuracy** — argmax must *equal* the
teacher index exactly; ties count against. The training-time metric
(`_group_accuracy`) counts ties as correct — fine on trained nets
(ties vanish), wrong for eval; `_group_accuracy_strict` is the eval
metric.

**Distillation / behavioral cloning (BC)** — supervised training on
teacher decisions: cross-entropy toward the teacher's candidate index.
The warm-start stage of every level. `TeacherRecorder` is the
pass-through agent that records (encoding, teacher index) pairs.

**Teacher** — the search agent whose decisions are copied (currently
the corrected `beam20x4`). *vs.* **reference** — the same agent *as a
gate opponent*. One agent, two roles, two words, on purpose.

**DAgger** (Ross et al. 2011) — the state-distribution fix for BC:
roll out with the *policy* (mixture play), label every visited state
with the teacher's choice, retrain. Round structure: β-decayed mixture,
per-round parallel collection, teacher **replay** mixed into every
training pass.

**β (beta) / mixture** — DAgger's knob: probability of playing the
teacher's move during collection (labels are always the teacher's).
Rounds decay 0.8 → 0.48 → 0.29 by default (a review noted nominal-β vs.
realized-rate should both be logged).

**DAgger safeguards** (v6, all three measured necessary): **replay**
(50k teacher decisions mixed into every pass — clean-board behavior
rehearsed), **¼-lr fine-tune** (`dagger_lr = distill_lr/4`, fresh
optimizer), **no-regression probe** (greedy probe before/after on the
probe band; regression beyond statistical noise ⇒ pre-DAgger weights
restored). The probe fired live for the first time in run 8 attempt 2.

**REINFORCE** — policy-gradient training from episode returns. In this
stack: a *fine-tune* stage, not an initializer (cold-start REINFORCE
failed at L1: sampled win ~30-39%, greedy ~3%). v7 fixed its entropy
bonus (see below). Reviews recommend holding it out of the loop until
a value baseline exists.

**Return / shaped return** — the episode quality scalar:
`−pieces + α·lines_dug + 50·win` with potential-based shaping whose
per-move terms telescope to that closed form. Review caveat (open):
the arithmetic can prefer a 20-piece failure (−20) over a 100-piece L10
win (−40) — a reliability-vs-efficiency contract that needs an explicit
decision (proposed: pieces for wins, cap+penalty for failures).

**Entropy bonus** — the exploration regularizer in the REINFORCE
loss. History: computed as a detached float since the first run ⇒
*zero gradient* (both reviews' witness; coef 0 vs 100 gave bit-identical
weights). v7: recomputed **on-graph** from the update's own segmented
logits — the witness now yields different weights, pinned by test.

**Segmented softmax** — the batched update's core: each decision's
softmax runs over *its own* candidate group within one concatenated
batch (scatter_reduce logsumexp), so differently-sized decisions train
together.

**Curriculum / ladder** — the training schedule: per level, distill
warm-start → REINFORCE → DAgger rounds → **gate**; advance on pass,
retry on block, stop after two consecutive blocks ("twice-blocked").

**Mixed-level distillation / `distill_level_window`** — the warm-start
collects from levels `L..L+window-1` (default 3) sharing the episode
budget, because a 1-piece L1 episode carries one decision (structural
data starvation — measured in run 1).

**Attempts / retry-freshness fix** (run 8, b2c5690) — a Curriculum
counter bumped on every block; distill/DAgger seed bands advance by
`+100M×attempts`, so a *retry* collects genuinely fresh data. Before:
attempt 2 re-collected byte-identical data and re-memorized it.

**Checkpoint** — a saved policy (JSON: architecture + plain-list
state_dict + input-dim meta). `cheese_policy_L{N}.json` after each
gate-passed level; `L{N}_distilled/reinforced/daggered.json` per stage.
**Input-dim guard** (v7): `load_policy` refuses mismatched checkpoints
— old-stack files are invalid by construction, and silent misuse is
impossible.

## 6. Evaluation and gates

**Gate** — the statistical pass/fail deciding level advancement:
learner vs reference on the same fresh seed band (300 episodes/side
default), greedy evaluation, fast path.

**GateStats** — `(mean_pieces, ci95, win_rate, episodes)`; means are
computed **over wins only** ("win-conditioned") — a documented caveat:
abandoning hard seeds changes which seeds enter the mean, which is why
reliability is checked separately.

**Strict / tie (gate rules)** — strict: learner mean + CI sits below
the reference mean (statistically separated improvement). Tie: mean
difference within the *combined* noise bands AND win rate not
statistically worse. Necessary at L1 (the reference plays the
theoretical optimum — strict improvement is impossible; matching the
optimum is mastery). Both branches require the win-rate check since v7.

**Win-rate check / `_pooled_win_se`** — the two-proportion pooled
standard error test guarding both gate branches (v7: previously only
the tie branch; review 1's 1%-win-lucky-mean witness is now blocked and
pinned).

**Twice-blocked** — the ladder stops after two consecutive blocks at
one level. The honest terminal state of runs 7b and 8.

**Retention** — re-gating every easier level after an advancement.
v7 change: a retention failure **blocks advancement** (the level
retries under the same twice-blocked rule); through run 6 it was merely
logged — the live confirmation of review 1's finding (L2 2.99 vs 2.34,
L3 5.75 vs 4.76 regressed during L4 training and the ladder advanced
anyway).

**Probe** — the DAgger no-regression evaluation (greedy, on the probe
band). v7: moved to its own seed band (`gate_seed0 + 1M`); it previously
shared the gate's seeds — model selection on validation data.

**Seed bands** — structurally disjoint numeric ranges: training <100M;
distill 100M + 1M·level (+100M·attempts); DAgger 200M + 1M·level
(+100M·attempts, round 50 reserved for replay); gate 900M+; probe
gate+1M. Gates never see a training seed. Reviews' fuller ask
(three *manifests* with saved per-seed files and overlap assertions)
is open.

**Paired comparison** — both gate sides play the *same* seeds; reviews
recommend keeping per-seed results and using paired tests (today only
means/CIs/win-rates are compared).

**Non-inferiority margin** — a reviews' (open) recommendation: the tie
rule should accept "difference below a *pre-declared* acceptable
margin," not "difference below its own noise" — the latter gets easier
as variance grows (299×1-piece + 1×40-piece passes vs a 1.01 baseline).

**Hole-class error** — an advisor-review error category: the chosen
move scores ≥20 eval points below the teacher's — i.e. catastrophic
(burying a hole) vs cosmetic disagreement. The afterstate encoding cut
these 11.5% → 0.9%.

**`J` / `R` (material audit)** — conservation accounting for a won
episode: with G garbage rows, P pieces placed, J player-only cleared
lines, R residual cells: `4P = G + 10J + R`. The advisor's diagnostic:
the old beam's waste showed up as J=15.8 junk lines/game vs the
corrected beam's 7.8. Cheap to log per episode; recommended as a
standard metric.

**Pieces/line** — mean pieces per cheese line: the headline efficiency
statistic (corrected beam20x4 at L10: 2.33-2.36; 80×6: 2.08).

**Win rate** — fraction of episodes cleared. First-class metric
alongside pieces (reviews: reliability should be lexicographically
primary, or at least an explicit tradeoff).

**Held-out / training accuracy** — generalization vs fit on the
training set. The project's most expensive lesson: run 6 reported
training accuracy only; run 7b's L1 showed 100% train vs 77.8% held-out.

**Memorization / imitation ceiling** — the convergent diagnosis of
runs 7b/8: the net fully fits its training set (100% train) while
held-out stalls (~65-72%), and the gameplay gap to the teacher stops
shrinking with data volume (L3: 5.70 → 5.40 → 5.00 across fresh-seeds
and 3×-data interventions, vs baseline 3.68). Structural cause: L1/L2's
optimal play is reactive (visible in the current board); L3+ needs the
teacher's *planning through hold+queue*, which per-state cloning cannot
express.

## 7. Operations (Molab / runs)

**Molab** — the remote marimo-notebook GPU sandbox used for server runs
(RTX PRO 6000, 20 cores, torch cu130). Paired via the marimo-pair
skill's `execute-code.sh` (HTTP + auth token).

**Scratchpad** — the execute-code.sh execution environment: server-side
time-limited, stateless between calls (module reloads needed), console
output truncated at ~2-4 MB (the reason the checkpoint puller chunks at
700 KB).

**Instance expiry / HTTP 410 Gone** — Molab sandboxes die server-side
after some hours; two runs have been killed by it (run 7's L2, run-8
era's L2 second attempt). The standing ops rule: durable artifacts live
in git, the sandbox is compute-only.

**Storage policy (Aug 2026)** — Molab retains only file-browser uploads
and `mo.persistent_cache` across shutdowns; everything script-written
(git clones, checkpoints, logs) is wiped on expiry.

**`pull_checkpoint.sh`** — the credential-free checkpoint-preservation
tool: the server snapshots a checkpoint (frozen gzip, md5s) and serves
gzip+base85 chunks ≤700 KB through the notebook console; the local side
reassembles, dual-md5-verifies, and commits. Built because GitHub write
credentials must not sit on a shared sandbox, and the console truncates
above ~2 MB. (Base85 alphabet gotcha: `|`, `@`, `:` are all *in* the
alphabet; the wire format delimits on `.`.)

**Detached runner** — training runs are launched `nohup` + own session
so a dying connection never signals them (learned when a poll SIGINT
killed run 2).

**gVisor / `/proc` scans** — the sandbox's `ps` is unreliable; liveness
is checked via `/proc/{pid}/cmdline`, log mtime, and GPU memory.

**Forkserver pool** (`parallel.py`) — the multiprocessing context for
parallel collection: plain `fork` of a torch parent deadlocks (OpenMP
threads — measured); per-episode tasks with `chunksize=1` load-balance
(3.6×→5.6× on 15 cores; 13.5 eps/s on 19 Molab workers); the teacher
ships by *name* (pool args are pickled, lambdas aren't); the policy
payload ships once per worker via the pool `initializer`.
Parallel-vs-sequential output is asserted byte-identical.

**Run numbering** — v5 laptop runs; server runs 1-6 (the old stack),
7/7b/8 (the v7 stack). The authoritative ledger is the run table in
[findings-and-directions.md](findings-and-directions.md).

**Alarm webhook** — the user's standing ops instruction: if Molab dies
and work can't continue, GET the MacroDroid alarm URL (phone alarm).
Used at both expiries.

## 8. Review and direction jargon

**Review 1 / review 2** — the two independent expert reviews (2026-09-09,
at `ee32fb0`), both fully validated by re-running every measurement:
[cheese-ai-review.md](cheese-ai-review.md) (the defect ledger, 14
witnesses) and [cheese-ai-advisor-review.md](cheese-ai-advisor-review.md)
(the measured re-weighting, repro package). Their terminology that has
entered the project's vocabulary:

**Witness** — an executable observation script demonstrating a defect
(`docs/cheese-ai-review-repro.py`, `docs/advisor-review-repro/e*.py`).
A witness *reproduces an observation*; exit success ≠ correctness.
Several are now inverted into regression tests.

**P1 (review sense)** — "fix before trusting the research conclusions,"
not a security severity. Six were co-equal in review 1; review 2's
measurement re-weighted them (teacher cutoff ≫ everything else).

**The 97% finding** — the old beam agreed with plain 1-ply on 97% of
decisions: distillation had been cloning a nine-feature linear eval.
The single reframing fact of the whole project history.

**Expert iteration** — the recommended direction: a corrected search
generates cost-to-go labels; a **value network** `V(afterstate)` regresses
on them; `V` becomes the search's leaf evaluator; the improved search
generates better labels; repeat. "Single-player" (no adversarial
self-play needed). Imitation's ceiling disappears because the search
can improve on `V`.

**Cost-to-go / value network / `V(afterstate)`** — remaining-pieces-
to-win estimated by a net on afterstates (failures tracked as a
separate probability head, so efficiency cannot hide unreliability).
Conceptual update: `Q(s,a) = 1 + E[V(next)]`, `V(goal) = 0`.

**Bootstrapped (n-step) targets** — pieces spent + `V` at the search
horizon (once `V` is decent) — a recommended rollout-cost cut.

**Soft targets / regret-aware labels** — the reviews' alternative to
exact teacher-index labels: near-equivalent placements should get
near-equal targets (softmax over root children's search values, or a
hinge on the value gap); exact-index accuracy punishes harmless ties.

**Clairvoyant reference** — a lower-bound search that sees past the
preview limit; useful as a bound, *not* a fair teacher.

**Bounded exact solver** — exhaustive/iterative-deepening search with
transposition keys for L2-L4 within the visible queue, producing
certificates (proven optima) or bounds. Recommended (both reviews) as a
measurement tool; review 2 puts it after the teacher fix.

**Stack-height axis** — the reviews' curriculum redesign: today's
"level" conflates goal size with board fullness (levels 1-8 are open
puzzles); separate `goal` from `cheese_rows` so the real skill —
digging under a full stack — is trained from the first rung.

**Difficulty-stratified data** — tag states by cover depth of the next
hole and sample to flatten the distribution (turns the "8:1 junk-state"
problem into sampling weights).

**Search node budget as curriculum** — review 2's alternative ladder:
train `V` until beam-5×2+`V` matches beam-80×6+linear-eval on a locked
manifest — a measurable "the net learned what the search knows."

**Three manifests** — the reviews' seed hygiene: train / development
(probes, selection) / locked final. Today: numeric bands (with the
attempt fix), per-seed files and overlap assertions still open.

**Rule profile** — the named, versioned statement of exactly which
game's rules a benchmark ran under (rotation profile, spawn/topout,
hold, refill, messiness, previews, timing) — the reviews' ask for
benchmark metadata.

**v5 / v6 / v7 stacks** — the learner-stack generations: v5 (laptop,
128×2, 62k decisions), v6 (server scale: parallel collection, DAgger-in-
ladder, statistical gates — the runs 1-6 stack), v7 (everything the
reviews fixed: corrected teacher, afterstate encoding, gate hygiene —
the runs 7b/8 stack). Old-stack checkpoints are load-refused by the
input-dim guard.

---

## Quick-reference index

| Term | Section |
|---|---|
| cheese race, level, garbage, hole, messiness | 1 |
| engine, tick, board, styles, SevenBag, refill | 2 |
| placement, Obs, Decision, candidate_moves, navigate, movegen, pathfinder, triangle | 3 |
| eval, 1-ply, beam, v7 repair, leaf rule, horizon, quiescence (retracted), lock_and_count, chance node | 4 |
| policy, afterstate, encoding, distillation, teacher, DAgger, β, REINFORCE, entropy, curriculum, attempts, checkpoint | 5 |
| gate, strict/tie, retention, probe, seed bands, paired, non-inferiority, J/R audit, pieces/line, memorization ceiling | 6 |
| Molab, scratchpad, expiry, storage policy, pull_checkpoint, detached runner, gVisor, forkserver, runs, alarm | 7 |
| reviews, witness, P1, 97%, expert iteration, V(afterstate), soft targets, solver, stack-height axis, manifests, rule profile, vN stacks | 8 |
