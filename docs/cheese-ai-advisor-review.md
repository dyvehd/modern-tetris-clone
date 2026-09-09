# Cheese-Race AI — Independent Advisor Review

Reviewed 2026-09-09 at commit `ee32fb0`, by a second reviewer working
independently of [cheese-ai-review.md](cheese-ai-review.md). I read the
engine, movegen, pathfinder, harness, search, policy, distillation,
DAgger, curriculum and parallel modules in full, ran the test suite
(275 passed, 389 s), and ran seven measurement experiments on this
machine (16 cores, RTX 3050). Every number below comes from those runs;
the scripts are in [advisor-review-repro/](advisor-review-repro/). Where
I cross-checked the other review, I say so explicitly in §6.

## 0. Summary

**The engine and the movement layer are sound. The learning stack is
built on a teacher that is far weaker than it needs to be, and on a
policy input encoding that generalizes poorly. Both are cheap to fix,
and the fixes should precede any further large training run.**

The two measurements that should reorder the project's priorities:

1. **The beam teacher throws away about half of its own strength.**
   Its "quiescence" rule (never look past a placement that clears no
   line) is justified by cheese refill — but the engine caps the cheese
   at the lines still needed, so **no refill ever happens at levels
   1–9, and exactly one row can ever be refilled at level 10**. With
   non-clearing moves expanded and hold/queue transitions made
   engine-exact, the same 20×4 beam plays level 10 in **23.3 pieces
   instead of 43.6** (2.33 vs 4.36 pieces per line; better on 91 of 100
   seeds, worse on 9). Widening to 80×6 reaches **2.08 pieces/line**.
   The "~4.86 pieces/line" ceiling in the direction document is an
   artifact of this rule, not a property of beam search.

2. **The current beam agrees with plain 1-ply on 97% of decisions.**
   The lookahead almost never changes the move, so distillation has been
   cloning a nine-feature linear eval. The learned MLP cannot even do
   that well because its input (board + 4×4 pattern + x one-hot) forces
   it to re-derive "where does this piece land" from scratch. Encoding
   the **afterstate** (board after the lock and clear) instead — same
   net, same data, same optimizer steps — raises held-out agreement from
   56% to 72% and cuts hole-burying errors from 11.5% to 0.9%.

Everything else (gate statistics, REINFORCE bugs, retention, RNG
hygiene) is real and worth fixing, but is second-order next to these
two. §4 lays out the direction I recommend: a **cost-to-go value
network on afterstates, trained from and then used inside the
corrected search** — the single-player expert-iteration loop that
every strong modern Tetris bot converges on.

## 1. What I validated and how

| check | result |
|---|---|
| Full suite `.venv/py.sh -m pytest` | 275 passed, 1 warning, 389 s |
| Engine read-through (`game.py`, `board.py`, `constants.py`, `rng.py`) | rule logic correct for the cheese task; one ordering defect (§5.1) |
| Movegen / pathfinder / harness | complete for the 0G + infinite-SDF movement model the env actually uses; fast path is outcome-equivalent (tested) |
| Corrected beam replays through the *real input path* (`navigate=True`) | every decision legal; e.g. L2/seed 35 solved in 2 pieces both ways |
| Initial garbage rows by level | L1→1, L2→2, L3→3, L5→5, L9→9, L10→9, L18→9 |
| Shipped `models/curriculum/cheese_policy_L1.json` | 128×2 net (51,841 params), *not* the 256×3 server model; greedy: L1 100 % / 1.005, L2 39 % win, L3 9 % win |

The docs describe the task as "the field starts with a 9-row garbage
stack". That is only true from level 9 up. Levels 1–8 are small boards
with L garbage rows and nothing above them — a different, much easier
task than downstacking under a 9-row stack (§3.4).

## 2. Finding 1 — the teacher (highest impact)

### 2.1 What is wrong

`search.py` has two defects, one large and one small.

**The quiescence cutoff (large).** Both at the root and at deeper plies,
a placement that clears nothing is a leaf: it is scored by the eval and
never expanded (`search.py:169`, `:191`, `:205`). The docstring
justifies this with Jstris refill ("the cheese tops back up with
unknowable hole positions"). But `Game._cheese_target` returns
`min(stack, goal − dug)`, and every level starts with exactly
`min(9, goal)` rows on the board. So at levels ≤ 9 the refill count is
always zero, and at level 10 it is one row, once. The board after a
non-clearing move is *fully known* almost everywhere in the curriculum.
The rule therefore removes every setup move from the lookahead:
downstacking is mostly "place a piece that does not clear now so the
next one clears two", and the teacher cannot see any of it.

**Hold/queue transitions (small but real).** After a root hold with an
empty hold slot, the engine pulls `queue[0]` out as the active piece
and the *next* piece is `queue[1]`; the search keeps `queue_rest =
obs.queue` and plans `queue[0]` again (`search.py:163–167`). After any
hold, the engine re-enables hold at the next spawn; the search sets
`can_hold=False` for the child (`:165`, `:210`). Deeper plies never
consider an empty-hold branch. Net effect: some plans are built on a
piece that will not exist, and legitimate hold plans are missing.

### 2.2 Measured effect (paired, same seeds, fast path)

`e2_teacher_ab.py` runs four variants of a 20×4 beam with the same eval
weights. "expand" removes the cutoff (a placement is a leaf only when
the engine would *actually* refill); "holdfix" uses engine-exact
transitions; the combined variant also compares plans at a common
horizon (wins by fewest pieces first, otherwise deepest leaves by eval
+ 50 × cheese dug).

| level | variant | win | pieces (mean ± 95 % CI) | pieces / line | paired Δ vs original (better / worse seeds) |
|---|---|---|---|---|---|
| 3 (n=30) | original | 100 % | 4.93 ± 1.30 | 1.64 | — |
| | holdfix only | 100 % | 4.83 ± 1.31 | 1.61 | −0.10 (2 / 0) |
| | expand only | 100 % | 3.87 ± 0.52 | 1.29 | −1.07 (5 / 0) |
| | **holdfix + expand** | 100 % | **3.67 ± 0.42** | **1.22** | **−1.27 (7 / 0)** |
| 5 (n=100) | original | 100 % | 14.70 ± 1.60 | 2.94 | — |
| | holdfix only | 100 % | 13.98 ± 1.69 | 2.80 | −0.72 (24 / 10) |
| | expand only | 100 % | 9.15 ± 0.60 | 1.83 | −5.55 (73 / 11) |
| | **holdfix + expand** | 100 % | **8.47 ± 0.55** | **1.69** | **−6.23 (77 / 6)** |
| 10 (n=100) | original | 100 % | 43.61 ± 3.41 | 4.36 | — |
| | holdfix only | 100 % | 39.58 ± 3.17 | 3.96 | −4.03 (40 / 27) |
| | expand only | 100 % | 25.94 ± 1.07 | 2.59 | −17.67 (85 / 14) |
| | **holdfix + expand** | 100 % | **23.31 ± 0.77** | **2.33** | **−20.30 (91 / 9)** |

Note also the variance: the corrected teacher's CI at level 10 is a
quarter of the original's. The original's 85-piece games are runs
where it dug itself into a hole it could not plan out of.

Search then scales with compute (`e6_scaling.py`, level 10, 40 seeds,
corrected transitions):

| beam | pieces | pieces / line | s / episode (pure Python, 1 core) |
|---|---|---|---|
| 5 × 2 | 29.50 ± 2.45 | 2.95 | 2.5 |
| 10 × 3 | 26.18 ± 1.95 | 2.62 | 7.6 |
| 20 × 4 | 23.60 ± 1.26 | 2.36 | 19.9 |
| 40 × 5 | 21.55 ± 1.20 | 2.16 | 43.8 |
| 80 × 6 | 20.77 ± 1.02 | 2.08 | 86.6 |

Even the *smallest* corrected beam (5×2, 2.5 s/episode) beats the
current 20×4 teacher by a wide margin. The current teacher is not a
"runway"; it is a bottleneck.

### 2.3 Where the pieces go (material audit)

For a won canonical episode with G garbage rows, P pieces, J cleared
rows that contained no garbage, and R cells left on the board,
`4P = G + 10J + R` holds exactly (the other review states this identity;
`e5_checkpoint.py` asserts it on every episode). Measured at level 10
(30 seeds): the original beam clears **J = 15.8** junk lines per game,
the corrected one **J = 7.8**, with R ≈ 10 in both. Every wasted piece
shows up as junk lines: the original teacher spends most of its pieces
building and clearing its own stack rather than opening holes. This
audit is cheap to log per episode and is a far better diagnostic than
"pieces per line" alone.

### 2.4 The beam is effectively 1-ply

`e4_label_noise.py`, on teacher-visited states (30 episodes per level):

| teacher | level | agrees with 1-ply | label changes when hidden previews (queue[1:]) are permuted |
|---|---|---|---|
| original 20×4 | 3 / 5 / 10 | 96.5 % / 96.8 % / 97.4 % | 1.5 % / 0.5 % / 0.2 % |
| corrected 20×4 | 3 / 5 / 10 | 73.3 % / 66.9 % / 71.5 % | 9.9 % / 14.5 % / 16.6 % |

Two consequences. First, the "search teacher" whose labels the policy
has been fitting is, for practical purposes, the hand-tuned linear
eval, and `1ply` vs `beam20x4` in the baseline table (5.12 vs 4.86
pieces/line) confirms how little the lookahead adds. Second, the
corrected teacher genuinely uses the previews — in 10–17 % of decisions
its move depends on `queue[1:]`, which the current policy input does
not contain (§3.2). Fixing the teacher and widening the policy's
preview input must go together.

### 2.5 What to build

- Represent a search node as `(rows, active, hold, can_hold, queue
  index, dug, on_board)` and write **one** tested transition function
  used by every branch: no-hold, swap-hold, empty-hold. Assert against
  `Game` on random states (the triangle test has no analogue for the
  search's transition model today — add one).
- Expand non-clearing placements. A placement is a leaf only when the
  engine would refill after it (`lines == 0 and min(stack, goal − dug) −
  on_board > 0`). At that boundary use a **chance node**: at 100 %
  messiness the new hole is uniform over the 9 columns other than the
  last hole, so average (or sample 3 of) the 9 children. This is the
  Cold Clear "speculation" mechanism applied to garbage, and it is the
  only place uncertainty enters at levels ≥ 10.
- Compare plans at a **consistent horizon**: terminal wins ordered by
  fewest pieces; otherwise leaves at the same depth by value. Do not
  mix depth-1 and depth-4 leaf evals as the current `best_seen` does.
- Add a **transposition table** keyed on (rows, hold, queue index,
  can_hold): hold/no-hold orderings reach identical boards and eat
  beam width.
- Move `movegen` + `eval_board` to a compiled path (numba, or the
  checked-out Cobra movegen behind a thin adapter) before scaling data
  collection: the corrected 20×4 beam costs ~20 s per level-10 episode
  in pure Python, so 24k teacher episodes would be ~7 h on 19 workers.

`advisor-review-repro/beam2.py` is a 120-line reference implementation
of the first three points (no transposition table) that produced the
tables above. It is scratch quality; the production version should be
written against tests, not copied.

## 3. Finding 2 — the policy's input representation

### 3.1 Afterstate vs the current encoding (matched experiment)

`e7_afterstate.py`: 600 level-5 teacher episodes → 8,987 decisions,
80/20 split by decision, the *same* 128×2 tanh net, same Adam (1e-3),
same 500-decision chunks, same epochs; labels from the original beam
(so the target is essentially the 1-ply argmax and should be easy).

| encoding | epochs (updates) | train acc | **held-out strict-argmax acc** | **hole-class errors** (chosen move ≥ 20 eval points below the teacher's) |
|---|---|---|---|---|
| A — current (274 dims) | 30 (450) | 73.7 % | 64.5 % | 5.2 % |
| A — current | 100 (1500) | 91.2 % | **56.3 %** | **11.5 %** |
| B — afterstate (255 dims) | 30 (450) | 74.6 % | 70.0 % | 0.3 % |
| B — afterstate | 100 (1500) | 98.6 % | **72.4 %** | **0.9 %** |

The current encoding *overfits*: training accuracy keeps rising while
held-out accuracy falls and catastrophic errors double. The afterstate
encoding generalizes, and its residual disagreements are almost all
harmless near-ties. This matters for reading the project's own logs:
the "76–84 % accuracy" figures are training-set accuracy
(`DistillTrainer` never evaluates on held-out data), so the true
held-out agreement of the server models is unknown and probably lower.

Why afterstates win: the label is the argmax of a near-linear function
of holes, heights, transitions and lines cleared *of the resulting
board*. With the afterstate as input, an MLP has to learn a few local
patterns. With the current input it must first compose the 4×4 pattern
with an x one-hot and a landing row onto the board — a conjunction that
tanh layers learn slowly and memorize instead. The afterstate also
sidesteps the x-clipping collision (§3.3) entirely, is naturally
mirror-symmetric (free 2× augmentation, see §4.4), and — the important
part — is exactly the input a **value function** needs (§4).

### 3.2 Previews

`encode_state` keeps only `queue[0]` (`policy.py:90–91`). With the
current teacher that costs ~1 % of labels (§2.4); with the corrected
teacher it costs 10–17 %. Encode all five previews (order-sensitive,
one-hot each) plus the hold piece. Cheap and necessary.

### 3.3 The x-clipping collision, quantified

`encode_placement` clips the bounding-box x to `0..9`
(`policy.py:112`), so placements whose box starts left of the field
(x = −2, −1, 0 for a vertical I) share an input row. `e3_representation.py`
measured how much this matters on teacher-visited states:

| level | decisions with *some* duplicate candidate rows | teacher's chosen move inside a collision | argmax would then pick the wrong twin |
|---|---|---|---|
| 1 | 95 % | 0.0 % | 0.0 % |
| 3 | 76 % | 0.5 % | 0.0 % |
| 10 | 60 % | 0.7 % | 0.0 % |

So the defect is real and ubiquitous in the candidate set but almost
never touches the chosen move (landing row usually disambiguates on a
cheese board, and colliding twins sort after the teacher's pick). It
should be fixed — encode the full x range or, better, drop the
parametrization for afterstates — but it does not explain the
distillation accuracy gap. I flag this because the other review lists
it as a P1 alongside the teacher; by measurement it is two orders of
magnitude smaller.

### 3.4 What the curriculum actually trains

Because level L puts L garbage rows on an otherwise empty board, levels
1–8 are small clearing puzzles with a wide-open field. The skill the
project wants — digging under a 9-row stack where every misplaced cell
buries a hole — is only exercised from level 9. The `L1` checkpoint
illustrates the transfer: 100 % at level 1, 39 % at level 2, 9 % at
level 3 (`e5_checkpoint.py`), and on held-out level-2 teacher states it
picks a hole-class error 37 % of the time and misses an *immediately
winning* move in ~7 % of states.

Recommendations:

- **Decouple the two axes.** Keep `goal` as the objective and make the
  stack height a separate curriculum knob (the engine already has
  `cheese_rows`; what caps it is `_cheese_target`'s `goal − dug` term,
  which is the Techmino rule — verify against live Jstris before
  treating it as canonical, the other review could not). A ladder on
  a fixed 9-row stack with goal 2 → 10, or on stack height 3 → 9 with a
  fixed goal, measures the real skill from the first rung.
- **Decouple teacher from reference.** A gate that says "beat the
  agent whose labels you copy" can only be passed by the tie rule, so
  it measures imitation convergence, not mastery. Freeze a benchmark
  (the original 20×4 numbers on a locked 500-seed manifest are a fine
  historical baseline) and report absolute pieces/line and failure
  rate; use the strongest available search as the teacher regardless.
- Train on **mixed levels with a fixed stack** rather than mixing
  levels with different stack heights; a level-1 episode (one decision
  on an empty field) is nearly worthless as data.

## 4. Recommended direction: value-guided search (expert iteration)

The design that the measurements point to, and that MisaMino, Cold
Clear and the fusion bot share in some form:

```
                 labels: cost-to-go / search values
   corrected beam ───────────────────────────────▶  V(afterstate) net
        ▲                                                  │
        └──────────── leaf evaluator / move ordering ──────┘
```

1. **Target = remaining pieces to clear, not the teacher's argmax.**
   From corrected-beam rollouts, every visited afterstate has a
   Monte-Carlo cost-to-go (pieces until win; a failure gets a cost
   above the cap, tracked separately as a failure probability head).
   Regress `V(afterstate)` on it. This removes the label-arbitrariness
   problem — near-equivalent placements get near-equal targets — and
   optimizes the actual objective. It is also exactly what makes the
   afterstate encoding the right input.
2. **Use `V` as the beam's leaf evaluator** (replacing or blending with
   the linear eval) and, optionally, the 1-ply `V` ordering to prune
   expansion. §2.2 shows depth buys 2.36 → 2.08 pieces/line at 4×
   compute; a good `V` lets a shallow beam play like a deep one.
3. **Iterate**: collect with the improved search on the policy's own
   state distribution (this *is* DAgger, with values as labels), retrain
   `V`, re-benchmark on the locked manifest. Improvement can exceed the
   original teacher because the search improves on `V`; imitation's
   ceiling disappears.
4. **Bootstrapped targets** (n-step: pieces spent + `V` at the search
   horizon) once `V` is decent, to cut rollout cost.
5. Keep REINFORCE out of the loop until §5.2 is fixed and there is a
   value baseline to use as an advantage; today it is an unbiased but
   near-useless perturbation (every decision in an episode gets the
   same advantage, the entropy term has no gradient, the win bonus is
   mis-scaled at level 10).

### 4.1 Distillation, if kept

If a policy head is still wanted for latency, distill the *value
ordering* rather than the argmax: cross-entropy against a softmax of
the root children's search values (temperature-scaled), or a hinge on
the value gap. The corrected beam produces these numbers for free. Exact
teacher-index accuracy penalizes harmless ties and treats a cosmetic
disagreement like a buried hole.

### 4.2 Evaluation hygiene to put in place first

- Always report **held-out** agreement/regret (split by episode, not by
  decision), plus **regret under the teacher's search value** — it
  separates cosmetic from catastrophic errors (§3.1).
- Log `J`, `R`, failure reason and per-episode piece counts; publish
  paired per-seed results so two agents can be compared with a paired
  test.
- Three seed manifests: train / development (probes, model selection)
  / locked final. Today `_probe` and `gate` share `gate_seed0`
  (`curriculum.py:641–646`, `:655–660`), so DAgger keep/restore
  decisions are made on the seeds later used to certify the level.
- Cache the reference's per-seed results by (rules, config, seed); the
  reference is deterministic and is currently re-run at every gate.

### 4.3 Compute

Data collection is the bottleneck and the corrected beam is 3–5× more
expensive than the current one. In order of payoff: compiled
movegen/eval (10–50×), transposition dedup in the beam, and batching
`V` evaluations across all beam children per ply (one forward pass per
ply instead of one per node). The forkserver pool is a good design and
carries over.

### 4.4 Ideas worth an experiment (not claims)

- **Mirror augmentation for `V`.** A board's cost-to-go is
  left-right symmetric; SRS asymmetries affect which placements are
  *reachable*, not the value of a board. Mirroring afterstates doubles
  the data for free. (Do not mirror *actions*/labels without regenerating
  reachability — the other review is right about that.)
- **Refill chance node with hole-column marginalization** (§2.5) —
  measure at level 18/100 where it actually matters.
- **A lower bound for regret reporting.** At 100 % messiness adjacent
  holes never align, so one piece fills at most two holes;
  `ceil(remaining_cheese / 2)` is an admissible (weak) bound on
  remaining pieces, and the conservation identity gives
  `P ≥ (G + 10J_min + R_min)/4`. Use these to report *certified*
  optimality gaps on small levels, and consider a bounded exact solver
  for levels 2–4 within the visible queue (the other review's
  suggestion; I did not build one).
- **Difficulty-stratified data**: tag each state by whether the next
  cheese hole is covered and by cover depth; sample training states to
  flatten that distribution instead of letting 30-piece failure tails
  dominate (the "8:1 junk" problem the DAgger safeguards were built
  around becomes a sampling-weights problem).
- **Search node budget as the curriculum**: rather than levels, train
  `V` so that beam 5×2 + `V` matches beam 80×6 + linear eval on the
  locked manifest; that is a direct, measurable statement of "the net
  has learned what the search knows".

## 5. Second-order findings (fix while you are there)

### 5.1 Engine / harness

- **Goal resolution order.** `_lock` spawns the next piece
  (`game.py:482`) before checking the goal (`:486–488`). If the spawn
  blocks out (or a refill overflows), `_game_over(False)` wins the race
  and the later `_game_over(True)` is a no-op — a legal final dig can be
  recorded as a topout. Check the goal (and lock-out) before spawning.
  Rare in cheese rollouts; wrong in principle.
- **`styles` is load-bearing.** The dug count uses the render-only
  style grid (`game.py:417`: a cleared row is cheese iff any cell has
  style −2), while the field says "AIs can ignore this". Keep a
  dedicated per-row garbage flag/bitmask so the rule authority does not
  depend on a rendering structure.
- **`SevenBag` docstring** says "every window of 7 consecutive deals
  contains each piece exactly once" — false across bag boundaries. The
  implementation is a correct 7-bag; fix the doc.
- Bag and garbage share one RNG stream. Fine, but it means changing the
  level changes the piece sequence for the same seed (verified: seed 0
  next bag differs between L1 and L2), which weakens common-random-number
  comparisons across levels. Split the streams and record both.
- The direction document's "starts with a 9-row garbage stack" should
  be corrected to "starts with `min(9, level)` rows".

### 5.2 REINFORCE (`curriculum.py`)

- `entropies` are Python floats from `decide` (`policy.py:197`), so
  `− entropy_coef · ent_t.mean()` (`curriculum.py:325–326`) has no
  gradient. Recompute entropy from the segmented logits inside
  `_update`.
- Every decision in an episode receives the same advantage
  (`:313`). The "dense shaping" in the docstring does not exist: with a
  single undiscounted episode return, the shaping term is a constant
  for wins and only re-orders failures. If RL is kept, use reward-to-go
  with a value baseline (which `V` from §4 provides).
- `WIN_BONUS = 50` with −1 per piece makes a 100-piece level-10 win
  (return −40) worse than a 20-piece failure (−20). Use a cost
  formulation: pieces for a win, `cap + penalty` for a failure, and
  track failure rate separately.
- `_update` uses raw scores while sampling used `scores / temperature`
  (`policy.py:191`) — harmless at T = 1, a mismatch otherwise.

### 5.3 Gates and DAgger

- `check_gate`'s **strict** branch ignores win rate
  (`curriculum.py:192, 205`): a policy winning 1 % of games at 1.0
  pieces passes against 100 % at 2.0. Require the reliability test on
  every path. Win-conditioned mean pieces is a *secondary* metric.
- **Retention is logged, not enforced** (`:686–692`): a failed lower
  level still checkpoints and advances. It also iterates from
  `start_level`, so a resumed run never re-gates the levels below its
  start.
- The tie rule accepts "difference ≤ its own noise", which gets easier
  as the policy gets noisier. Use a pre-declared non-inferiority margin
  on the paired per-seed difference (both agents already play the same
  seeds — keep the per-seed results and use them).
- `_group_accuracy` counts ties as correct (`distill.py:186`). On
  trained nets ties are essentially absent (my strict and tie-inclusive
  numbers were identical), so this has not distorted reported figures,
  but it is the wrong metric; report strict argmax.
- `_dagger_episode` seeds NumPy but samples with `torch.multinomial`
  from the global torch RNG (`dagger.py:78`), so repeated tasks give
  different data and parallel collection is not reproducible. Pass a
  `torch.Generator` per task.
- `PolicyAgent.decide` appends device tensors to `trace` on every
  call, including greedy evaluation, and `gate_stats` never clears it —
  unbounded GPU memory growth across a 300-episode gate. Make recording
  opt-in.
- The replay budget is computed in *episodes* (`dagger_replay_decisions
  // 3`) but named in decisions; at level 10 it collects ~15× more than
  intended. Stop at a decision count.
- DAgger β decays 0.8 → 0.48 → 0.29 over the default three rounds and
  never reaches 0; report the realized teacher-move rate, not the
  nominal β.

## 6. Cross-validation of the other review

I read [cheese-ai-review.md](cheese-ai-review.md) after forming my own
view of the code, then checked its claims against the code and my
experiments.

**Confirmed by code reading or by my runs**: the hold/queue transition
bug; the quiescence cutoff and the "no refill below level 10" fact;
the L2/seed-35 4-vs-2 witness (my corrected beam finds the 2-piece
solution, also via real inputs); the x-clipping collision; the single
preview; the zero-gradient entropy; the return arithmetic; strict gate
ignoring win rate; retention not enforced; probe and gate sharing
seeds; unseeded torch sampling in DAgger workers; the tie-counting
accuracy metric; trace accumulation in greedy evaluation; the
goal/spawn ordering; the replay-budget unit mismatch; the `SevenBag`
docstring; the shared RNG stream. Its analysis is careful and I found
no factual error in the parts I checked.

**Where I weigh things differently** (measurement, not disagreement on
facts):

- The review presents six P1 items as co-equal. By measurement, the
  teacher cutoff is worth ~20 pieces per level-10 game; the encoding
  collision touches < 1 % of teacher labels and the missing previews
  ~1 % *with the current teacher*. I would fix all of them, but the
  project should not read the list as "six equally urgent problems".
- The review attributes the distillation accuracy gap chiefly to
  optimizer-step count and the representation defects. The step-count
  confound is real. But the matched experiment in §3.1 shows the
  bigger lever is the **afterstate input**: same steps, +16 points
  held-out and 12× fewer hole-class errors. And the current encoding
  *overfits with more steps*, so "more updates" alone may make the
  server models worse on held-out data.
- The review recommends a "modest shared board encoder plus candidate
  head". I would go one step further and make the candidate *be* the
  afterstate, scored by a value head trained on cost-to-go (§4). That
  removes the imitation ceiling and the label-arbitrariness problem at
  once, which a policy head over candidates does not.
- The review's experiment ladder puts exact small-instance solvers at
  stage 2. I agree they are valuable for regret certification, but the
  corrected beam alone already reveals a 2× gap; I would put the
  teacher fix and the afterstate switch first and the solver after,
  as a measurement tool rather than a training source.

**One nuance the review does not state**: the current beam agrees with
1-ply 97 % of the time. This reframes the whole "distill the search"
narrative — the project has been distilling a linear eval — and it
explains why the previews and collisions have had so little effect so
far.

## 7. Answers to the project's five open questions

1. **Is a 300-episode DAgger probe worth it?** The probe's real
   problem is not its size but that it decides on the gate's seeds and
   compares means of wins. Give it its own development manifest, compare
   paired per-seed costs with failures included, and keep it small; a
   probe is for catching disasters, not certifying non-regression.
2. **Why is distillation accuracy lower on the server?** Unknown,
   because only training accuracy was recorded. Likely contributors,
   in my order: the input encoding's poor generalization (measured),
   fewer optimizer updates per example at chunk 20k (arithmetic), and
   the mixed-level data. Measure held-out per-level accuracy and regret
   before concluding anything about scale.
3. **Should a 1–2 point reliability loss pass?** Decide it up front as
   a non-inferiority margin on failure rate, and make the failure rate
   a first-class metric everywhere. With a corrected teacher that
   completes 100/100 with a quarter of the variance, the policy's
   failures will stand out more clearly, not less.
4. **Search as permanent teacher?** Not as it stands. Corrected, it is
   an excellent teacher *and* a component of the final agent: §2.2
   shows the beam has plenty of headroom, and §4 shows how a learned
   value function lets you buy that headroom cheaply. The learner should
   overtake the linear eval, not the search.
5. **What to add to the triangle tests?** A search-transition test
   (search node vs `Game` after every branch type, random states); full
   afterstate equality after clears; goal resolution at the final lock
   with a blocked spawn; a no-refill assertion for levels ≤ 9 and a
   one-row assertion at level 10; determinism of parallel DAgger
   collection (the teacher collector has this test; DAgger does not).

## 8. Suggested order of work

| step | change | acceptance evidence |
|---|---|---|
| 1 | Corrected search transitions + non-clearing expansion + horizon-consistent comparison + transition test | paired ≥ 90/100 seeds not worse at L5 and L10; L10 ≤ 25 pieces at 20×4 (this review's numbers) |
| 2 | Afterstate encoding with 5 previews + hold; held-out metrics and regret in `DistillTrainer` | held-out strict agreement and hole-class error rate reported per level |
| 3 | Three seed manifests; reliability on every gate path; retention enforced; paired per-seed logging; frozen benchmark reference | gate decisions reproducible from saved per-seed files |
| 4 | Cost-to-go `V` on afterstates from corrected-beam rollouts; `V` as leaf eval | beam 10×3 + `V` ≥ beam 40×5 + linear eval on the locked manifest |
| 5 | Compiled movegen/eval; batched `V` in the beam | ≥ 10× episodes/s at equal strength |
| 6 | Stack-height axis in the curriculum; refill chance node; L18 / L100 | absolute pieces/line and failure-rate curves vs the frozen benchmark |
| 7 | (optional) policy head distilled from search values for low-latency play | latency / strength Pareto vs the search |

Keep the existing checkpoints and logs as the historical baseline; the
teacher and encoding changes both invalidate old labels, so version the
data format and regenerate.
