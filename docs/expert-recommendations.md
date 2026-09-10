# Expert Recommendations — Compiled and Reconciled

This document compiles every recommendation from the two independent
expert reviews (both 2026-09-09, reviewing commit `ee32fb0`), records
how each was validated and what became of it, and merges them into the
current roadmap. The reviews' own documents —
[cheese-ai-review.md](cheese-ai-review.md) (review 1, "the defect
ledger") and [cheese-ai-advisor-review.md](cheese-ai-advisor-review.md)
(review 2, "the measured re-weighting") — are committed verbatim with
their repro packages; nothing here paraphrases them loosely, and where
the two experts weighed the same evidence differently, both positions
are recorded.

Validation protocol: every measurement from both reviews was re-run by
us (review 1's 14 witnesses: 14/14 reproduce; review 2's e2-e8 package:
every headline number reproduced, one script timed out at our limit
where its own README documents a ~4-minute runtime). No factual error
was found in either review. Their two disagreements are about
*weighting*, not facts — see §5.

---

## 1. The defect ledger and what became of each finding

Review 1's findings table, with review 2's measured weighting, our
validation, and the implementation status after v7 + runs 7b/8.

| # | Finding (both reviews' words, condensed) | Weight (r2 measured) | Validated | Status |
|---|---|---|---|---|
| 1 | Beam hold/queue transitions wrong (empty hold consumes queue[0]; can_hold resets per lock) | large but second to the cutoff (−4.03 pieces at L10 alone) | yes — witness + our engine-clone replay | **FIXED (v7)** — `_hold_transitions`, tested function; engine-clone regression test |
| 2 | Non-clearing cutoff too broad — every no-clear placement a leaf | **the big one: −20.3 pieces at L10 combined** | yes — refill-math derived and measured | **FIXED (v7)** — refill-exact leaf rule; setup moves planned |
| 3 | Encoding collisions (x clipped to 0..9) + only 1 of 5 previews encoded | collision real but **<1% of labels, 0.0% argmax changes**; missing previews cost ~1% (old teacher) but **10-17% (corrected teacher)** | yes — measured on teacher-visited states | **FIXED (v7)** — afterstate encoding kills collisions by construction; all 5 previews in context |
| 4 | Entropy bonus zero gradient; return can prefer 20-piece failure over 100-piece win | real (r2: "near-useless perturbation") | yes — coef 0 vs 100 bit-identical weights | entropy **FIXED (v7)** — on-graph, witness inverted and pinned. Return contract **OPEN** (win-bonus mis-scale at L10; needs explicit reliability/cost decision) |
| 5 | Strict gate ignores win rate; tie = failure-to-reject | real | yes — 1%-win lucky-mean passes; 299×1+1×40 passes vs 1.01 | win-rate **FIXED (v7)** on every path. Tie-as-non-inferiority **OPEN** (pre-declared margin + paired per-seed tests recommended) |
| 6 | Probe shares gate seeds; retention logged not enforced | real | yes — witness injects L1 failure, ladder advances | **FIXED (v7)** — probe on its own band (`gate_seed0+1M`); retention blocks advancement (run-8 live behavior) |
| 7 | DAgger sampling unseeded per-task; nominal β vs realized rate unlogged | real | yes — repeated task = different data | **OPEN** (per-task torch generators; realized-β logging) |
| 8 | Greedy eval retains candidate tensors (memory leak) | real | yes — trace grows across episodes | **FIXED (v7)** — greedy never records |
| 9 | Goal-check-after-spawn terminal ordering | real, constructed edge case | yes — high-buffer fixture replays | **OPEN** (rare in cheese rollouts; wrong in principle) |
| 10 | Replay budget named in decisions, allocated in episodes (~15× over-collect at L10) | real | yes — arithmetic | **OPEN** (stop at a decision count / reservoir sample) |
| 11 | `styles` grid load-bearing in dug counting | real | yes — code-read | **OPEN** (dedicated garbage flag/bitmask) |
| 12 | `SevenBag` docstring false ("every window of 7") | real | yes — bag-boundary case | **FIXED** (docstring corrected) |
| 13 | Bag/garbage share one RNG stream (level change ⇒ different piece stream per seed) | real | yes — verified seed 0 L1 vs L2 | **OPEN** (split streams, record both) |
| 14 | Training accuracy computed on training batches; ties counted correct | real | yes — all-zeros vector scores 100% | **FIXED (v7)** — held-out split + strict metric, logged in-run |
| 15 | "9-row stack" docs claim wrong below L9 | real | yes — engine gives min(9, level) rows | **FIXED** (docs corrected everywhere) |

Plus review 2's own additions, all validated: the L1-checkpoint-is-the-
128×2-laptop-model audit (docs corrected); the DAgger mixture sampling
ignoring agent config (greedy/temperature/RNG — **OPEN**); the shared
state-vector-per-candidate compute waste (**OPEN**, subsumed by the
value-network direction).

## 2. The two reframing measurements

Both reviews agree these two numbers reorder everything:

1. **The corrected teacher is ~2× stronger.** With engine-exact
   transitions and non-clearing expansion (review 2's `beam2.py`
   reference, validated by us and ported in v7): L10 43.61 → 23.31
   pieces at 20×4 (91/100 paired seeds better), 20.77 at 80×6
   (2.08 pieces/line). Even the smallest corrected beam (5×2) beats the
   old 20×4 teacher. Our v7 port measures 24.20 on a 20-seed batch —
   consistent.
2. **The old beam was effectively 1-ply.** Agreement with plain 1-ply
   on 96.5-97.4% of decisions (L3/5/10); the corrected beam's agreement
   drops to 67-73% and its move *depends on queue[1:]* in 10-17% of
   decisions. Distillation had been cloning a nine-feature linear eval.

Review 2's corollary, which run 7b/8 confirmed live: fixing the teacher
*and* the preview input had to go together (the corrected teacher's
decisions are preview-dependent), and the corrected teacher raises the
gate's bar — the L3 baseline moved 4.76 → 3.68, and the student's gap
widened accordingly. An honest gate got *harder* to pass.

## 3. Review 2's recommended direction (the convergence point)

Review 2's §4, which review 1's §"Use search to improve a value
function" independently agrees on — **value-guided search / expert
iteration**:

```
                 labels: cost-to-go / search values
   corrected beam ───────────────────────────────▶  V(afterstate) net
        ▲                                                  │
        └──────────── leaf evaluator / move ordering ──────┘
```

1. **Target = remaining pieces to clear** (not the teacher's argmax):
   every visited afterstate gets a Monte-Carlo cost-to-go from
   corrected-beam rollouts; failures tracked as a separate probability
   head "so that good efficiency cannot hide bad reliability."
   Regress `V(afterstate)` on it.
2. **Use `V` as the beam's leaf evaluator** (replace or blend the
   linear eval) and optionally as a 1-ply move-ordering pruner. Review
   2's acceptance: **beam 10×3 + V ≥ beam 40×5 + linear eval** on a
   locked manifest — a direct, measurable "the net learned what the
   search knows."
3. **Iterate**: collect with the improved search on the policy's own
   state distribution ("this *is* DAgger, with values as labels"),
   retrain, re-benchmark. "Improvement can exceed the original teacher
   because the search improves on `V`; imitation's ceiling disappears."
4. **Bootstrapped n-step targets** once `V` is decent (pieces spent +
   `V` at the search horizon) to cut rollout cost.
5. **Keep REINFORCE out of the loop** until a value baseline exists
   (today it is "an unbiased but near-useless perturbation": same
   advantage to every decision in an episode, entropy had no gradient,
   win bonus mis-scaled at L10).

Review 1's convergent formulation: `Q(s, a) = 1 + E[V(next obs)]`,
`V(goal) = 0`; failure handling per the reliability contract; use the
policy to order/prune candidate expansion; spend extra search on
uncertainty and teacher-student disagreement; distill the improved
search and repeat.

**Our run-7b/8 evidence endorses this independently**: the four L3
attempts (5.70 → 5.40 → 5.00 vs 3.68, win rate converged, held-out
65→72%) are exactly the imitation ceiling this design removes. The v7
afterstate encoding was built for it ("exactly the input a value
function needs" — review 2).

### If distillation is kept at all (review 2 §4.1)

Distill the *value ordering*, not the argmax: cross-entropy against a
softmax of the root children's search values (temperature-scaled), or a
hinge on the value gap — "exact teacher-index accuracy punishes
harmless ties and treats a cosmetic disagreement like a buried hole."
(Review 1's "soft targets / near-optimal action sets" is the same
recommendation.)

## 4. The experiment ladders, side by side

| Review 1 (7 stages) | Review 2 (8 steps) | Status after v7/runs 7b-8 |
|---|---|---|
| 1. Correctness: transitions, expansion, encoding, entropy, termination, gates, retention | 1. Corrected search + transition test | **DONE** (v7; all acceptance evidence met: seed-35 2-piece, L10 ≤ 25 at 20×4, ≥90/100 paired) |
| 2. Exact L1 cases; bounded L2-L4 certificates | — (deliberately deprioritized by r2: "a measurement tool, not a training source") | **OPEN** — L1 optimum derived (1 + 1/105 ≈ 1.0095, 496/500 measured); solver unbuilt |
| 3. Representation: corrected MLP vs shared/afterstate encoder, all previews, held-out | 2. Afterstate encoding + 5 previews + held-out metrics/regret | **DONE** (v7) — held-out split in-run; regret-under-teacher-value not yet logged |
| 4. Learning: distill vs +greedy DAgger vs +sampled DAgger vs actor/value | — | partially superseded: runs 7b/8 measured distill+DAgger on the corrected stack; greedy-vs-sampled DAgger comparison OPEN |
| 5. Search improvement: policy vs repaired beam vs policy/value-guided search | 4. Cost-to-go V on afterstates; V as leaf eval | **NEXT** (the roadmap's step 1) |
| — | 3. Three seed manifests; reliability on every path; retention enforced; paired per-seed logging; frozen benchmark | partial: reliability+retention **DONE** (v7); manifests/paired-logging OPEN |
| 6. Scale: native movegen/feature extraction, batched inference, then bigger data/models | 5. Compiled movegen/eval; batched V in the beam (≥10× episodes/s at equal strength) | **OPEN** — the compute enabler for expert iteration |
| 7. Transfer: L10/L18/L100; varied stack heights, messiness, rotation profiles; separate in-distribution and transfer tables | 6. Stack-height axis; refill chance node; L18/L100 | **OPEN** — requires the Techmino-vs-Jstris refill verification first |
| — | 7. (optional) low-latency policy head distilled from search values | OPEN, optional |

Review 2's other experiment ideas (worth an experiment, not claims):
mirror augmentation for `V` (board cost-to-go is left-right symmetric —
2× data for free; never mirror *actions* without regenerating
reachability), refill chance node with hole-column marginalization
(measure at L18/100 where it matters), admissible lower bounds for
certified regret (`ceil(remaining_cheese/2)`; the conservation identity
`P ≥ (G + 10·J_min + R_min)/4`), difficulty-stratified data (tag states
by cover depth; flatten the distribution — turns the junk-state
problem into sampling weights), and search-node-budget-as-curriculum.

## 5. Where the experts differed (both positions, resolved by evidence)

1. **Are the six P1s co-equal?** Review 1 listed them co-equal; review
   2 measured: the teacher cutoff is worth ~20 pieces/game at L10; the
   collision touches <1% of labels. Resolution: review 2's ordering
   was adopted (teacher first, encoding second, hygiene while-you're-
   there) and validated by outcomes — the teacher fix alone changed
   every baseline number.
2. **What explains the server distillation gap — optimizer steps or
   representation?** Review 1 foregrounded the step-count confound
   (my own earlier "26×" arithmetic corrected to ~10.8×); review 2's
   matched experiment showed the afterstate input buying +16 points
   held-out at the *same* steps, and the old encoding overfitting with
   *more* steps. Resolution: both were real; both fixed; the step fix
   (run 6) and the encoding fix (v7) each moved the ladder. Runs 7b/8
   then showed the *remaining* gap is neither — it's the imitation
   ceiling.
3. **Policy head vs value head.** Review 1: "a modest shared board
   encoder plus a candidate head is a reasonable next architecture."
   Review 2: go further — the candidate should *be* the afterstate,
   scored by a value head trained on cost-to-go, "which removes the
   imitation ceiling and the label-arbitrariness problem at once."
   Resolution: adopted review 2 (the v7 encoding is deliberately the
   value-function input).
4. **Exact solvers now or later?** Review 1 put them at stage 2; review
   2 put the teacher fix first and the solver after, "as a measurement
   tool rather than a training source." Resolution: review 2's
   sequencing adopted (v7 first); the solver remains on the roadmap for
   regret certification.

## 6. The merged roadmap (what we actually do next)

The evidence-ordered merge of both ladders, updated for what v7 and
runs 7b/8 settled:

1. **Cost-to-go `V(afterstate)`** from corrected-beam rollouts;
   `V` as the beam's leaf evaluator. Acceptance: beam-10×3+V ≥
   beam-40×5+linear-eval on a locked manifest. (Both reviews'
   convergence; runs 7b/8's diagnosed ceiling is exactly what this
   removes.)
2. **Compiled movegen/eval + batched V-in-beam** — ≥10× episodes/s at
   equal strength; without it each iteration round costs ~20 s/episode/
   core in pure Python. (Review 2 step 5; review 1 stage 6.)
3. **Three seed manifests + non-inferiority gates + paired per-seed
   logging** — gate decisions reproducible from saved per-seed files;
   pre-declared margins; frozen benchmark reference. (Review 2 step 3;
   review 1 stage 4/§4.)
4. **Stack-height curriculum axis** (after verifying the Techmino-style
   refill cap against live Jstris — the one rule both reviews could not
   pin), then refill chance nodes and L18/L100 transfer tables.
5. **Bounded exact solver for L2-L4** — certificates and certified
   regret reporting (both reviews; as a measurement tool).
6. **The small hygiene ledger** (any order, none blocking): engine
   goal-check ordering; styles-grid dependency; RNG stream split;
   DAgger per-task generators + realized-β; replay budget in
   decisions; reward win/failure contract; RL reward-to-go + value
   baseline if REINFORCE returns.
7. **(Optional)** low-latency policy head distilled from search
   values, for play-time latency — never as the training objective.

## 7. What the experts said to keep (and we have kept)

- The engine and movement layer as-is ("sound" — review 2 read them in
  full; review 1's ask is *more tests* at the boundaries, not changes).
- The curriculum idea with honest gates (both reviews stress: don't
  let "a statistical tie to a weak heuristic become the definition of
  a solved level" — the twice-blocked stops in runs 7b/8 are this
  principle working).
- The forkserver pool design ("a good design and carries over" —
  review 2).
- All existing checkpoints and raw run artifacts as the historical
  baseline — "correcting the teacher changes labels, and correcting
  the encoding changes the network interface; version both changes and
  regenerate affected data before combining results across versions"
  (review 1) — implemented as the input-dim checkpoint guard.
- For every reported run: save code revision, rule profile, encoding
  version, checkpoint epoch, seed manifest, actual optimizer steps,
  data counts, decision latency, and per-episode outcomes (review 1's
  reporting protocol — partially adopted; the seed manifests complete
  it).

## 8. Provenance

- Review 1: [cheese-ai-review.md](cheese-ai-review.md) +
  [cheese-ai-review-repro.py](cheese-ai-review-repro.py) (14 witnesses,
  all reproduced). Sources consulted by the reviewer: Hard Drop /
  TetrisWiki (hold, SRS, Jstris, T-spin, randomizer, garbage), Ng et
  al. 1999 (reward shaping), Ross et al. 2011 (DAgger), and the
  checked-out bot implementations (MisaMino, Cold Clear, Cobra,
  Zetris).
- Review 2: [cheese-ai-advisor-review.md](cheese-ai-advisor-review.md)
  + [advisor-review-repro/](advisor-review-repro/) (e2-e8 experiments,
  all numbers re-verified by us, including through the real input path).
- Implementation of the merged fixes: branch `ai/v7-teacher-repair`
  (commits 742994f, 72d673b, 9ad33c9), merged to `main` at 53f63a5;
  run-8's retry-freshness fix at b2c5690; results ledger in
  [ai-direction-and-results.md](ai-direction-and-results.md); the
  synthesis in [findings-and-directions.md](findings-and-directions.md).
