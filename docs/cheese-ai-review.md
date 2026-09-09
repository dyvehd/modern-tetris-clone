# Cheese AI: implementation review and research recommendations

Reviewed **2026-09-09**, at commit **`ee32fb0`**. Scope: the three project guides, the engine and placement interface, search, policy encoding, training, DAgger, evaluation, and representative tests and upstream implementations. This is an advisory review; production code and model checkpoints were not modified.

**Recommendation: keep the architecture, but fix the teacher, representation, and evaluation before scaling training.** Several observed limitations are implementation defects or information loss. The current evidence does not establish that the remaining gap is principally a data-volume or network-capacity problem.

The strongest constructive counterexample is **level 2, seed 35**: `beam20x4` takes **4 pieces**, while a **2-piece optimal solution** replays through the actual engine. This is already enough to make exact small-instance search a useful research tool.

## What was validated

- Ran the complete existing suite: **275 passed, 1 warning, 209.32 seconds**. The warning concerns converting a tensor requiring gradients to a scalar in a test. Environment: Python 3.12.14, PyTorch 2.11.0+cu128, NumPy 2.5.3. This differs from the guides' older 274-test count.
- Ran focused executable witnesses for search transitions, information loss, gradients, gate behavior, retention, RNG handling, and a terminal engine edge case.
- Consulted Hard Drop and TetrisWiki for hold, SRS, T-spins, Jstris, TETR.IO, and the random generator. Sources are listed below. Exact parity with live Jstris cheese refill behavior was **not** established; its guide endpoint returned HTTP 403.
- Inspected MisaMino's hold/search branches, Cold Clear's search-state and unknown-piece handling, Cobra's supported movement rules, and Zetris's placement interface. I did not benchmark those bots against this environment. No MochBot-named checkout was present under `tmp/`; `fusion` was present, but I have not treated it as MochBot or independently verified a competitive SOTA ranking.

Reproduce the focused observations from the repository root:

```bash
.venv/py.sh docs/cheese-ai-review-repro.py
```

The [witness script](cheese-ai-review-repro.py) prints JSON observations. Its successful exit means the witnesses executed, **not** that the reported behaviors are correct. It uses CPU only and writes no checkpoints. It is an audit snapshot, separate from the project's regression suite.

### Update from the latest supplied training screenshots

The user supplied newer remote-run summaries during this review. These are screenshot-reported results, not independently retrieved raw logs:

| Run/stage | Reported result | Interpretation |
|---|---|---|
| Run 5, L2 attempt 1 | Mean 2.71 pieces, 92% wins; beam mean 2.34 | Blocked; substantial reliability gap |
| Run 5, L2 attempt 2 | Mean 3.33 pieces, 99% wins; beam mean 2.34 | Blocked; better reliability, higher win-conditioned cost |
| Run 5 distillation | 76.8% teacher accuracy after 60 epochs; loss still falling | Supports investigating insufficient optimization |
| Run 6, resumed from the L1 checkpoint | 85.5% teacher accuracy at epoch 80; configured for 300 epochs and 10k-decision chunks | A promising supervised-learning improvement; no completed gameplay gate is shown |

This strengthens the optimizer-budget explanation as a **contributing factor**. It does not isolate it as the primary cause: the runs differ in warm start, training duration, and batch size, and the representation defects still limit what can be learned. The two L2 means are also conditional on different sets of wins; do not interpret their difference alone as a clean efficiency regression.

The step-count calculations in the screenshots use floor division. The implementation processes a final partial chunk, so the count is `epochs * ceil(actual_decisions / chunk_decisions)`. If the quoted datasets contain exactly 62,334 and 134,000 decisions, the corresponding counts are:

| Configuration | Optimizer steps |
|---|---:|
| Laptop: 62,334 decisions, chunk 2,000, 300 epochs | 9,600 |
| Run 5: 134,000 decisions, chunk 20,000, 60 epochs | 420 |
| Run 6: 134,000 decisions, chunk 10,000, 300 epochs | 4,200 |

Those exact dataset sizes must be confirmed from raw run metadata. Even using the screenshot's own rounded counts, **3,900 / 360 is about 10.8×, not 26×**. With the example counts above, run 6 gets 10× run 5's updates and less than half the laptop's. Update counts are a useful confound to control, but are not a complete measure of optimization equivalence. The replay collector parallelization described in the screenshot is already present in reviewed commit `ee32fb0`; the separate replay-budget issue below remains.

## Findings that should precede another large training run

| Priority | Finding | Consequence |
|---|---|---|
| P1 | Beam search has incorrect hold/queue transitions | Teacher labels can be based on impossible continuations |
| P1 | All non-clearing moves terminate search, even without refill | Essential setup moves receive no lookahead |
| P1 | Distinct placements have identical input encodings; four previews are omitted | Some decisions cannot be represented or teacher labels inferred |
| P1 | Entropy regularization has zero gradient; reward can prefer early failure | The implemented RL objective differs from its description and intended task |
| P1 | Strict gates ignore reliability; “tie” means failure to reject a difference | Advancement does not certify mastery or non-inferiority |
| P1 | Gate seeds select DAgger checkpoints; retention failure does not block advancement | Reported validation independence and retention guarantees do not hold |
| P2 | DAgger task seeds do not seed policy sampling | Collection changes with RNG history and worker scheduling |
| P2 | Greedy evaluation retains candidate tensors; replay is not bounded by its decision budget | Evaluation and collection memory grow unnecessarily |
| P2 | Completing the goal can still lose on the following spawn | A legal successful final placement can be classified as a failure |

P1 here means “fix before trusting the research conclusions,” rather than a security severity.

### 1. The beam teacher is not simulating its advertised search space

**Hold transitions are wrong.** In [search.py](../src/tetris/ai/search.py), lines 163–167 retain the entire `obs.queue` after an empty-hold placement. But empty hold consumes `queue[0]` as the placed piece, so the next active piece must be the original `queue[1]`. The search instead places `queue[0]` again.

The same root constructor uses `can_hold = not hold`, and the deeper hold branch sets it to `False` again. These are *post-lock* nodes. Hold becomes available after locking, including after a held piece locks, as both the engine and the hold reference specify [1]. The deeper search also omits empty hold when `node.hold is None`.

The witness at L3/seed 0 records a root empty-hold branch that places **O**. Its search node says the next piece is **O** and hold is unavailable; the engine says the next piece is **T** and hold is available. This is a transition mismatch, not a heuristic preference.

Represent a search node explicitly as `(board, active, hold, queue, hold_enabled, garbage_state, remaining_goal)`. After every completed placement, restore hold availability if the rules permit hold. Advance the queue according to whether the action used no hold, occupied hold, or empty hold. Make the transition a tested function instead of duplicating it across branches.

**The non-clear cutoff is too broad.** Lines 149–169 and 188–205 of `search.py` only extend clearing moves. Deeper non-clearing successors are discarded altogether. Increasing depth does not recover the missing setup sequences.

The justification is contradicted by the engine's own refill condition. Its target is:

```text
target = min(stack, goal - dug)
new_rows = max(0, target - cheese_on_board)
```

Consequently:

- L1 starts with **1** garbage row, L2 with **2**, and L9 with **9**.
- At every level up to 9, all required garbage is present initially. **No refill occurs anywhere in the episode**, regardless of combo breaks.
- L10 starts with 9 rows and introduces at most one further garbage row over the entire episode.
- Even in longer races, a non-clear on an already full garbage stack does not introduce new rows.

Search through deterministic non-clearing moves normally. At a transition that really introduces unknown garbage, use a chance node, sampled continuations, or a value estimate at that boundary. Do not equate “no line clear” with “unknown next board.” Also pass refill mode and stack configuration into the search state: `Obs` currently lacks them, while `CheeseEnv` exposes alternatives the teacher cannot distinguish.

**An actual optimality witness:** at L2/seed 35, the following two placements win, with no hold:

| Piece | Rotation | Bounding-box x | Bounding-box y | Engine inputs |
|---|---:|---:|---:|---|
| S | 1 | 4 | 36 | RIGHT, CW, HARD_DROP |
| L | 3 | 8 | 37 | RIGHT ×4, SOFT_DROP, CCW, HARD_DROP |

The script also checks that no one-placement win exists in the harness candidate set. The current beam takes four placements. This witness establishes teacher suboptimality; it does not by itself attribute all two wasted placements to a single defect.

Finally, winning nodes still include board evaluation and line bonuses alongside `win_decay`. A fixed decay of 50 does not mathematically ensure the shortest winning plan always wins the comparison. Use an explicit terminal ordering: success first, fewer placements second, optional tie-breakers last. For nonterminal search, accumulate piece cost and compare values with a consistent horizon instead of comparing arbitrary-depth leaf scores as if they were equivalent.

### 2. The policy cannot distinguish some decisions the teacher makes

**Placement encoding is not injective.** [policy.py](../src/tetris/ai/policy.py), lines 112–113, clips the bounding-box x coordinate into `0..9`. SRS bounding boxes can legally start left of column zero when their occupied cells remain in bounds.

On an empty board, vertical I placements with `(rot=1, y=36)` and **x = −2, −1, 0** occupy columns **0, 1, 2**, respectively. They have identical encodings. For any network weights, the three candidates therefore receive identical scores; greedy tie-breaking always favors the first. More data or a larger MLP cannot remove that collision.

Encode the full legal coordinate range, or preferably encode the absolute occupied cells / resulting board. Add a representation test that distinct cheese-relevant outcomes cannot silently collapse to the same input. Do not require distinguishing truly equivalent actions solely because their rotation labels differ.

**The teacher has five previews; the network gets one.** `encode_state`, lines 90–91, keeps only `obs.queue[0]`. The beam uses the rest. This is not merely a prospective concern: on the L3/seed-2 board, changing the visible queue from `ITSJO` to `IOTJS` changes the beam's selected move, while the **entire encoded candidate matrix remains identical**. Both tails permute the same remaining pieces. A deterministic student cannot reproduce both labels from the supplied input.

Expose all allowed previews with an order-sensitive encoding. For planning beyond those previews, retain observable bag history / remaining-piece information; never expose the simulator's hidden future or seed. Also include all board rows relevant to survival, or define and validate a sufficient compression: the current 20-row occupancy and height encodings both discard the buffer above the visible field.

A modest shared board encoder plus a candidate head is a reasonable next architecture. Encode the board once, combine it with queue/hold/goal context, then score candidate absolute cells or afterstates. This also avoids copying the same state vector into every candidate and recomputing its representation from scratch.

### 3. The RL implementation needs mathematical corrections

**The entropy bonus is a constant with respect to the parameters.** `PolicyAgent.decide` calculates entropy under `torch.no_grad()` and converts it to a Python float. [curriculum.py](../src/tetris/ai/curriculum.py), lines 325–326, builds a new tensor from those floats. Subtracting that tensor changes the displayed loss, but contributes no gradient.

With identical data and initial weights, changing `entropy_coef` from **0 to 100** produces **bit-identical updated weights** in the witness. Recompute entropy from the current differentiable segmented logits:

```text
log p(a | s) = logits(a) - logsumexp(logits for this decision)
H(s) = -sum_a p(a | s) log p(a | s)
```

The update also ignores `PolicyAgent.temperature`: rollout sampling uses `scores / temperature`, whereas `_update` uses raw scores. Defaults currently use temperature 1, but any other setting creates a behavior/update mismatch. Use the same logits definition in both paths.

**Telescoping alone does not establish policy invariance.** The implemented episode return is:

```text
R = -placements + alpha * garbage_dug + 50 * success
```

For successful episodes at a fixed goal, garbage dug is constant; the shaping term does not change their ordering. Across failures and successes it is not constant. Under the current defaults, a 100-piece L10 success earns **−40**, while a 20-piece failure that digs nothing earns **−20**. Those returns prefer the failure. This is a calculation of the objective, not a claim that the current agent deliberately chooses that failure.

Potential shaping has the form `gamma * Phi(next_state) - Phi(state)` [6]. In a finite episodic implementation, terminal potentials must be handled consistently, including topouts and caps. Setting the potential of every terminal state to zero is a straightforward safe convention. Keeping a different final dug count as the potential in each terminal state does not “telescope away” into a policy-independent constant.

Decide how reliability and piece efficiency trade off. A practical first objective assigns every failed episode a cost above the placement cap and every success its placement count. That orders individual failures behind individual successes, but a finite penalty still creates an expected-cost tradeoff; retain a separate reliability requirement. If reliability must be lexicographically primary, encode that explicitly in selection and evaluation.

**The current trainer is not using a dense temporal learning signal.** It computes one total episode return and gives it to every decision. The per-decision dug counters telescope into that scalar. This can be a valid high-variance episodic policy-gradient approach, but it does not provide the advertised intermediate reward-to-go benefit. A state-value baseline and reward-to-go / advantages should be the next RL comparison after the correctness fixes. Normalize the episodic gradient by the number of episodes when estimating expected episode return; the present mean over all decisions adds batch-dependent scaling as episode lengths change.

There are smaller robustness issues: a skipped zero-decision episode compresses `returns` while `dec_episode` retains original episode indices, and persistent REINFORCE Adam moments survive separate distillation/DAgger updates to the same weights. The first is latent under the usual low-stack setup; the second deserves a controlled optimizer-reset comparison, not an assumed universal fix.

### 4. The curriculum gate does not establish mastery

**Reliability is only checked in the tie branch.** `check_gate`, lines 192–207, accepts strict improvement in win-conditioned mean pieces without consulting `win_ok`. A learner with 1% wins averaging one piece passes against a reference with 100% wins averaging two. The non-CI path has the same omission. A single lucky successful episode can also get a zero effective uncertainty because missing CIs are treated as zero.

Require the reliability criterion for *every* acceptance path. Do not treat the mean among successful episodes as an unconditional performance measure: abandoning difficult seeds changes which seeds enter that mean.

**Not detecting a difference is not demonstrating equivalence.** The current tie rule accepts `mean_difference <= estimated_noise`. Higher variance makes acceptance easier. For example, 299 one-piece wins and one 40-piece win give mean **1.13**, CI half-width **0.2548**, and pass against a reference at **1.01 ± 0.01**. That is uncertainty about degradation, not evidence of mastery.

Use a predeclared non-inferiority or equivalence margin. For a non-inferiority test, the upper confidence bound on the policy's excess cost must be **below an acceptable cost margin**, rather than the observed difference being below its own noise. A strict-improvement claim should use uncertainty in the paired difference, not only the learner's CI against the reference's point estimate.

Both agents already play the same seed IDs. Preserve episode-level paired records. Use paired cost differences, discordant success/failure outcomes, and paired resampling where appropriate. Near 100% success, use a boundary-aware binomial or paired method; a pooled normal approximation and a zero-variance bootstrap cannot certify extremely low failure rates.

For perspective, **300/300 successes** give a one-sided exact 95% lower success bound of approximately **99.006%**, not 100%. With no observed failures, approximately **2,995 independent trials** are needed for that bound to exceed 99.9%. This is a sample-size requirement, not a reason to change the goalposts after seeing a result.

**Gate data are used to choose the model.** `_probe` and `gate` use the same `gate_seed0`. DAgger restore/keep decisions depend on the first 100 seeds of the same 300-seed gate. Reusing that gate across attempts and levels adds adaptive selection. Those seeds are disjoint from gradient-training seeds, but they are not untouched test data.

Use three purposes with distinct manifests: training, development/model selection, and final locked evaluation. A probe belongs to development. Repeated curriculum decisions can use fresh evaluation batches or a prespecified sequential procedure, with an untouched final benchmark for the reported result. Cache deterministic reference outcomes by exact rules/configuration/seed manifest.

**Retention is measured but not enforced.** In `Curriculum.run`, lines 686–698, a failed lower-level gate is logged and stored, but the model is still checkpointed as passed and the level still advances. The witness injects an L1 retention failure after an L2 pass and observes advancement to L3. Make the overall pass depend on all required retention results; define whether failure triggers replay training, restoration, or a blocked rung.

This also matters to the newly supplied run-6 setup: retention iterates from `cfg.start_level`, so a resumed run starting at L2 **never re-gates L1**, even when retention is enabled. Track previously mastered levels in checkpoint metadata rather than inferring them from the current run's starting level.

### 5. The data and scaling story has several confounders

**DAgger sampling does not follow its advertised policy configuration.** [dagger.py](../src/tetris/ai/dagger.py) directly samples a new softmax, ignoring `PolicyAgent.greedy`, temperature, and NumPy RNG. It therefore gathers stochastic-policy states even when deployment is greedy. This is a legitimate exploration choice if explicit, but it can contribute to the observed excess of long recovery trajectories. Compare greedy-policy DAgger against stochastic-policy DAgger at matched label budgets.

In [parallel.py](../src/tetris/ai/parallel.py), `_dagger_episode` seeds NumPy, but the actual action selection uses `torch.multinomial` without an explicit generator or task-specific Torch seed. Repeating an identical task in the same initialized worker produces different data. Worker scheduling changes the global Torch RNG history. The byte-identical parallel/sequential claim is supported by the teacher collector test, **not** by the DAgger test, which is a smoke test.

Use explicit per-episode RNG streams for mixture choice and policy sampling. Seed all model parameters, including biases, from one controlled initialization stream. Save RNG states and optimizer states if “resume” is intended to reproduce training rather than just load weights. The current checkpoint format saves only architecture and parameters.

**Fresh seed allocation is incomplete.** `train_level` restarts its iteration counter on each level and each retry, reusing the same REINFORCE seed schedule. Distillation and DAgger retries also reuse their level/round bands. Numeric bands are useful, but unbounded episode counts and fixed round offsets are not a structural proof of non-overlap. Use an allocator keyed by experiment, split, level, attempt, round, and episode, with saved manifests and overlap assertions.

**Replay is named in decisions but allocated in episodes.** `_dagger_level` turns 50,000 replay decisions into 16,666 episodes using an assumed three decisions per episode, then keeps everything. At roughly 49 placements per L10 game, that would be around **0.8 million decisions**, not 50,000. This is an estimate using the reported L10 length, not a rerun of that collection. Stop at a measured decision budget or reservoir-sample a bounded replay pool. Retain teacher data already collected for warm-starting instead of recollecting a large corpus for every refinement stage.

Three default DAgger rounds use beta **0.8, 0.48, 0.288**; they do not reach zero. The nominal beta is also reported as the teacher-move rate without measuring actual choices. Record both the intervention rate and agreement rate, since the teacher and policy may independently choose the same action.

**Greedy evaluation accumulates training traces.** `PolicyAgent.decide` always appends candidate tensors. `gate_stats` uses one agent for all episodes and never clears that list. The witness records 3 retained decisions after one capped episode and 6 after the next. On GPU, this retains device tensors throughout evaluation. Make recording opt-in, use a non-recording inference agent, and bound rollout storage by candidate count or bytes.

**Changing chunk size changes optimization, not just memory.** `DistillTrainer` takes one Adam step per chunk. A 62,334-decision corpus trained for 300 epochs at chunk size 2,000 receives **9,600 updates**; a 67,000-decision corpus trained for 60 epochs at chunk size 20,000 receives **240 updates**. The supplied screenshots report chunk size 2,000 for the laptop and a larger, approximately 134k-decision corpus for run 5; the update above gives that comparison separately. This is a major confound to resolve before attributing the server accuracy gap to capacity or a harder distribution. Match optimizer updates, examples seen, learning-rate schedules, and candidate volume in scaling comparisons.

**Teacher-move accuracy is not an independent quality metric.** The printed training accuracy is computed on the batches being optimized, not a held-out dataset. `_group_accuracy` also counts a target as correct whenever its score ties the maximum, even if greedy argmax would choose another candidate. An all-zero score vector reports 100% for any target; the witness demonstrates this.

Evaluate actual greedy selection, action-equivalent correctness, and search-estimated regret on held-out states. The local `tmp/distill_scaled2.log` also shows that the **2.44-piece L2 result belongs to epoch 200**, while **96.4% teacher accuracy belongs to epoch 300**, where L2 is **2.64 pieces**. The guide should distinguish those checkpoints. These numbers can all be real without describing the same model.

### 6. Engine agreement is valuable, but is not full rule validation

The pure engine, immutable observations, bitmask boards, and shared movement interface are good foundations. Tests already cover many useful rule cases. For the canonical generated cheese boards, a contiguous bottom region of garbage can make region-based dug counting exact; formalize and test that invariant instead of introducing a complex garbage mask without need. Editor/imported boards are a separate contract and may need explicit row provenance.

The current triangle establishes important **soundness on its fixtures**. It does not establish completeness relative to an independent movement implementation. Movegen and pathfinder share `successors`; a missing edge can be missing from both. Moreover, the triangle currently compares exact cell differences only for non-clearing, non-spin placements. Its clear/spin branch compares event labels and counts, not the entire resulting board. The fixture set is three boards crossed with seven pieces, not all reachable game states.

Add tests that compare full afterstates after clears, hold/queue state after successive placements, garbage counters and row provenance, and terminal outcomes. Use randomized *reachable* boards with saved failing seeds, plus adversarial high-stack and narrow-cavity fixtures. Compare move sets against an independent, rules-matched implementation such as Cobra, and replay discrepancy witnesses through the engine.

The fast harness path checks collision and resting position, but **not reachability or consistency of the supplied `cells` field**. This is acceptable for a trusted movegen-backed internal fast path, but its public contract currently promises that unreachable decisions raise. Either validate membership at that boundary or explicitly separate trusted and checked entry points. Current bundled agents use the shared movegen, so this is not evidence that existing rollouts regularly teleport into sealed cavities.

**A reproduced terminal-order bug:** `_lock` spawns the next piece before checking the goal. If that spawn blocks out, `_game_over(False)` makes the later `_game_over(True)` a no-op. In the supplied high-buffer fixture, an I legally clears the final garbage row; the following O spawn collides, and the result is `dug=goal=1, won=False`. Resolve victory and the intended lockout precedence before an unnecessary next spawn. This is a constructed edge case, not a measured frequency in cheese rollouts.

**Define a named rule profile.** “Guideline,” “Jstris,” and “TETR.IO SRS+” are not interchangeable specifications [2–5]. The engine uses standard SRS I kicks plus a TETR.IO-style six-test 180 table. TETR.IO's SRS+ also changes the I kick symmetry [4]. A universal claim that these constants implement both games needs independent evidence. Preserve deliberate extensions, but put rotation profile, spawn/topout rules, hold, refill, messiness, previews, and timing into benchmark metadata.

The chosen 0G/infinite-SDF placement task is a useful benchmark, but its move generator deliberately excludes finite-drop midair stops. Claims of exhaustive legal placements and optimality must name that movement model. The claim that all useful spin entries survive the restriction needs independent testing. T-spin classification itself has no direct cheese reward here; prioritize movement legality, afterstates, and termination over attack-table tuning.

Finally, garbage and bag shuffles share `SevenBag._rng`. At seed 0, simply changing the initial goal from L1 to L2 changes the subsequent bag returned by `bag.peek(7)`. This need not bias the marginal randomizer distribution, but it prevents independently controlling piece and garbage streams and weakens common-random-number comparisons. Separate those streams and save their manifests. Observe bag information through legal history; do not give an agent hidden RNG state.

## A more useful experimental direction

### First establish small exact references

L1 is almost a finite classification problem, not a useful long-running mastery gate. With empty hold, the only initial one-piece failures are hole 0 with the first two pieces S/O in either order, or hole 9 with Z/O in either order. The next distinct bag piece can dig that edge after an appropriate discard.

Under uniform independent hole columns and a uniform first bag:

```text
P(one piece is impossible) = 2 * (1/10) * (2/(7*6)) = 1/105
E[optimal placements] = 1 + 1/105 ≈ 1.009524
```

Thus “the universal L1 optimum is 1.00” is false. A particular seed batch may average exactly 1.00, and two-decimal formatting may hide rare cases. Enumerate the structural cases directly, replay their solutions, and use L1 as a deterministic regression/certificate set. Existing observed 496 one-piece cases out of 500 are compatible with this calculation.

For L2–L4, build bounded exhaustive search or iterative deepening with correct hold, all legal setup moves, transposition keys, and exact terminal piece cost. Record proven optima when the search closes; otherwise record a lower bound and a best known upper bound. A failed bounded search is not an impossibility proof beyond its bound.

Respect the preview limit. A solver that uses an entire generated future sequence is a **clairvoyant lower-bound reference**, not a fair online teacher. Exact certificates contained within the currently visible queue are especially valuable. With unknown future pieces or garbage, the online problem requires a contingent policy or an expectation over future observations.

### Use search to improve a value function, then improve search

The natural long-term system here is a **correct simulator + policy-guided search + learned remaining-cost value**. A learned dynamics model adds little while the exact engine is available.

1. Train the policy on exact small puzzles and a repaired search teacher. Supply all visible information and remove encoding collisions.
2. Learn remaining placement cost and failure probability from afterstates. Separate those outputs so that good efficiency cannot hide bad reliability.
3. Use the policy to order/prune candidate expansion and the value function to evaluate leaves. Keep a diversity budget across root actions, hold choices, and important hole-access patterns.
4. Generate stronger labels by spending more search on uncertainty, teacher/student disagreement, and high-regret decisions. Distill the improved search and repeat.

For an episodic cost objective, the conceptual update is:

```text
Q(s, a) = 1 + E[V(next observation)]
V(goal) = 0
```

Failure handling follows the chosen reliability/cost contract. Search can improve on its current policy, so imitation is not a permanent teacher ceiling. Pure behavioral cloning has no explicit mechanism to optimize beyond its labels, although generalization can incidentally outperform a teacher; single-player policy improvement does not require adversarial self-play.

Use soft targets or sets of near-optimal actions where several moves have equivalent cost. Exact teacher-index accuracy punishes harmless ties and weights a catastrophic hole burial the same as a cosmetic placement difference. A regret-aware target is more aligned with the objective.

### Three research ideas worth testing

These are proposed experiments, not claims of novel publication priority.

**A. A conservation-based efficiency audit.** For a completed canonical episode with G garbage rows, P placed tetrominoes, J cleared rows containing only player blocks, and R occupied cells remaining at the finish:

```text
9G + 4P = 10(G + J) + R
4P = G + 10J + R
```

Assumptions: each of the G generated garbage rows initially has nine cells, all garbage is eventually cleared, and no cells are removed by top overflow or editing. This holds for successful standard harness races; it is not a formula for arbitrary imported boards or topouts.

Log J and R alongside placements. The identity is both an accounting invariant and a way to explain inefficiency: extra player-only line clears and residual blocks consume the material budget. `ceil(G/4)` is a very weak initial lower bound; use relaxed geometry or small pattern databases to strengthen it, and prove admissibility before using a learned estimate to certify optimality. Do not independently minimize J or R as though their tradeoff were free.

**B. Search by the next garbage clear.** Replace the clearing-only cutoff with a bounded search for sequences that reach the *next dig event*, allowing non-clearing setup placements. Each option has a piece cost and ends in an exact afterstate. Rank `option_cost + remaining_cost_value`, rather than rewarding a long combo merely because it stays expandable. This should directly test whether the teacher is missing efficient setups. Unknown garbage still creates a real observation boundary; it must be handled explicitly.

**C. Counterfactual DAgger focused on avoidable waste.** At a difficult policy state, compare the chosen action, the teacher action, and a few alternatives using the same sampled future piece/garbage streams. Record estimated excess pieces and failure risk, not just one teacher index. Allocate teacher effort to costly disagreements, keep a bounded representative replay sample, and stratify recovery data by distance to the next exposed garbage hole. This addresses both label cost and long failure trajectories overwhelming clean-board examples.

Avoid blindly mirroring training examples: standard SRS I kicks and ordered kick tests can break naive left/right symmetry. Regenerate and verify transformed action sets under the selected rule profile before using reflection augmentation.

## Experiments in priority order

| Stage | Experiment | Evidence required to proceed |
|---|---|---|
| 1. Correctness | Repair hold transitions, deterministic setup expansion, encoding, entropy, termination, gates, and retention | Focused regression tests catch the current witnesses; existing suite still passes |
| 2. Reference | Exact L1 cases and bounded L2–L4 certificates; revised search baseline | Raw per-seed results and optimality gaps; engine replay for certificate solutions |
| 3. Representation | Corrected MLP versus shared board/afterstate encoder, with all previews | Matched data, updates, inference budget, and held-out episode performance |
| 4. Learning | Distillation only versus +greedy DAgger, +sampled DAgger, and +corrected actor/value training | Independent model-selection split; several training seeds; cost and reliability together |
| 5. Search improvement | Policy alone versus repaired beam versus policy/value-guided search | Pareto curves for placement cost, failure rate, and decision latency/node budget |
| 6. Scale | Native movegen/feature extraction and batched inference; then larger datasets/models | Profiled throughput and improved quality per CPU-hour/GPU-hour |
| 7. Transfer | L10, L18, and L100; varied stack heights, messiness, and legal rotation profiles | Separate in-distribution and transfer tables on locked seed manifests |

This preserves the useful curriculum idea while expanding difficulty beyond “number of garbage rows.” Hole transitions, local overhangs, buried player holes, required setup depth, hold dependence, and remaining preview horizon are separate axes. Mix them deliberately. Do not let a statistical tie to a weak heuristic become the definition of a solved level.

For every reported run, save the code revision, rule profile, model/encoding version, checkpoint epoch, seed manifest, actual optimizer steps, data counts by level/state type, decision latency, and per-episode outcomes. Report reliability with intervals, a declared unconditional cost, win-conditioned pieces as a secondary diagnostic, and exact regret wherever certificates exist.

Keep the present checkpoints and raw run artifacts as historical baselines. Correcting the teacher changes labels, and correcting the encoding changes the network interface; version both changes and regenerate affected data before combining results across versions.

## Answers to the project's five open questions

1. **Is a 300-episode DAgger probe worth it?** First separate probe and final-test seeds, use paired outcomes, and choose an acceptable regression margin. A small probe can catch disasters; non-significance at 100 or 300 episodes is not proof of no regression. Keep the pre-stage checkpoint and use a development selection policy with explicit uncertainty.
2. **Why is distillation accuracy lower on the server?** The omitted previews, placement collisions, batch/update-count change, state distribution, and training-only tie-aware metric are concrete confounders. Do not conclude “more scale” until they are controlled. Measure per-level and per-state-type regret as well as action accuracy.
3. **Should a 1–2 percentage-point reliability loss pass?** That is a product/research tolerance to declare in advance. A failure to reject a difference does not answer it. For a mastery claim, use a reliability floor or explicit non-inferiority margin and a sample size that can substantiate it.
4. **Should the beam remain the permanent teacher?** Use the repaired beam as a starting point and compare it with exact small references. Then iteratively improve search with learned values and policies. Wider/deeper versions of the present broken transition model are a poor first investment.
5. **What should be added to the triangle tests?** Full afterstates after clears; hold/empty-hold queue transitions across multiple locks; independent move-set comparisons; real refill-boundary cases; near-topout goal resolution; and adversarial rotation fixtures for the exact target game. The existing `assert same or not same` in the DAgger restore test is a tautology—replace it with a forced regression and an exact restored-state assertion.

## What to borrow from the checked-out engines

- **MisaMino:** its `dllai/ai.cpp` search explicitly distinguishes empty-hold queue consumption and occupied-hold swaps. This is a useful implementation comparison for the local transition bug. Its battle evaluation objective should not be inherited unchanged for cheese.
- **Cold Clear:** `bot/src/dag.rs` distinguishes reserve/hold state and known versus speculative generations, and discusses expected values for unknown pieces. Borrow state design, transposition/reuse ideas, and explicit uncertainty handling. Its implementation does not justify a blanket “stop after any non-clear” rule.
- **Cobra movegen:** use its independent configurable geometry and full finite-drop move generation for differential testing and potential acceleration after matching rule profiles. Its quoted empty-board perft throughput is not a benchmark of this project's entire search/training pipeline.
- **Zetris:** the TETR.IO interface emits expected cells alongside placements. Preserve that execution-validation approach and extend it to full cheese afterstates in tests.
- **MochBot / other competitive bots:** evaluate under identical cheese rules, observations, and compute budgets before treating battle strength as a cheese reference. This review makes no fresh ranking claim about them.

## Sources

External sources consulted on 2026-09-09; game-specific behavior should ultimately be pinned to fixtures and a versioned profile.

1. [Hard Drop: Hold piece](https://harddrop.com/wiki/Hold_piece) — hold becomes reusable after the outgoing piece locks.
2. [Hard Drop: SRS](https://harddrop.com/wiki/SRS) and [TetrisWiki: Tetris Guideline](https://tetris.wiki/Tetris_Guideline) — baseline rotation, spawn, hold, and topout conventions, with implementation variation.
3. [Hard Drop: Jstris](https://harddrop.com/wiki/Jstris) and [TetrisWiki: Jstris](https://tetris.wiki/Jstris) — five previews, cheese modes, game-specific scoring. These pages do not establish all refill details asserted by the repository.
4. [TetrisWiki: TETR.IO, Rotation System](https://tetris.wiki/TETR.IO#Rotation_System) — custom 180 kicks and SRS+ I-kick changes.
5. [Hard Drop: T-Spin](https://harddrop.com/wiki/T-Spin), [TetrisWiki: Random Generator](https://tetris.wiki/Random_Generator), and [TetrisWiki: Garbage](https://tetris.wiki/Garbage) — spin-rule variation, concatenated seven-bags, and game-dependent garbage behavior. Seven-bag means each *bag* contains all seven; not every sliding seven-piece window does, despite the `SevenBag` docstring.
6. Ng, Harada, Russell (1999), [Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping](https://www.cs.utexas.edu/~shivaram/readings/b2hd-NgHR1999.html) — potential-based reward-shaping formulation; linked page provides the bibliographic record.
7. Ross, Gordon, Bagnell (2011), [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a.html) — DAgger and learner-induced state distributions.

Local implementation references are linked throughout. Historical log numbers were inspected in the local ignored `tmp/` directory; this review does not reproduce the remote training runs or the complete historical 500-seed benchmark table.
