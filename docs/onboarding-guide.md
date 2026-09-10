# Onboarding Guide — Modern Tetris Clone

Welcome. This guide gets you from zero to productive in this codebase in
about an hour. It is written for a technical reviewer or new contributor;
companion references: [glossary.md](glossary.md) (every term and jargon
used in the project), [findings-and-directions.md](findings-and-directions.md)
(directions taken and findings to date), [expert-recommendations.md](expert-recommendations.md)
(external review compilation), [cheese-ai-learning.md](cheese-ai-learning.md)
(plain-language ML concepts), and
[ai-direction-and-results.md](ai-direction-and-results.md) (the full run
ledger and numbers).

## What this project is

A Tetris Guideline clone (Jstris rules primary; TETR.IO S1 / PPT
secondary) with a strict architecture split: a **pure-Python engine**
(`tetris.engine`) that is deterministic, runs a fixed 60 Hz timestep,
and has zero rendering dependencies — plus a **pygame-ce renderer** on
top. The split exists for the second half of the project: **an AI
stack** trained against the engine, currently a *cheese-race* learner
(clear garbage lines with the fewest pieces).

The load-bearing rule: **the engine is the rule authority**. Every AI
component mirrors engine behavior only as an optimization; whenever AI
code and engine code disagree about what is legal, the engine wins, and
tests assert the agreement (the "triangle" cross-validation — see
below).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[ai,dev]"    # engine + AI + tests
.venv/bin/pip install -e ".[game]"     # only if you want to play
.venv/bin/pytest                       # 287 tests, ~13 min CPU
```

- AI tests skip automatically when `torch` is missing (`[ai]` installs
  it; the engine tests run without).
- The repo lives at `github.com/dyvehd/modern-tetris-clone` (public).
  Branch naming: `ai/v1-navigation` … `ai/v7-teacher-repair` — an
  ordered ladder, each branch the deliverable that was merged to `main`
  after it. Read them in order to replay the project's history. Two
  security-hygiene commits (token scrub, `.env` ignore) sit on `main`
  after v7.
- **Never commit credentials.** `pull_checkpoint.sh` reads
  `MOLAB_URL`/`MOLAB_TOKEN` from the environment; a Molab notebook token
  briefly lived in the committed script (void since the sandbox expired;
  scrubbed in 3bd6392). Keep secrets in `.env` (gitignored) or the
  environment.

## The three layers (read in this order)

### 1. Engine — `src/tetris/engine/` (read first)

| file | what it is | why it matters to a reviewer |
|---|---|---|
| `constants.py` | piece cells, kick tables (SRS/180/I), spawn coords, cheese knobs | the ground truth for every geometry claim anywhere else |
| `board.py` | bitmask playfield ops (rows are ints; collision, merge, clear) | every hot path; review for off-by-ones here first |
| `game.py` | the `Game` state machine: `tick(actions)`, hold, spins, cheese refill, goal | the rule authority — longest file, read `tick` and `_apply_gravity` first |
| `config.py` | `GameConfig` dataclass | one place to see every ruleset knob |
| `rng.py` | `SevenBag` randomizer (+ the seedable cheese hole stream) | bag and garbage RNG — one shared stream today (a known, documented caveat) |
| `scoring.py`, `env.py` | scoring tables, game modes | read on demand |

Key property: **determinism** — same seed + same input sequence ⇒
identical game. Fixed 60 Hz; all timing (DAS/ARR/lock delay) in engine
time, not wall time. A test enforces that the engine imports no pygame.

### 2. AI movement layer — `src/tetris/ai/` (read second)

The AI never emits raw inputs. It decides *placements* (where a piece
should lock); a shared navigation layer converts a placement to real
inputs. This is the position-prediction / navigation decoupling.

| file | what it is |
|---|---|
| `movegen.py` | BFS over the movement graph: every *reachable* resting placement of a piece, with the engine's own spin verdict (spin-ins found, unreachable never listed) |
| `pathfinder.py` | shortest input sequence from spawn to a target placement |
| `cheese.py` | the episode harness: `CheeseEnv`, `Obs`, `run_episode` (navigated or fast-path), batch stats, `candidate_moves` (the shared action space) |
| `agents.py` | `RandomAgent`, `GreedyDigAgent` (lexicographic hand heuristic) |
| `eval.py` | board eval features (`EvalWeights`) used by search |
| `search.py` | `OnePlyAgent`, `BeamAgent` — the search agents. **The beam was rewritten in v7** (engine-exact hold/queue transitions, refill-exact leaf rule, horizon-consistent comparison); `docs/advisor-review-repro/beam2.py` is the validated reference |
| `tuning.py` | CEM weight tuning (measured: doesn't transfer — kept for the record) |

**The triangle test** (`tests/test_pathfinder.py`) is the heart of the
correctness story: for *every* placement the movegen enumerates × the
pathfinder's path × replay through a real `Game`, the lock must match
the prediction exactly (cells, lines, spin class). Any divergence in
physics between the three components fails the suite. Since v7 the
search has its own analogue: `test_empty_hold_transition_matches_engine`
replays every hold-flagged beam branch through a real engine clone.

### 3. AI learning layer — `src/tetris/ai/` (read third)

| file | what it is |
|---|---|
| `policy.py` | the learned policy: score-per-candidate MLP (`PolicyNet`, `PolicyAgent`), the **afterstate encoding** (v7: each candidate row = the board after lock+clear + outcome flags; context = 5 previews + hold + counters; INPUT_DIM 256), save/load with an input-dim guard |
| `curriculum.py` | the ladder: REINFORCE trainer (on-graph entropy bonus since v7), statistical gates (win-rate checked on every path since v7), `Curriculum` controller with retention *enforced* (blocks advancement) and attempt-disjoint seed bands (run-8 fix) |
| `distill.py` | behavioral cloning from the beam teacher (`TeacherRecorder`, chunked GPU trainer, held-out split + strict accuracy since v7) |
| `dagger.py` | DAgger rounds: policy-visited states, teacher labels |
| `parallel.py` | forkserver process pool for rollout/data collection |

Runner: `examples/cheese_learner.py` (CLI — every budget knob).
Baselines table: `examples/cheese_baselines.py`.
`examples/server_run.txt` documents the server-scale launch command and
per-run budget rationale.

**Checkpoints** live in `models/curriculum/` (JSON state-dicts). Current
gate-passed artifacts (v7 stack): `cheese_policy_L1.json`,
`cheese_policy_L2.json` (run 7b), plus run-8's best-L3 policy as
`run8_L3_best.json`. Everything older is the invalidated old-stack
baseline — `load_policy` refuses mismatched `input_dim` checkpoints by
design.

## Test suite map

287 tests, ~13 min, all pure CPU (the search tests are slow since v7 —
the corrected beam expands setup moves):

- `test_engine_*.py`, `test_rules_*.py` — the Guideline correctness core
  (spins, kicks, scoring, cheese refill, determinism, no-pygame).
- `test_pathfinder.py` — the triangle cross-validation.
- `test_cheese_harness.py` — harness legality, fast-path equivalence.
- `test_search_baselines.py` — agent behaviors, measured claims, and the
  v7 teacher-repair regression tests (seed-35 2-piece win, engine-exact
  transitions, shared action space).
- `test_weight_tuning.py`, `test_curriculum_learner.py`,
  `test_distillation.py`, `test_dagger.py`, `test_parallel_collect.py` —
  the learner stack (torch-guarded), including the v7 gate-hygiene
  tests (strict-gate win rate, retention blocking, probe band, entropy
  gradient, trace leak).

Conventions worth knowing before reading:

- **Measured claims in docstrings are load-bearing.** Numbers quoted in
  module docs come from actual runs in this repo's history; when a test
  asserts one, it pins the truth it measured, not an aspiration.
- **The engine never learns; the learner never bends the rules.** Any
  "optimization" that changes engine-visible outcomes is a bug, and the
  fast path (`navigate=False`) is asserted equivalent to the real input
  path on cheese outcomes.
- Seeds are partitioned into disjoint numeric bands (training / distill
  / DAgger / probe / gate — see `CurriculumConfig`); gates never see a
  training seed, and since the run-8 fix, **retry attempts** draw
  disjoint bands too.

## Server / Molab workflow (the operational layer)

Training runs happen on Molab (marimo notebook sandbox, RTX PRO 6000).
The recurring operational facts, learned the hard way:

- **Sandbox storage is disposable** (Molab's Aug 2026 policy): files
  written by scripts are wiped when the instance expires, and instances
  *do* expire mid-run (twice so far, HTTP 410 Gone). Treat the sandbox
  as compute-only.
- **Everything durable goes to git.** Checkpoints are pulled out through
  the notebook console (`scripts/pull_checkpoint.sh` + a server-side
  `pull_checkpoint.py` helper — gzip + base85 chunks, dual md5
  verification, no credentials on the server) and committed.
- The launch command and budget rationale for each run: see
  `examples/server_run.txt` and the run table in
  `findings-and-directions.md`.
- `ps` lies inside gVisor; check `/proc/{pid}/cmdline`. Launch runners
  detached (`nohup`, own session). The scratchpad (execute-code.sh) has
  a server-side time limit — long steps must be split or detached.

## Glossary (fast)

See [glossary.md](glossary.md) for the complete reference. The four
terms you need before reading any file here:

- **cheese race** — dig `level` garbage lines, fewest pieces wins.
- **placement** — a reachable resting position of a piece (`Placement`).
- **teacher / reference** — the beam search agent; *teacher* when its
  decisions are copied, *reference* when it is the gate opponent.
- **policy** — the learned net (never "learner" in code docs).

## Where to start reading (30-minute path)

1. `README.md` — project + game modes (10 min)
2. `src/tetris/engine/game.py` — `Game.tick` (10 min)
3. `src/tetris/ai/cheese.py` — the harness contract (5 min)
4. `src/tetris/ai/search.py` — the beam teacher (5 min)
5. `docs/glossary.md` — the vocabulary (skim, 5 min)
6. `docs/findings-and-directions.md` — where this is going and why (10 min)
7. `docs/expert-recommendations.md` — what the reviewers said, validated
