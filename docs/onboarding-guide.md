# Onboarding Guide — Modern Tetris Clone

Welcome. This guide gets you from zero to productive in this codebase in
about an hour. It is written for a technical reviewer or new
contributor; a companion plain-language reference for the AI training
concepts lives in [cheese-ai-learning.md](cheese-ai-learning.md), and
the current research direction and results in
[ai-direction-and-results.md](ai-direction-and-results.md).

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
.venv/bin/pytest                       # 275 tests, ~90 s
```

- AI tests skip automatically when `torch` is missing (`[ai]` installs
  it; the engine tests run without).
- The repo lives at `github.com/dyvehd/modern-tetris-clone` (public).
  Branch naming: `ai/v1-navigation` … `ai/v6-server-scale` — an ordered
  ladder, each branch the deliverable that was merged to `main` after
  it. Read them in order to replay the project's history.

## The three layers (read in this order)

### 1. Engine — `src/tetris/engine/` (read first)

| file | what it is | why it matters to a reviewer |
|---|---|---|
| `constants.py` | piece cells, kick tables (SRS/180/I), spawn coords, cheese knobs | the ground truth for every geometry claim anywhere else |
| `board.py` | bitmask playfield ops (rows are ints; collision, merge, clear) | every hot path; review for off-by-ones here first |
| `game.py` | the `Game` state machine: `tick(actions)`, hold, spins, cheese refill, goal | the rule authority — ~longest file, read `tick` and `_apply_gravity` first |
| `config.py` | `GameConfig` dataclass | one place to see every ruleset knob |

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
| `cheese.py` | the episode harness: `CheeseEnv`, observations, `run_episode` (navigated or fast-path), batch stats |
| `agents.py` | `RandomAgent`, `GreedyDigAgent` (lexicographic hand heuristic) |
| `eval.py` | board eval features (`EvalWeights`) used by search |
| `search.py` | `OnePlyAgent`, `BeamAgent` — the search agents; the beam is the *teacher* for the learner |
| `tuning.py` | CEM weight tuning (measured: doesn't transfer — kept for the record) |

**The triangle test** (`tests/test_pathfinder.py`) is the heart of the
correctness story: for *every* placement the movegen enumerates × the
pathfinder's path × replay through a real `Game`, the lock must match
the prediction exactly (cells, lines, spin class). Any divergence in
physics between the three components fails the suite.

### 3. AI learning layer — `src/tetris/ai/` (read third)

| file | what it is |
|---|---|
| `policy.py` | the learned policy: score-per-candidate MLP (`PolicyNet`, `PolicyAgent`), encodings, save/load |
| `curriculum.py` | the ladder: REINFORCE trainer, statistical gates, `Curriculum` controller |
| `distill.py` | behavioral cloning from the beam teacher (`TeacherRecorder`, chunked GPU trainer) |
| `dagger.py` | DAgger rounds: policy-visited states, teacher labels |
| `parallel.py` | forkserver process pool for rollout/data collection |

Runner: `examples/cheese_learner.py` (CLI). Baselines table:
`examples/cheese_baselines.py`. Checkpoints in `models/` (JSON
state-dicts) — note `models/curriculum/cheese_policy_L1.json` is the
128×2 laptop-era model (51,841 params), not a server-run checkpoint.

## Test suite map

275 tests, ~90 s, all pure CPU:

- `test_engine_*.py`, `test_rules_*.py` — the Guideline correctness
  core (spins, kicks, scoring, cheese refill, determinism, no-pygame).
- `test_pathfinder.py` — the triangle cross-validation.
- `test_cheese_harness.py` — harness legality, fast-path equivalence.
- `test_search_baselines.py` — agent behaviors and measured claims.
- `test_weight_tuning.py`, `test_curriculum_learner.py`,
  `test_distillation.py`, `test_dagger.py`,
  `test_parallel_collect.py` — the learner stack (torch-guarded).

Conventions worth knowing before reading:

- **Measured claims in docstrings are load-bearing.** Numbers quoted in
  module docs ("measured 5.6x", "99%→97% wobble") come from actual runs
  in this repo's history; when a test asserts one, it pins the truth it
  measured, not an aspiration.
- **The engine never learns; the learner never bends the rules.** Any
  "optimization" that changes engine-visible outcomes is a bug, and the
  fast path (`navigate=False`) is asserted equivalent to the real input
  path on cheese outcomes.
- Seeds are partitioned into disjoint numeric bands (training /
  distill / DAgger / gate — see `CurriculumConfig`); gates never see a
  training seed.

## Glossary (fast)

- **cheese race** — dig `level` garbage lines, fewest pieces wins.
- **placement** — a reachable resting position of a piece (`Placement`).
- **teacher / reference** — the beam search agent; *teacher* when its
  decisions are copied, *reference* when it is the gate opponent.
- **policy** — the learned net (never "learner" in code docs).
- **gate** — the statistical pass/fail at each curriculum level.
- **ladder run** — one full curriculum pass over levels 1..N.

## Where to start reading (30-minute path)

1. `README.md` — project + game modes (10 min)
2. `src/tetris/engine/game.py` — `Game.tick` (10 min)
3. `src/tetris/ai/cheese.py` — the harness contract (5 min)
4. `src/tetris/ai/search.py` — the beam teacher (5 min)
5. `docs/cheese-ai-learning.md` — the training workflows (10 min)
6. `docs/ai-direction-and-results.md` — where this is going and why
