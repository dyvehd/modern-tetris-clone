# Cheese AI learning: terms and workflows

This document explains the machine-learning part of the project in
Simplified Technical English. Each term is defined once and used with the
same name everywhere. Read it top to bottom: later sections build on
earlier ones.

## The task

**Cheese race**: a game mode. The board starts with garbage rows ("cheese").
Each garbage row has one empty cell (the **hole**). The player must clear
all cheese lines. The winner is the player who uses the **fewest pieces**
to clear all cheese.

**Level**: the number of cheese lines to clear. Level 1 = clear 1 line.
Level 10 = clear 10 lines. Levels are the difficulty steps of the task.

## The game components

### Engine

The **engine** is the pure-Python game simulation (`tetris.engine`). It
runs the real rules: gravity, hold, lock, line clears, cheese refill. The
engine is the only rule authority. Nothing in the AI stack may invent its
own rules.

### Movegen

The **movegen** (`tetris.ai.movegen`) answers one question: for the current
board and the current piece, **which resting positions can this piece
reach through legal inputs?** Its output is a list of **placements**. A
placement says: piece type, rotation, x position, y position, spin class,
and the exact cells the piece would occupy.

### Pathfinder

The **pathfinder** (`tetris.ai.pathfinder`) takes one chosen placement and
returns the shortest list of button inputs (for example: LEFT, ROT_CW,
HARD_DROP) that moves the piece from its spawn position into that
placement, through the real engine rules.

### Candidates

**Candidates** are all placements available at one decision point. The
function `candidate_moves(obs)` returns them: every placement of the
active piece, plus every placement of the piece that a hold would bring
out. Candidates are the complete list of choices. This list is the shared
"action space": the search agents and the learned policy both choose from
the same list.

### Navigation

**Navigation** means: turning a chosen placement into button inputs and
applying those inputs to the engine. The navigation problem (how to move
the piece) is solved by movegen + pathfinder. It is separate from the
prediction problem (which placement is best), which is what the AI learns
or searches.

## The agents

An **agent** is any program that plays cheese episodes. Every agent has one
method: `decide(obs) -> Decision`. It receives an observation and returns
one decision.

**Observation (obs)**: a frozen snapshot of everything an agent may know
at a decision point: the board rows, the active piece, the held piece,
the queue, the cheese counters.

**Decision**: the chosen placement, plus a hold flag (true when the agent
wants to hold first).

These agents exist today, in increasing strength:

| Agent | File | What it does |
|---|---|---|
| RandomAgent | `agents.py` | picks a random candidate |
| GreedyDigAgent | `agents.py` | picks the candidate that clears the most lines now |
| OnePlyAgent | `search.py` | scores each candidate with the eval, picks the best |
| BeamAgent | `search.py` | searches several pieces ahead, plans the best line |
| PolicyAgent | `policy.py` | the learned agent (see below) |

### The search agents

The search agents are hand-written programs. They score board positions
with the **eval**: a linear formula over hand-chosen board features (holes,
column heights, and so on). The eval assigns each board a number. Higher is
better.

`OnePlyAgent` evaluates every candidate once and picks the best.
`BeamAgent` simulates several pieces ahead: it keeps the best `width`
plans at each step and explores `depth` pieces deep. Example:
`beam20x4` = width 20, depth 4.

The beam agent plays cheese near-optimally at low levels (level 1 mean 1.00
pieces). It is slow (it must simulate many futures per decision).

### The learned policy

The **policy** is a neural network that learns to choose placements. The
term "learner" in older notes means the same thing as "policy" — this
document uses **policy** consistently.

The policy is a **score-per-candidate MLP**: a small fully-connected
network. Its input is one candidate combined with the board state. Its
output is one score (a number). The network runs once per candidate; the
candidate with the highest score is the chosen move.

- The **state part** of the input contains: the full 10×20 board occupancy
  (every cell, filled or empty), the height of each column (0 to 20,
  divided by 20), the piece type of the active piece, the hold piece, and
  the next queue piece (each as a one-hot: a 7-slot vector with exactly one
  1), and the cheese counters (lines remaining, lines dug, goal), all
  scaled to small numbers.
- The **candidate part** contains: the 4×4 cell pattern of the candidate,
  its column position, its landing row, its rotation, whether it uses
  hold, and the piece type.

**Torch-native** means: the policy is written with the PyTorch framework,
not hand-written matrix math. The same program runs on the small local GPU
(RTX 3050) and on the big remote GPU (RTX Pro 6000) without code changes.

**Greedy mode**: when the policy is evaluated, it picks the candidate with
the highest score (no randomness). **Sampling mode**: during training it
picks candidates with probability according to the scores (softmax), so it
sometimes explores other moves.

## Training workflows

This section explains each training method, in the order we tried them.

### Episode and rollout

An **episode** is one full game: from the fresh cheese board to either
victory (all cheese cleared) or failure (topout: the stack reaches the
top, or the piece cap: 400 placements without finishing).

A **rollout** is one episode played to collect training data.

**Rollouts run on the CPU.** The engine is pure Python; playing games is
CPU work. The GPU cannot speed up this part. This is what "CPU rollouts
(engine-bound regardless)" meant: no matter which GPU you have, generating
the games themselves is CPU-limited. The GPU is used only for the network
update step.

### REINFORCE

**REINFORCE** is a policy-gradient training method. Plain language:
play games with the current policy, give each complete game a return (a
single quality number), then nudge the network weights so that the moves
of good games become more likely and the moves of bad games become less
likely.

**Return**: the sum of rewards of one episode. Our rewards: −1 per piece
placed, plus α (a weight, currently 1.0) per cheese line dug, plus 50 for
a win.

**Potential-based shaping**: the dug-line bonus is chosen so that its total
over a whole episode collapses to a simple closed form
(`−pieces + α × lines dug + win bonus`). The bonus changes the learning
signal during the episode but cannot change what the best policy is. It
"**telescopes**" means: when you sum the per-move shaping terms, the middle
terms cancel pairwise and only the start and end values remain.

**Why "GPU-batched"**: at the end of an iteration, all decisions from all
episodes of that iteration are combined into one large forward/backward
pass through the network on the GPU. One Adam (optimizer) step per
iteration. The per-decision softmax must then be computed separately per
decision — "segmented by candidate group" means: each decision has its own
candidate list, and the softmax runs within each list, not across the
whole batch. Implemented with `scatter_reduce` logsumexp.

**Cold-start REINFORCE** means: training the policy from random initial
weights with REINFORCE only, no teacher, no prior data. **This failed at
level 1.** Measured: the win rate of the sampled (exploring) policy
reached ~30-39%, but the win rate of the greedy policy stayed near 3%.
Reason: at level 1, winning requires exactly one specific move per
situation, and each situation (piece type + hole column) appears rarely,
so the gradient signal is too sparse. Fix: start from distillation
instead.

### Distillation

**Distillation** means: train the policy by copying the decisions of a
stronger program. The strong program is called the **teacher**. The
teacher here is the beam search agent (`beam20x4`). The term "oracle" in
older notes means the same thing — this document uses **teacher**.

Workflow:

1. The teacher plays many episodes on the CPU.
2. At every decision point, record: the candidate list (encoded) and the
   index of the candidate the teacher chose.
3. Train the policy network with a cross-entropy loss: the correct answer
   is the teacher's choice.

Measured result: with 62,000 recorded teacher decisions and 300 training
passes, the policy reaches 96.4% agreement with the teacher and plays
level 1 at the optimal 1.00 pieces per clear, level 2 at 100% wins with
2.44 pieces, level 3 at ~96% wins.

**Cross-entropy loss**: the training signal that says "make the chosen
candidate's score high, all others low, in proportion to how wrong they
are."

### DAgger (hard-state mining)

Distillation alone has a known weakness: the policy copies the teacher on
board states that the teacher itself visited. When the policy makes one
small mistake, it lands on a board state the teacher never saw, and there
it has no guidance. Small errors can grow into topouts. We measured this
at level 2: the distilled policy wins 99% of episodes but loses ~1% to
topouts the teacher would have won easily.

**DAgger** (a published method, Ross et al. 2011) fixes the state
distribution problem. Workflow per round:

1. Roll out episodes with the **current policy** (not the teacher).
2. At every board state the policy reaches, ask the teacher: "what would
   you do here?" Record that answer as the label.
3. Train the policy on these records.
4. Repeat. The policy's own rare mistakes now have teacher corrections.

The **mixture (beta, β)**: during a DAgger rollout the player sometimes
plays the teacher's move and sometimes the policy's move. β is the
probability of playing the teacher's move. β = 1.0 means the teacher plays
everything; β = 0.0 means the policy plays everything and only the labels
come from the teacher. Rounds typically start at β = 0.8 and decay to 0.
The label is always the teacher's choice, whatever is played.

Measured result at level 2: 7 rounds × 400 episodes improved mean pieces
from 2.71 to 2.51 at the same 99% win rate. The remaining gap to the beam
search (2.18) is a scale problem (more rounds, more episodes, a bigger
network), which is the remote-GPU workload.

### The curriculum

The **curriculum** is the level-by-level training schedule (user's rule,
made statistical):

1. Train the policy at the current level.
2. **Gate**: evaluate the policy on a fresh seed batch (seeds never used in
   training) and compare against the **reference** (the strongest search
   agent, `beam20x4`).
3. Advance only if the gate passes. If blocked twice in a row, stop.

The gate passes in two ways:

- **Strict**: the policy's mean pieces-to-clear + its 95% confidence
  interval sits strictly below the reference mean. This means: the policy
  is measurably better.
- **Tie**: the policy matches the reference within the combined noise of
  both evaluations, and its win rate is not lower. The tie rule is
  necessary at level 1, where the reference plays the theoretical optimum
  (1.00 pieces) — strict improvement is impossible there; matching the
  optimum consistently is mastery.

**Retention re-gating**: after an advancement, every easier level is
re-gated. This measures forgetting instead of hoping it away. A pass on
level N does not count if level 1 was forgotten.

**Ladder run**: one full pass of the curriculum from the start level to
the max level (or until blocked twice). The first ladder run stopped at
level 2: the policy passed level 1 (tie rule), but at level 2 its win rate
(97%) was below the reference (100%), so the gate blocked it twice. This
is correct behavior: the policy has not earned level 3 yet.

**Checkpoint**: a saved copy of the policy network after passing a level,
stored under `models/curriculum/cheese_policy_L{N}.json`.

## How the pieces fit together

```
                 choose placement          turn into inputs
   ┌─────────┐  ─────────────────▶  ┌─────────┐  ─────────────▶  ┌─────────┐
   │ policy  │     (candidate)     │ pathfin-│   (Actions)      │ engine  │
   │ network │                     │ der     │                  │ (rules) │
   └─────────┘                     └─────────┘                  └─────────┘
        ▲                                                            │
        │          teacher labels (distill/DAgger) or returns        │ episodes
        └────────────────────────────────────────────────────────────┘
```

At every decision: the observation is built from the engine state, the
candidates are enumerated by movegen, the policy (or search) picks one,
and the harness applies it (directly for speed, or through pathfinder as
real inputs for full fidelity).

## Glossary quick reference

| Term | Meaning |
|---|---|
| cheese race | game mode: clear all garbage lines with fewest pieces |
| level | number of cheese lines to clear |
| hole | the empty cell of a garbage row |
| engine | pure-Python game simulation, the rule authority |
| movegen | lists every reachable placement of a piece on a board |
| pathfinder | shortest input sequence from spawn to a chosen placement |
| placement | one resting position: piece, rotation, x, y, spin, cells |
| candidates | all placements available at one decision point |
| navigation | turning a chosen placement into inputs (movegen + pathfinder) |
| agent | any program with `decide(obs) -> Decision` |
| observation | the frozen snapshot an agent sees at a decision point |
| decision | a chosen placement + hold flag |
| eval | linear scoring formula over hand-chosen board features |
| policy | the learned placement-choosing network |
| score-per-candidate | net scores each candidate separately; softmax picks |
| teacher | the search agent whose decisions the policy copies |
| reference | the search agent the policy must beat at the gate |
| return | one episode's total reward (−pieces + α·lines + win bonus) |
| rollout | one episode played to collect training data |
| REINFORCE | policy-gradient training from episode returns |
| distillation | supervised training on teacher decisions |
| DAgger | distillation on policy-visited states with teacher labels |
| β (beta) | DAgger mixture: probability of playing the teacher's move |
| gate | the statistical pass/fail test for advancing a level |
| strict / tie | the gate's two pass modes (improvement / matched optimum) |
| retention | re-gating easier levels after an advancement |
| ladder run | one full curriculum pass from start level to max level |
| checkpoint | saved policy network after passing a level |

---

> **Note (v7 update)**: sections above describing the policy's *input
> encoding* predate the v7 afterstate encoding. The candidate part is no
> longer the 4×4 pattern + position + rotation; it is the **board after
> the placement locks and clears** plus its outcome flags, and the state
> part now carries all 5 previews. See [glossary.md](glossary.md)
> (afterstate) and [ai-direction-and-results.md](ai-direction-and-results.md)
> (the v7 stack) for current definitions; the training-workflow concepts
> in this document remain accurate.
