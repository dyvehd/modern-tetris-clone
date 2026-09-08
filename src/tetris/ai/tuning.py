"""Rung-1 learning: cross-entropy tuning of the downstack eval weights.

The historical quality jump in heuristic Tetris (Dellacherie → BCTS) came
from tuning weights on a fixed feature set — exactly what this module does,
directly on the cheese objective: minimize mean pieces-to-clear (failures
counted as the piece cap, so survival stays a hard constraint).

Cross-entropy method (CEM): sample Gaussian weight vectors around the mean,
evaluate each candidate on the same seeded episode batch, recenter on the
elite quantile, iterate. Cheap, gradient-free, embarrassingly parallel —
each candidate's episodes run in a fork-pool worker (the engine is pure
Python, no shared state; forks inherit the parent's memory copy-on-write).

The tuner tunes the *search agent's* weights, not just 1-ply: the cost is
always measured through a full agent playing real episodes, so the weights
fit the agent that will use them (a beam's quiescence leaves interact with
hole/height tradeoffs in ways a static eval misses).
"""

from __future__ import annotations

import json
import multiprocessing as mp
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .cheese import CheeseEnv, run_batch
from .eval import EvalWeights
from .search import BeamAgent, OnePlyAgent


def make_agent(name: str, weights: EvalWeights):
    """The agent a candidate weight vector is evaluated through.
    ``"1ply"`` or ``"beam<width>x<depth>"`` (e.g. ``"beam20x4"``)."""
    if name == "1ply":
        return OnePlyAgent(weights=weights)
    if name.startswith("beam") and "x" in name:
        width, depth = name[4:].split("x")
        return BeamAgent(width=int(width), depth=int(depth), weights=weights)
    raise ValueError(f"unknown agent spec: {name}")


@dataclass(frozen=True)
class TuningConfig:
    """CEM hyperparameters. ``elite_frac`` of candidates survive each
    generation; ``smoothing`` blends the new elite mean/variance into the
    sampling distribution (0 = jump to elite, 1 = never move).

    ``levels`` is the training curriculum: the cost is the mean capped-cost
    across all listed levels (same seeds per level for every candidate).
    Tuning on a single level overfits to it — measured: weights tuned at
    level 10 alone improved level 10 but regressed levels 3-5; a
    multi-level objective is what the curriculum gates actually demand."""

    agent: str = "beam20x4"
    levels: tuple[int, ...] = (10,)
    episodes: int = 40  # per level, per candidate, per generation
    generations: int = 10
    candidates: int = 24
    elite_frac: float = 0.25
    smoothing: float = 0.7
    seed0: int = 0
    piece_cap: int = 400

    @property
    def n_elite(self) -> int:
        return max(2, int(self.candidates * self.elite_frac))


# weights that may be tuned, with clip bounds — lines/win stay fixed so the
# objective (clear fast) is never traded away by the tuner
TUNABLE: dict[str, tuple[float, float]] = {
    "holes": (-40.0, -1.0),
    "covered": (-10.0, 0.0),
    "height": (-2.0, 0.0),
    "bumpiness": (-4.0, 0.0),
    "row_transitions": (-4.0, 0.0),
    "col_transitions": (-8.0, 0.0),
    "wells": (-8.0, 0.0),
    "lowest_row_hole": (-60.0, -1.0),
}


def _cost_of_batch(batch, cap: int) -> float:
    """Mean pieces with failures at the cap — lower is better. The cap keeps
    survival a hard constraint: a topout costs the full budget, so the tuner
    can never trade win rate for speed."""
    return sum(p if r == "cleared" else cap for p, r in zip(batch.pieces, batch.reasons)) / len(
        batch.pieces
    )


def _evaluate_candidate(args) -> float:
    """Worker: play ``episodes`` seeded episodes at every training level
    with the candidate weights; the cost is the mean capped mean-pieces
    across levels. Runs in the fork-pool child."""
    vector, cfg = args
    weights = _vector_to_weights(vector)
    agent = make_agent(cfg.agent, weights)
    cost = 0.0
    for level in cfg.levels:
        env = CheeseEnv(level=level, piece_cap=cfg.piece_cap)
        batch = run_batch(agent, env, cfg.episodes, seed0=cfg.seed0)
        cost += _cost_of_batch(batch, cfg.piece_cap)
    return cost / len(cfg.levels)


def _weights_to_vector(weights: EvalWeights) -> list[float]:
    return [getattr(weights, name) for name in TUNABLE]


def _vector_to_weights(vector: list[float]) -> EvalWeights:
    fields = dict(zip(TUNABLE, vector))
    return replace(EvalWeights(), **fields)


def tune(
    cfg: TuningConfig = TuningConfig(),
    init_weights: EvalWeights | None = None,
    init_std: float = 0.3,
    verbose: bool = True,
) -> tuple[EvalWeights, list[dict]]:
    """Run CEM: returns (best weights, per-generation history). Deterministic
    given cfg — the sampler RNG is seeded from cfg.seed0, and every
    candidate plays the same seeded episode batch."""
    import random

    rng = random.Random(cfg.seed0 + 987_654_321)
    dim = len(TUNABLE)
    names = list(TUNABLE)
    lo = [TUNABLE[n][0] for n in names]
    hi = [TUNABLE[n][1] for n in names]
    span = [h - l for l, h in zip(lo, hi)]

    start = init_weights or EvalWeights()
    mean = _weights_to_vector(start)
    # scale the initial std to each weight's allowed span
    std = [init_std * s for s in span]

    best_weights = start
    best_cost = float("inf")
    history: list[dict] = []

    with _make_pool(cfg) as pool:
        for gen in range(cfg.generations):
            # sample candidates: mean + std * clipped normal, clamped to bounds
            vectors = []
            for _ in range(cfg.candidates - 1):
                v = [
                    min(hi[i], max(lo[i], mean[i] + std[i] * rng.gauss(0, 1)))
                    for i in range(dim)
                ]
                vectors.append(v)
            vectors.append(list(mean))  # the incumbent always gets evaluated

            costs = pool.map(_evaluate_candidate, [(v, cfg) for v in vectors])
            order = sorted(range(len(vectors)), key=lambda i: costs[i])
            elites = [vectors[i] for i in order[: cfg.n_elite]]
            elite_costs = [costs[i] for i in order[: cfg.n_elite]]

            gen_best_i = order[0]
            if costs[gen_best_i] < best_cost:
                best_cost = costs[gen_best_i]
                best_weights = _vector_to_weights(vectors[gen_best_i])

            # recenter on the elite mean; blend with smoothing
            new_mean = [sum(e[i] for e in elites) / len(elites) for i in range(dim)]
            new_std = [
                (sum((e[i] - new_mean[i]) ** 2 for e in elites) / len(elites)) ** 0.5
                for i in range(dim)
            ]
            alpha = 1.0 - cfg.smoothing
            mean = [alpha * m + cfg.smoothing * old for m, old in zip(new_mean, mean)]
            std = [max(alpha * s + cfg.smoothing * old, 0.02 * span[i])
                   for i, (s, old) in enumerate(zip(new_std, std))]

            rec = {
                "generation": gen,
                "best_cost": min(costs),
                "elite_mean_cost": sum(elite_costs) / len(elite_costs),
                "best_weights": dict(zip(names, vectors[gen_best_i])),
            }
            history.append(rec)
            if verbose:
                print(
                    f"gen {gen:3d}: best {min(costs):7.2f} | elite mean "
                    f"{sum(elite_costs) / len(elite_costs):7.2f} | all-time {best_cost:7.2f}"
                )
    return best_weights, history


def _make_pool(cfg: TuningConfig):
    """A fork-pool sized so a whole generation evaluates in parallel
    wall-clock (one worker per candidate, capped by CPU count)."""
    ctx = mp.get_context("fork")
    return ctx.Pool(processes=min(cfg.candidates, ctx.cpu_count()))


def save_result(path: Path, cfg: TuningConfig, weights: EvalWeights, history: list[dict], best_cost: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": asdict(cfg),
        "tunable_bounds": TUNABLE,
        "best_cost": best_cost,
        "best_weights": asdict(weights),
        "history": history,
    }
    path.write_text(json.dumps(payload, indent=2))
