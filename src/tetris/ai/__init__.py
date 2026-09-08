"""AI layer: movement-graph movegen + pathfinding on the pure engine.

Built for a 0G planning model (the search assumes any enumerated placement
can be reached; timing feasibility is a later, separate filter). The cheese
harness (``cheese.py``) wraps the engine + navigation into the episode
protocol the AI stack is evaluated on.
"""

from .agents import BaseAgent, GreedyDigAgent, RandomAgent
from .cheese import (
    BatchResult,
    CheeseEnv,
    Decision,
    EpisodeResult,
    InvalidDecision,
    Obs,
    apply_placement,
    count_holes,
    run_batch,
    run_episode,
    stack_height,
)
from .eval import EvalWeights, eval_board
from .movegen import Placement, enumerate_placements, spin_class
from .pathfinder import find_path
from .search import BeamAgent, OnePlyAgent

__all__ = [
    "BaseAgent",
    "BatchResult",
    "BeamAgent",
    "CheeseEnv",
    "Decision",
    "EpisodeResult",
    "EvalWeights",
    "GreedyDigAgent",
    "InvalidDecision",
    "Obs",
    "OnePlyAgent",
    "Placement",
    "RandomAgent",
    "apply_placement",
    "count_holes",
    "enumerate_placements",
    "eval_board",
    "find_path",
    "run_batch",
    "run_episode",
    "spin_class",
    "stack_height",
]
