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
from .movegen import Placement, enumerate_placements, spin_class
from .pathfinder import find_path

__all__ = [
    "BaseAgent",
    "BatchResult",
    "CheeseEnv",
    "Decision",
    "EpisodeResult",
    "GreedyDigAgent",
    "InvalidDecision",
    "Obs",
    "Placement",
    "RandomAgent",
    "apply_placement",
    "count_holes",
    "enumerate_placements",
    "find_path",
    "run_batch",
    "run_episode",
    "spin_class",
    "stack_height",
]
