"""Baseline agents for the cheese harness.

Both decide *placements* only — the harness does the navigation (see
``cheese.py`` for the protocol). :class:`RandomAgent` is the floor baseline;
:class:`GreedyDigAgent` is the trivial digger whose level-1 play is provably
optimal, making it the end-to-end sanity check of the whole stack: harness +
movegen + pathfinder + engine.
"""

from __future__ import annotations

import random

from .cheese import Decision, Obs, apply_placement, count_holes, stack_height
from .movegen import Placement


class BaseAgent:
    """A cheese agent: :meth:`decide` maps an observation to one placement
    decision. Stateless agents may be reused across episodes; stateful ones
    (e.g. a seeded random bot) keep their stream across a batch, which keeps
    seeded batches reproducible."""

    name = "agent"

    def decide(self, obs: Obs) -> Decision:
        raise NotImplementedError


class RandomAgent(BaseAgent):
    """The floor baseline: a uniformly random reachable placement, never
    holds. ``seed`` fixes its stream — same seed + same episode seed = the
    same game, so batches are reproducible."""

    def __init__(self, seed: int | None = None):
        self.name = "random"
        self.rng = random.Random(seed)

    def decide(self, obs: Obs) -> Decision:
        placements = obs.placements()
        if not placements:  # a spawned piece always has its own ghost rest
            raise RuntimeError(f"no reachable placement for a live {obs.active.name}")
        return Decision(self.rng.choice(placements))


class GreedyDigAgent(BaseAgent):
    """Greedy downstacker: the placement (or hold-swap placement) that clears
    the most lines right now, ties broken by fewest holes, then lowest stack.
    Hold is only used when strictly better — active-piece candidates come
    first, and ``max`` keeps the first maximal one.

    At level 1 (one cheese line) this is optimal whenever one piece can win:
    I/J/L/T dig any hole column, S every column but 0, Z every column but 9,
    and an O start swaps in the queued piece via hold. The remaining ~1% of
    seeds (hole 0 with only S/O available, hole 9 with only Z/O) genuinely
    need 2+ pieces — greedy may take more than the minimum there, which is
    exactly the headroom search has to close (see the pinned seed tests).
    """

    def __init__(self):
        self.name = "greedy-dig"

    @staticmethod
    def _score(obs: Obs, placement: Placement) -> tuple[int, int, int]:
        rows_after, lines = apply_placement(obs.rows, placement)
        return (lines, -count_holes(rows_after), -stack_height(rows_after))

    def decide(self, obs: Obs) -> Decision:
        candidates: list[tuple[tuple[int, int, int], bool, Placement]] = [
            (self._score(obs, p), False, p) for p in obs.placements()
        ]
        if obs.can_hold and obs.hold_piece is not None:
            candidates += [
                (self._score(obs, p), True, p) for p in obs.hold_placements()
            ]
        _, hold, best = max(candidates, key=lambda c: c[0])
        return Decision(best, hold=hold)
