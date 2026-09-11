"""Blockfish as a harness cheese agent — the SOTA external oracle.

Adapts the trainer's :class:`BlockfishBackend` (ctypes over
``libblockfish_shim.so``) to the AI-harness ``decide(obs) -> Decision``
protocol: blockfish thinks on a minimal game view of the observation,
its advice's absolute lock cells are matched against our own movegen's
enumerated placements (cell-set identity — the engine-legal move that
locks the same cells), and the harness navigates/plays it like any
other agent. Navigation and legality stay 100% ours, so blockfish can
never play a move our engine disagrees with.

The shim returns the full ranked candidate list with per-candidate
``rating`` (blockfish's B*-search value, lower = better). This agent
keeps that ranking around as ``last_rank`` for the value-label
collection (round 2): each visited decision's candidates can be
matched to our movegen enumeration and labelled with blockfish's
relative preference.
"""

from __future__ import annotations

from .agents import BaseAgent
from .cheese import Decision, Obs
from .movegen import enumerate_placements


class _GameView:
    """The minimal game-like object ``Backend.think`` reads (rows, active,
    queue, hold_type, can_hold, cfg.hold_enabled, over)."""

    def __init__(self, obs: Obs):
        self.rows = list(obs.rows)
        self.active = self
        self.type = obs.active
        self.queue = tuple(obs.queue)
        self.hold_type = obs.hold
        self.can_hold = obs.can_hold

        class _Cfg:
            hold_enabled = True

        self.cfg = _Cfg()
        self.over = False


class BlockfishAgent(BaseAgent):
    """Plays blockfish's rank-0 advice. Stateless between decisions except
    for the exposed ``last_rank`` (labels + diagnostics)."""

    name = "blockfish"

    def __init__(self, search_limit: int = 50_000):
        from ..trainer.backends import BlockfishBackend

        self.backend = BlockfishBackend(search_limit=search_limit)
        self.last_rank: list[tuple[object, float]] | None = None

    def decide(self, obs: Obs) -> Decision:
        advice = self.backend.think(_GameView(obs))
        if advice is None or not advice.candidates:
            raise RuntimeError("blockfish returned no advice")
        piece = advice.piece
        if advice.hold:
            brings = obs.hold_piece
            if brings is not piece:
                raise RuntimeError(
                    f"hold would bring {brings.name if brings else None}, "
                    f"blockfish says {piece.name}"
                )
        placements = enumerate_placements(list(obs.rows), piece, allow_180=obs.allow_180)
        target = set(advice.cells)
        placement = next(
            (p for p in placements if set(p.cells) == target), None
        )
        if placement is None:
            raise RuntimeError(
                f"blockfish placement {sorted(target)} unreachable in our engine "
                f"({len(placements)} placements enumerated)"
            )
        # keep the ranked list (cells -> score, higher = better as the
        # backend negates blockfish's lower-is-better rating) for labels
        self.last_rank = [(c.piece, c.hold, c.cells, c.score) for c in advice.candidates]
        return Decision(placement, hold=advice.hold)
