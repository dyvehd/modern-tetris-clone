"""Search baselines for the cheese race: 1-ply and beam-search agents.

Both are placement deciders on the harness protocol: enumerate the active
(and holdable) piece's reachable placements, lock each onto a row copy,
score the result, commit the best first move. The beam carries plans over
the observed preview queue — a beam of depth d plans exactly the pieces the
agent can see, the user's fixed-6-piece window made concrete.

Node model (exactly the engine's hold rules):
- placing a piece consumes it; the child's active piece is queue[0], and
  can_hold resets to True (the engine resets hold at the next spawn),
- the hold branch (only when can_hold) places what the hold brings out,
  stashing the current piece; the child cannot hold again immediately.
Hold is therefore a branch at every ply, not just the first.

Two approximations, both documented and only affecting plan *accuracy*
beyond what the agent commits, never legality:

- **Dug counting**: cheese is bottom-anchored, so a cleared row counts as
  dug cheese iff it lies within the bottom ``cheese_on_board`` rows of the
  pre-clear board (mixed cheese/junk rows count — matching the engine's
  contains-garbage style test; a fully-junk row inside the cheese region
  is the only divergence, and it cannot arise while the region is pure).
- **Quiescence leaves**: a placement that clears nothing is a beam leaf.
  Under Jstris refill semantics the cheese would top back up (with RNG hole
  positions the search cannot know), so the position after a combo break is
  unknowable in advance — the move is scored by eval alone and not expanded
  (the Cold Clear quiescence analog). Clearing plans stay exact: within a
  combo the stack only drains, never refills.
"""

from __future__ import annotations

from ..engine import board as B
from ..engine.constants import FIELD_H, PieceType
from .agents import BaseAgent
from .cheese import Decision, Obs, apply_placement, candidate_moves
from .eval import EvalWeights, eval_board
from .movegen import Placement, enumerate_placements


def lock_and_count(
    rows: list[int], placement: Placement, cheese_left: int
) -> tuple[list[int], int, int]:
    """Lock ``placement`` onto a copy of ``rows``: returns the new rows,
    lines cleared, and how many of them were cheese (bottom-``cheese_left``
    region rows — see the module docs for the approximation)."""
    merged = list(rows)
    B.merge_piece(merged, placement.piece, placement.rot, placement.x, placement.y)
    cleared = B.full_rows(merged)
    if not cleared:
        return merged, 0, 0
    dug = sum(1 for i in cleared if i >= FIELD_H - cheese_left)
    B.clear_rows(merged, cleared)
    return merged, len(cleared), dug


class OnePlyAgent(BaseAgent):
    """The 1-ply baseline: the single placement (active or hold) with the
    best eval score after it locks — eval + line bonus + win bonus. With
    hand-set weights this differs from :class:`GreedyDigAgent`'s lexicographic
    (lines, holes, height) by trading clears against shape quality."""

    def __init__(self, weights: EvalWeights | None = None):
        self.name = "1ply"
        self.weights = weights or EvalWeights()

    def decide(self, obs: Obs) -> Decision:
        w = self.weights
        best_score = float("-inf")
        best: Decision | None = None
        for placement, hold in _candidates(obs):
            rows_after, lines, dug = lock_and_count(list(obs.rows), placement, obs.cheese_on_board)
            score = eval_board(rows_after, w) + w.lines * lines
            if obs.cheese_dug + dug >= obs.goal:
                score += w.win
            if score > best_score:
                best_score = score
                best = Decision(placement, hold=hold)
        assert best is not None, "a live piece always has its ghost rest"
        return best


def _candidates(obs: Obs) -> list[tuple[Placement, bool]]:
    """(placement, hold) pairs — the shared action-space definition in
    :func:`tetris.ai.cheese.candidate_moves`."""
    return candidate_moves(obs)


class _Node:
    """One beam node: a plan prefix. The pieces consumed after the first
    move are queue_rest[0], queue_rest[1], ... (a hold swaps queue_rest[0]
    with hold)."""

    __slots__ = (
        "rows", "hold", "can_hold", "queue_rest", "dug", "cheese_left",
        "score", "first", "pieces",
    )

    def __init__(
        self,
        rows: list[int],
        hold: PieceType | None,
        can_hold: bool,
        queue_rest: tuple[PieceType, ...],
        dug: int,
        cheese_left: int,
        score: float,
        first: Decision,
        pieces: int,
    ):
        self.rows = rows
        self.hold = hold
        self.can_hold = can_hold
        self.queue_rest = queue_rest
        self.dug = dug
        self.cheese_left = cheese_left
        self.score = score
        self.first = first
        self.pieces = pieces


class BeamAgent(BaseAgent):
    """Beam search over the preview queue. ``width`` best nodes survive
    each ply; ``depth`` pieces are planned at most (capped by the visible
    queue + hold). Win terminals score the win bonus minus a per-piece
    decay, so among winning plans the shortest is preferred — the property
    that finds the 2-piece wins on level-1 edge seeds that 1-ply misses."""

    def __init__(
        self,
        width: int = 40,
        depth: int = 5,
        weights: EvalWeights | None = None,
        win_decay: float = 50.0,
    ):
        if width < 1 or depth < 1:
            raise ValueError("width and depth must be >= 1")
        self.name = f"beam{width}x{depth}"
        self.width = width
        self.depth = depth
        self.weights = weights or EvalWeights()
        self.win_decay = win_decay

    def decide(self, obs: Obs) -> Decision:
        w = self.weights

        # ply 1: the decision itself — active or hold, scored and kept.
        # A no-clear placement is a quiescence leaf at every ply (including
        # this one): under Jstris refill semantics the cheese tops back up
        # with unknowable hole positions after it, so no future win may be
        # credited to a no-clear move.
        frontier: list[_Node] = []
        leaves: list[_Node] = []
        for placement, hold in _candidates(obs):
            rows_after, lines, dug = lock_and_count(
                list(obs.rows), placement, obs.cheese_on_board
            )
            score = eval_board(rows_after, w) + w.lines * lines
            if obs.cheese_dug + dug >= obs.goal:
                score += w.win - self.win_decay  # a 1-piece win beats longer wins
            # the hold branch stashes the active piece; the queue is untouched
            hold_after = obs.active if hold else obs.hold
            node = _Node(
                rows_after, hold_after, not hold, obs.queue,
                obs.cheese_dug + dug, max(0, obs.cheese_on_board - dug),
                score, Decision(placement, hold=hold), 1,
            )
            (frontier if lines > 0 else leaves).append(node)
        if not frontier and not leaves:
            raise RuntimeError("no reachable placements for a live piece")
        best_seen = max(frontier + leaves, key=lambda n: n.score)
        frontier.sort(key=lambda n: n.score, reverse=True)
        frontier = frontier[: self.width]

        # plies 2..depth: expand clearing plans only — both the parent and
        # the child must have cleared (see the quiescence rule above)
        for _ in range(self.depth - 1):
            if not frontier:
                break
            nxt: list[_Node] = []
            for node in frontier:
                if node.dug >= obs.goal or not node.queue_rest:
                    continue  # won already, or no visible piece left to plan
                piece = node.queue_rest[0]
                rest = node.queue_rest[1:]
                rows = node.rows
                for p in enumerate_placements(rows, piece, allow_180=obs.allow_180):
                    rows2, lines, dug = lock_and_count(rows, p, node.cheese_left)
                    if lines == 0:
                        continue  # quiescence: no-clear continuations are leaves
                    score = eval_board(rows2, w) + w.lines * lines
                    if node.dug + dug >= obs.goal:
                        score += w.win - self.win_decay * (node.pieces + 1)
                    nxt.append(_Node(
                        rows2, node.hold, True, rest,
                        node.dug + dug, max(0, node.cheese_left - dug),
                        score, node.first, node.pieces + 1,
                    ))
                # the hold swap: place what hold brings instead of queue_rest[0]
                if node.can_hold and node.hold is not None:
                    for p in enumerate_placements(rows, node.hold, allow_180=obs.allow_180):
                        rows2, lines, dug = lock_and_count(rows, p, node.cheese_left)
                        if lines == 0:
                            continue
                        score = eval_board(rows2, w) + w.lines * lines
                        if node.dug + dug >= obs.goal:
                            score += w.win - self.win_decay * (node.pieces + 1)
                        nxt.append(_Node(
                            rows2, piece, False, rest,
                            node.dug + dug, max(0, node.cheese_left - dug),
                            score, node.first, node.pieces + 1,
                        ))
            if not nxt:
                break
            nxt.sort(key=lambda n: n.score, reverse=True)
            frontier = nxt[: self.width]
            best_seen = max(best_seen, max(frontier, key=lambda n: n.score), key=lambda n: n.score)
        return best_seen.first
