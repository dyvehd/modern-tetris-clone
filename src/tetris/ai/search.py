"""Search baselines for the cheese race: 1-ply and beam-search agents.

Both are placement deciders on the harness protocol: enumerate the active
(and holdable) piece's reachable placements, lock each onto a row copy,
score the result, commit the best first move. The beam carries plans over
the observed preview queue — a beam of depth d plans exactly the pieces the
agent can see, the user's fixed-6-piece window made concrete.

Node model (exactly the engine's hold rules — the transitions match
``Game._spawn``/``_hold`` semantics, validated in review 2's repro):
- placing a piece consumes it; the child's active piece is queue[0] and
  can_hold resets to True after the lock (the engine resets hold at the
  next spawn),
- a hold branch with a filled hold places the held piece and stashes
  queue[0]; a hold branch with an empty hold stashes queue[0] and places
  queue[1] (the engine consumes queue[0] to fill the hold) — the child's
  next piece is queue[2] there,
- the child of a hold cannot hold again immediately (can_hold False).

One approximation, documented and affecting plan *accuracy* beyond what
the agent commits, never legality:

- **Dug counting**: cheese is bottom-anchored, so a cleared row counts as
  dug cheese iff it lies within the bottom ``cheese_on_board`` rows of the
  pre-clear board (mixed cheese/junk rows count — matching the engine's
  contains-garbage style test; a fully-junk row inside the cheese region
  is the only divergence, and it cannot arise while the region is pure).

Leaf rule (refill-exact): a placement is expanded unless the engine would
refill after it — a no-clear lock when the cheese on board has dropped
below its target (``min(stack, goal − dug)``). Below level 10 the target is
always already on board (no refill ever fires); at level 10 a top-up row
can appear after a no-clear lock, and positions past such a lock are
unknowable in advance (the hole position is engine RNG) — those, and only
those, are leaves. Wins are terminals. All other placements — clearing or
not — are expanded, so setup moves are planned like any other move.

Plans are compared at a common horizon: winning plans by fewest pieces
(pieces dominate eval: a win is a win), then shortest; non-winning plans by
eval + lines-weight × cheese dug along the plan (progress toward the goal
is comparable across stopping depths only when credited along the plan,
not just at the board face).
"""

from __future__ import annotations

from ..engine import board as B
from ..engine.constants import FIELD_H, PieceType
from .agents import BaseAgent
from .cheese import Decision, Obs, candidate_moves
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
        for placement, hold in candidate_moves(obs):
            rows_after, lines, dug = lock_and_count(list(obs.rows), placement, obs.cheese_on_board)
            score = eval_board(rows_after, w) + w.lines * lines
            if obs.cheese_dug + dug >= obs.goal:
                score += w.win
            if score > best_score:
                best_score = score
                best = Decision(placement, hold=hold)
        assert best is not None, "a live piece always has its ghost rest"
        return best


class _Node:
    """One beam node: a plan prefix. ``pieces`` placements have been
    consumed; the next piece to plan is ``queue_rest[0]`` (a hold swaps
    what it places, not the queue's order — see :func:`_hold_transitions`).
    ``refill`` marks the refill-exact leaf: the placement that made this
    node cleared nothing while cheese was below target, so the engine
    would top it up with unknowable rows next spawn."""

    __slots__ = (
        "rows", "hold", "can_hold", "queue_rest", "dug", "cheese_left",
        "score", "first", "pieces", "refill",
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
        refill: bool = False,
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
        self.refill = refill


def _hold_transitions(
    node: _Node,
) -> list[tuple[PieceType, bool, PieceType | None, bool, tuple[PieceType, ...]]]:
    """The engine's exact piece branches at a node, each as
    ``(piece, hold_used, hold_after, can_hold_after, queue_after)``:
    the piece to place now, whether it took the hold action, and the
    child's hold / hold-usability / queue state.

    Engine semantics (``Game._hold`` + ``_spawn``): a plain placement
    consumes the piece and hold resets to usable at the next lock; a hold
    with a filled hold places the held piece and stashes the active one
    (the queue is untouched); a hold with an empty hold stashes the
    active piece into hold and places queue[0], consuming it.
    """
    out = []
    if node.queue_rest:
        # plain placement: the active piece; hold resets after the lock
        out.append((node.queue_rest[0], False, node.hold, True, node.queue_rest[1:]))
    if node.can_hold:
        if node.hold is not None and node.queue_rest:
            # filled hold: place the held piece, stash the active one
            out.append((node.hold, True, node.queue_rest[0], True, node.queue_rest[1:]))
        elif node.hold is None and len(node.queue_rest) >= 2:
            # empty hold: stash the active piece, place queue[0]
            out.append((node.queue_rest[1], True, node.queue_rest[0], True, node.queue_rest[2:]))
    return out


class BeamAgent(BaseAgent):
    """Beam search over the visible queue. ``width`` best nodes survive
    each ply; ``depth`` pieces are planned at most. Engine-exact hold
    transitions (see :func:`_hold_transitions`), the refill-exact leaf
    rule, and horizon-consistent plan comparison (module docs). Among
    winning plans the shortest is preferred — the property that finds
    the 2-piece wins on edge seeds that 1-ply misses."""

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
        stack = obs.stack  # the env's cheese-stack cap (the refill target)

        # root: the engine state as observed — queue_rest carries the
        # active piece ahead of the visible previews, so ply-1 branches
        # are exactly candidate_moves(obs) (the shared action space).
        root = _Node(
            list(obs.rows), obs.hold, obs.can_hold,
            (obs.active,) + tuple(obs.queue),
            obs.cheese_dug, obs.cheese_on_board,
            0.0, None, 0,
        )

        children = self._expand(obs, root, stack)
        if not children:
            raise RuntimeError("no reachable placements for a live piece")
        wins = [c for c in children if c.dug >= obs.goal]
        leaves = [c for c in children if c.refill]
        frontier = [c for c in children if not c.refill and c.dug < obs.goal]

        # plans are compared at a common horizon: the best frontier node
        # or leaf seen at the deepest ply reached so far
        best = max(children, key=lambda n: n.score)
        frontier.sort(key=lambda n: -n.score)
        frontier = frontier[: self.width]

        for _ in range(self.depth - 1):
            if not frontier:
                break
            nxt: list[_Node] = []
            for node in frontier:
                nxt.extend(self._expand(obs, node, stack, first=node.first))
            if not nxt:
                break
            for n in nxt:
                if n.dug >= obs.goal:
                    wins.append(n)
                elif n.refill:
                    leaves.append(n)
            frontier = [n for n in nxt if not n.refill and n.dug < obs.goal]
            frontier.sort(key=lambda n: -n.score)
            frontier = frontier[: self.width]
            if frontier or leaves:
                best = max([best] + frontier + leaves, key=lambda n: n.score)

        # a win is a win: among winning plans the shortest is best
        if wins:
            return min(wins, key=lambda n: (n.pieces, -n.score)).first
        return best.first

    def _expand(
        self, obs: Obs, node: _Node, stack: int, first: Decision | None = None
    ) -> list[_Node]:
        """Children of ``node`` over every engine-branch placement. A child
        is a refill leaf when the engine would top the cheese back up after
        its lock: a no-clear lock with cheese below target (the hole position
        of the new row is engine RNG — unknowable, so not plannable)."""
        w = self.weights
        out = []
        for piece, hold_used, hold_after, can_hold_after, queue_after in _hold_transitions(node):
            for p in enumerate_placements(node.rows, piece, allow_180=obs.allow_180):
                rows2, lines, dug = lock_and_count(node.rows, p, node.cheese_left)
                total_dug = node.dug + dug
                on_board = max(0, node.cheese_left - dug)
                won = total_dug >= obs.goal
                target = min(stack, obs.goal - total_dug)
                refill = lines == 0 and target - on_board > 0
                # credit digging progress along the plan so stopping depths
                # are comparable; wins dominate by pieces, shortest first
                score = eval_board(rows2, w) + w.lines * total_dug
                if won:
                    score = w.win - self.win_decay * (node.pieces + 1)
                out.append(_Node(
                    rows2, hold_after, can_hold_after, queue_after,
                    total_dug, on_board,
                    score,
                    first if first is not None else Decision(p, hold=hold_used),
                    node.pieces + 1, refill,
                ))
        return out
