"""Target-to-inputs navigation (pathfinder) for the AI layer.

Given a :class:`~tetris.ai.movegen.Placement`, find a shortest input
sequence that steers the piece from spawn into it. The result is a list of
:class:`~tetris.engine.game.Action` values — one per logic tick — ending in
``HARD_DROP``; replaying them through ``Game.tick`` locks the piece exactly
on the target cells with the engine's own spin detection matching
``Placement.spin``. That replay equivalence is the desync guard
(Zetris-style expected cells) and is asserted as a test oracle over every
placement the movegen produces.

The pathfinder walks the *same* movement graph as the movegen (same
``successors`` edge enumerator, same kick application), so the two cannot
diverge on reachability or spin semantics. Plain BFS is the right
algorithm: edges are uniform-cost, the state space is a few thousand
(rot, x, y, arrival-class) nodes at most, and exhaustiveness matters —
spin-in entries are only found by an order-complete search.

Terminal states are any state whose hard-drop ghost equals the target
position and whose arrival class yields the target's spin verdict. This
covers all three ways a placement is entered: sonic-dropping onto the
rest (spin "none"), rotating into a tucked slot and locking with a
zero-distance hard drop (spin kept — ``_hard_drop`` preserves
``last_action``), and rotating above the slot before the drop (class
preserved through the sonic fall, since infinite-SDF soft drop is one
``Action``).

A modern T-spin always ends rotation -> hard drop; paths for spin targets
come out that way naturally because the terminal check requires the
rotation class.
"""

from __future__ import annotations

from collections import deque

from ..engine import board as B
from ..engine.constants import SPAWN_X, SPAWN_Y, PieceType
from ..engine.game import Action
from .movegen import CLS_PLAIN, Placement, spin_class, successors


def find_path(
    rows: list[int], piece: PieceType, target: Placement, *, allow_180: bool = True
) -> list[Action] | None:
    """Shortest input sequence from spawn to ``target``, or None if the
    target is not reachable on ``rows`` (e.g. a hand-crafted placement
    inside a sealed pocket)."""
    assert target.piece is piece, "target placement is for a different piece"

    start = (0, SPAWN_X[piece], SPAWN_Y, CLS_PLAIN)
    # node -> (parent node, action that produced this node); None for start
    parents: dict[tuple, tuple | None] = {start: None}

    def terminal(node: tuple) -> bool:
        rot, x, y, cls = node
        if rot != target.rot or x != target.x:
            return False
        if B.ghost_y(rows, piece, rot, x, y) != target.y:
            return False
        return spin_class(rows, piece, rot, x, target.y, cls) == target.spin

    def enqueue(cur: tuple, nxt: tuple, action: Action, queue: deque) -> None:
        if nxt not in parents:
            parents[nxt] = (cur, action)
            queue.append(nxt)

    queue = deque([start])
    while queue:
        node = queue.popleft()
        if terminal(node):
            path: list[Action] = []
            cur: tuple | None = node
            while cur is not None:
                step = parents[cur]
                if step is not None:
                    cur, action = step
                    path.append(action)
                else:
                    cur = None
            path.reverse()
            path.append(Action.HARD_DROP)
            return path
        rot, x, y, _cls = node
        for n_rot, nx, ny, cls, action in successors(rows, piece, rot, x, y, allow_180):
            enqueue(node, (n_rot, nx, ny, cls), action, queue)
    return None
