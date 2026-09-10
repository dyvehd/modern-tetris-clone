"""The V-guided beam — the value network as the search's leaf evaluator.

Expert iteration's consumer half (review 2 step 4 / review 1 stage 5;
the reviews' convergence point): a :class:`ValueNet` trained on
Monte-Carlo cost-to-go labels scores each expansion's child by its
predicted remaining pieces, replacing the linear eval inside the
corrected beam. The beam's structure — engine-exact hold transitions,
the refill-exact leaf rule, horizon-consistent comparison — is
inherited untouched from v7's :class:`BeamAgent` through the
``_score_children`` hook.

Scoring (all children of one ply in ONE batched net forward — the
"batched V in the beam" compute shape; review 2's step 5 asks further
for a compiled feature layer, still open):

- a winning child keeps the base beam's guarantee: a win is a win, the
  shortest wins first — real wins always dominate value guesses;
- a non-winning child's plan score is ``−(prefix_pieces + λ_q·q̂ +
  λ_f·f̂)``: the pieces already spent on the plan plus the net's
  estimated total from the child's lock on (own piece included — the
  exact label the data pipeline regressed) and an explicit failure
  surcharge. Sign-flipped because the base ranks by ``max(score)``;
  lower estimated totals are better plans.

Because every plan's score is literally "estimated total pieces to
goal" (+ reliability surcharge), horizon-consistent comparison becomes
exact rather than heuristic — plans stopped at different depths
estimate the same quantity.

Feature-row fidelity: each child's row must equal what
``policy.encode_candidates`` would produce at that decision — the
context from the PARENT node (piece in hand, previews, hold, counters
at the decision), the afterstate from the CHILD (its locked board,
this placement's lines/dug/win outcome, hold flag). Pinned by
``test_node_encoding_matches_data_pipeline``.
"""

from __future__ import annotations

import numpy as np
import torch

from .cheese import Obs
from .policy import (
    BOARD_COLS,
    BOARD_ROWS,
    PIECE_INDEX,
    PIECES,
    PLACEMENT_DIM,
    STATE_DIM,
)
from ..engine.constants import FIELD_H, FIELD_W, PieceType
from .search import BeamAgent, Decision, _Node
from .value import ValueNet

INPUT_DIM = STATE_DIM + PLACEMENT_DIM  # 256, the v7 encoding


def _context_row(
    queue_rest: tuple[PieceType, ...],
    hold: PieceType | None,
    can_hold: bool,
    cheese_left: int,
    dug: int,
    goal: int,
) -> np.ndarray:
    """A DECISION's context row — ``policy.encode_state``'s layout fed
    from the parent beam node: the piece in hand (``queue_rest[0]``),
    the 5 previews behind it (``queue_rest[1:6]``), the hold slot, and
    the decision-time cheese counters. Not the child's post-placement
    view (that would shift the queue — the net was trained on decision
    contexts; deep nodes honestly see fewer previews as the plan
    consumes the visible window).

    ``can_hold`` has no context slot in the v7 encoding (hold-flagged
    candidates encode it); the parameter stays for signature parity
    with the Obs fields it mirrors."""
    n = len(PIECES)
    v = np.zeros(STATE_DIM, dtype=np.float32)
    for k, piece in enumerate(queue_rest[1:6]):
        v[k * n + PIECE_INDEX[piece]] = 1.0
    i = 5 * n
    if hold is not None:
        v[i + PIECE_INDEX[hold]] = 1.0
    i += n
    v[i + PIECE_INDEX[queue_rest[0]]] = 1.0
    i += n
    v[i] = cheese_left / 20.0
    v[i + 1] = dug / 100.0
    v[i + 2] = goal / 100.0
    return v


def _child_row(child: _Node, goal: int) -> np.ndarray:
    """One child's full (context | afterstate) row — the feature pair
    ``policy.encode_candidates`` produces for that candidate at that
    decision: context from the parent, afterstate from the child (its
    locked board, this placement's lines/dug-by-placement, win flag,
    hold flag)."""
    p = child.parent
    assert p is not None, "scored children always have a parent"
    v = np.zeros(INPUT_DIM, dtype=np.float32)
    v[:STATE_DIM] = _context_row(
        p.queue_rest, p.hold, p.can_hold, p.cheese_left, p.dug, goal
    )
    i = STATE_DIM
    base = FIELD_H - BOARD_ROWS
    for br in range(BOARD_ROWS):
        row = child.rows[base + br]
        for bc in range(FIELD_W):
            if (row >> bc) & 1:
                v[i + br * BOARD_COLS + bc] = 1.0
    i += BOARD_ROWS * BOARD_COLS
    v[i] = child.lines / 4.0
    v[i + 1] = (child.dug - p.dug) / 4.0  # dug BY this placement
    v[i + 2] = 1.0 if child.dug >= goal else 0.0
    v[i + 3] = 1.0 if child.hold_used else 0.0
    return v


class ValueBeamAgent(BeamAgent):
    """BeamAgent whose non-winning children are ranked by the learned
    value net instead of the linear eval.

    Args mirror the base beam plus:

    - ``lam_q``: how literally to trust the net's pieces estimate
      (1.0 = the plan score IS the estimated total pieces).
    - ``lam_f``: the failure head's surcharge in pieces-equivalents —
      the reliability contract made explicit (a 20% failure chance at
      λ_f=20 costs like 4 extra pieces).
    - ``device``: where the batched forward runs.
    """

    def __init__(
        self,
        net: ValueNet,
        width: int = 10,
        depth: int = 3,
        lam_q: float = 1.0,
        lam_f: float = 20.0,
        device: str = "cpu",
        name: str | None = None,
    ):
        super().__init__(width=width, depth=depth)
        self.net = net.to(device).eval()
        self.lam_q = lam_q
        self.lam_f = lam_f
        self.device = device
        self.name = name or f"beam{width}x{depth}+V"

    def _score_children(self, obs: Obs, children: list[_Node]) -> list[_Node]:
        """Rescore every non-winning child with one batched net forward
        over the ply's candidate rows."""
        todo = [c for c in children if c.dug < obs.goal]
        if not todo:
            return children
        x = np.stack([_child_row(c, obs.goal) for c in todo])
        xt = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_hat, fail_hat = self.net(xt)
        for c, q, f in zip(todo, q_hat.tolist(), fail_hat.tolist()):
            # pieces spent before this child's own placement + the net's
            # total-from-this-lock estimate = the estimated plan total
            c.score = -(c.pieces - 1 + self.lam_q * q + self.lam_f * f)
        return children
