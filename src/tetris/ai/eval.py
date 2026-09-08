"""Downstack evaluation function for the cheese-race search baselines.

A linear board eval in the Dellacherie lineage, adapted for digging: holes
are worse than in stacking modes (they block the cheese shaft), covered
holes add depth penalty, row transitions punish jagged surfaces that waste
piece cells, and a big win bonus makes the beam prefer guaranteed wins over
eval-scored continuations. Every feature is a plain loop over the row
bitmasks — no engine objects, no allocation beyond small lists.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..engine.constants import FIELD_H, FIELD_W


@dataclass(frozen=True)
class EvalWeights:
    """Weights for the downstack eval. `lines` dominates: clearing is the
    objective, everything else manages the shape while digging. Deliberately
    hand-set (the rung-1 learning step tunes exactly these numbers)."""

    lines: float = 50.0  # lines cleared by the placement
    holes: float = -8.0  # each covered empty cell
    covered: float = -1.5  # each filled cell directly above a hole (depth)
    height: float = -0.3  # aggregate column height
    bumpiness: float = -0.4  # sum of adjacent-column height differences
    row_transitions: float = -1.0  # filled/empty changes per row
    col_transitions: float = -2.0  # filled/empty changes per column
    wells: float = -1.5  # each cell of a 3-wide-or-narrower well
    lowest_row_hole: float = -20.0  # each empty cell of a partially-filled bottom row (the dig target); a fully dug board is exempt
    win: float = 1_000_000.0  # episode won (terminal)

    def vector(self) -> tuple[float, ...]:
        return tuple(getattr(self, f.name) for f in self.__dataclass_fields__.values())


def column_heights(rows: list[int]) -> list[int]:
    """Height of each column measured from the floor (0 = empty column)."""
    heights = [0] * FIELD_W
    for i, row in enumerate(rows):
        if not row:
            continue
        for x in range(FIELD_W):
            if not heights[x] and (row >> x) & 1:
                heights[x] = FIELD_H - i
        if all(heights):
            break
    return heights


def count_row_transitions(rows: list[int]) -> int:
    """Filled/empty changes along each row, walls count as filled. Bumpy
    surfaces and floating overhangs both score badly."""
    transitions = 0
    for row in rows:
        if not row:
            continue
        prev = 1  # the left wall
        for x in range(FIELD_W):
            cur = (row >> x) & 1
            if cur != prev:
                transitions += 1
                prev = cur
        if prev == 0:
            transitions += 1  # the right wall
    return transitions


def count_col_transitions(rows: list[int]) -> int:
    """Filled/empty changes along each column, the floor counts as filled
    (Dellacherie convention — an empty bottom cell reads as a hole signal)."""
    transitions = 0
    for x in range(FIELD_W):
        prev = 1  # the floor
        for i in range(FIELD_H - 1, -1, -1):
            cur = (rows[i] >> x) & 1
            if cur != prev:
                transitions += 1
                prev = cur
    return transitions


def count_well_cells(rows: list[int]) -> int:
    """Depth of every well: an empty column cell whose left and right
    neighbors are both filled (walls count). Deeper wells score worse."""
    wells = 0
    for i, row in enumerate(rows):
        if not row:
            continue
        for x in range(FIELD_W):
            if (row >> x) & 1:
                continue
            left = x == 0 or (row >> (x - 1)) & 1
            right = x == FIELD_W - 1 or (row >> (x + 1)) & 1
            if left and right:
                wells += 1
    return wells


def count_holes_covered(rows: list[int]) -> tuple[int, int]:
    """(holes, covered cells above them). A hole is an empty cell with any
    filled cell above it in its column; its cover is the run of filled cells
    directly above it (counted once per hole, not per cell)."""
    holes = 0
    covered = 0
    for x in range(FIELD_W):
        col = [(rows[i] >> x) & 1 for i in range(FIELD_H)]  # top -> bottom
        seen = False
        i = 0
        while i < FIELD_H:
            if col[i]:
                seen = True
                i += 1
            elif seen:
                # a hole: count the filled run directly above it
                holes += 1
                j = i - 1
                while j >= 0 and col[j]:
                    covered += 1
                    j -= 1
                # skip the rest of this empty run (one hole per run)
                while i < FIELD_H and not col[i]:
                    i += 1
            else:
                i += 1
    return holes, covered


def eval_board(rows: list[int], weights: EvalWeights = EvalWeights()) -> float:
    """Score a board position: sum of features x weights (higher = better)."""
    w = weights
    heights = column_heights(rows)
    holes, covered = count_holes_covered(rows)
    # only a *partially*-filled bottom row has dig-target holes; an empty
    # bottom (mid-combo fully dug region, or a cleared board) is progress,
    # not a hole to penalize
    bottom = rows[FIELD_H - 1]
    empty_bottom = sum(1 for x in range(FIELD_W) if not (bottom >> x) & 1) if bottom else 0
    return (
        w.holes * holes
        + w.covered * covered
        + w.height * sum(heights)
        + w.bumpiness * sum(abs(heights[i] - heights[i + 1]) for i in range(FIELD_W - 1))
        + w.row_transitions * count_row_transitions(rows)
        + w.col_transitions * count_col_transitions(rows)
        + w.wells * count_well_cells(rows)
        + w.lowest_row_hole * empty_bottom
    )
