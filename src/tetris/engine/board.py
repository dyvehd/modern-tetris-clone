"""Playfield helpers operating on row bitmasks (see constants for layout).

The board is a plain ``list[int]`` of length FIELD_H. Everything here is a
module-level function on plain data so the AI layer (and a later
Numba/Cython port) can work without object overhead.
"""

from __future__ import annotations

from .constants import (
    FIELD_H,
    FIELD_W,
    FULL_ROW,
    PIECE_ROWS,
    PieceType,
)


def _shift(mask: int, x: int) -> int | None:
    """Shift a row mask horizontally; None means it left the field."""
    if x >= 0:
        if mask << x >= 1 << FIELD_W:
            return None
        return mask << x
    if mask & ((1 << -x) - 1):  # cells fall off the left edge
        return None
    return mask >> -x


def collides(rows: list[int], piece: PieceType, rot: int, x: int, y: int) -> bool:
    """Would the piece at (x, y) overlap the stack or leave the field?"""
    for dy, mask in PIECE_ROWS[piece][rot]:
        ry = y + dy
        if ry < 0 or ry >= FIELD_H:
            return True
        shifted = _shift(mask, x)
        if shifted is None or rows[ry] & shifted:
            return True
    return False


def ghost_y(rows: list[int], piece: PieceType, rot: int, x: int, y: int) -> int:
    """Lowest valid y for the piece (hard-drop / ghost position)."""
    while not collides(rows, piece, rot, x, y + 1):
        y += 1
    return y


def merge_piece(rows: list[int], piece: PieceType, rot: int, x: int, y: int) -> None:
    for dy, mask in PIECE_ROWS[piece][rot]:
        shifted = _shift(mask, x)
        assert shifted is not None, "merge called on an out-of-field position"
        rows[y + dy] |= shifted


def full_rows(rows: list[int]) -> list[int]:
    return [i for i, row in enumerate(rows) if row == FULL_ROW]


def clear_rows(rows: list[int], indices: list[int]) -> None:
    """Remove the given row indices and shift everything above down."""
    if not indices:
        return
    drop = set(indices)
    kept = [row for i, row in enumerate(rows) if i not in drop]
    rows[:] = [0] * (FIELD_H - len(kept)) + kept


def garbage_row(hole: int) -> int:
    """A garbage row: all columns filled except ``hole``."""
    return FULL_ROW & ~(1 << hole)


def from_ascii(art: str) -> list[int]:
    """Build a 40-row field from bottom-anchored ASCII art.

    Only the last lines matter; art is aligned to the bottom of the field.
    '#' or 'X' = filled, '.' or ' ' = empty. Lines beyond 40 are ignored.
    """
    lines = [ln for ln in art.strip("\n").splitlines()]
    rows = [0] * FIELD_H
    base = FIELD_H - len(lines)
    for i, line in enumerate(lines):
        ry = base + i
        if ry < 0:
            break
        for x, ch in enumerate(line[:FIELD_W]):
            if ch in "#Xx":
                rows[ry] |= 1 << x
    return rows


def to_ascii(rows: list[int], visible_only: bool = False) -> str:
    start = FIELD_H - 20 if visible_only else 0
    out = []
    for row in rows[start:]:
        out.append("".join("#" if row >> x & 1 else "." for x in range(FIELD_W)))
    return "\n".join(out)
