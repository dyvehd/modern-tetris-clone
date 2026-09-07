"""Playfield helpers operating on row bitmasks (see constants for layout).

The board is a plain ``list[int]`` of length FIELD_H. Everything here is a
module-level function on plain data so the AI layer (and a later
Numba/Cython port) can work without object overhead.
"""

from __future__ import annotations

from .constants import (
    EDITOR_GRAY,
    FIELD_H,
    FIELD_W,
    FULL_ROW,
    PIECE_CELLS,
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


def gray_component(
    styles: list[list[int]], ry: int, x: int, max_size: int = 5
) -> list[tuple[int, int]] | None:
    """The 4-connected cells of editor-gray style containing (ry, x).

    Returns None as soon as the component exceeds ``max_size`` cells (so a
    large gray region never gets auto-colored). Assumes (ry, x) itself is
    editor-gray.
    """
    seen = {(ry, x)}
    out: list[tuple[int, int]] = [(ry, x)]
    queue = [(ry, x)]
    while queue:
        cy, cx = queue.pop()
        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
            if (ny, nx) in seen or not (0 <= ny < FIELD_H and 0 <= nx < FIELD_W):
                continue
            if styles[ny][nx] == EDITOR_GRAY:
                seen.add((ny, nx))
                out.append((ny, nx))
                if len(out) > max_size:
                    return None
                queue.append((ny, nx))
    return out


def piece_from_cells(cells) -> PieceType | None:
    """The tetromino exactly matching a set of 4 connected cells, else None.

    Any connected 4-cell polyomino is one of the 7 tetrominoes, so the match
    is unique (chirality included: an L-shaped set is an L or a J, never
    both). Cells are (row, col) pairs; they are translated and compared
    against every rotation of every piece, so position/orientation don't
    matter — only the shape does.
    """
    cells = sorted(set(cells))
    if len(cells) != 4:
        return None
    # connected? (every cell must reach every other via 4-directional steps)
    reach = {cells[0]}
    queue = [cells[0]]
    while queue:
        cy, cx = queue.pop()
        for nb in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
            if nb in cells and nb not in reach:
                reach.add(nb)
                queue.append(nb)
    if len(reach) != 4:
        return None
    min_y = min(y for _, y in cells)
    min_x = min(x for x, _ in cells)
    norm = tuple(sorted((y - min_y, x - min_x) for x, y in cells))
    for piece, rots in PIECE_CELLS.items():
        for rot_cells in rots:
            rmin_y = min(y for _, y in rot_cells)
            rmin_x = min(x for x, _ in rot_cells)
            rnorm = tuple(sorted((y - rmin_y, x - rmin_x) for x, y in rot_cells))
            if rnorm == norm:
                return piece
    return None
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
