"""Pure-data constants for the Tetris engine (no pygame, no I/O).

Coordinate conventions — read this before touching anything:

- The playfield is a list of ``FIELD_H`` integer rows. Index 0 is the top of the
  buffer zone, index ``FIELD_H - 1`` is the bottom row. The bottom
  ``VISIBLE_H`` rows are the visible playfield; everything above is buffer.
- Each row is a 10-bit mask: bit ``x`` set means column ``x`` is occupied
  (bit 0 = leftmost column).
- Piece positions are ``(x, y)`` of the piece bounding box's top-left corner,
  with **+y pointing down** (screen convention).
- SRS wall-kick tables are stored verbatim from the Hard Drop wiki, where
  **+y points up**. ``game.py`` converts when applying a kick
  (screen dy = -kick dy). Tests pin the tables to the wiki data.
"""

from __future__ import annotations

from enum import IntEnum

FIELD_W = 10
FIELD_H = 40  # 20 buffer rows on top + 20 visible rows
VISIBLE_H = 20
VISIBLE_TOP = FIELD_H - VISIBLE_H  # first visible row index (20)
FULL_ROW = (1 << FIELD_W) - 1

TICKS_PER_SEC = 60  # fixed logic rate; all engine timings are integer ticks


def ms_to_ticks(ms: float) -> int:
    return round(ms * TICKS_PER_SEC / 1000)


class PieceType(IntEnum):
    I = 0
    J = 1
    L = 2
    O = 3
    S = 4
    T = 5
    Z = 6


ALL_PIECES: tuple[PieceType, ...] = tuple(PieceType)

# Bounding box size per piece (SRS: I lives in 4x4, O in 2x2, others in 3x3).
BOX_SIZE: dict[PieceType, int] = {
    PieceType.I: 4,
    PieceType.O: 2,
    PieceType.J: 3,
    PieceType.L: 3,
    PieceType.S: 3,
    PieceType.T: 3,
    PieceType.Z: 3,
}

# Spawn cell layout (bounding-box coordinates, +y down) for the spawn
# orientation (rot 0), flat-side down. Rotations are generated from these.
SPAWN_CELLS: dict[PieceType, tuple[tuple[int, int], ...]] = {
    PieceType.I: ((0, 1), (1, 1), (2, 1), (3, 1)),
    PieceType.O: ((0, 0), (1, 0), (0, 1), (1, 1)),
    PieceType.T: ((1, 0), (0, 1), (1, 1), (2, 1)),
    PieceType.S: ((1, 0), (2, 0), (0, 1), (1, 1)),
    PieceType.Z: ((0, 0), (1, 0), (1, 1), (2, 1)),
    PieceType.J: ((0, 0), (0, 1), (1, 1), (2, 1)),
    PieceType.L: ((2, 0), (0, 1), (1, 1), (2, 1)),
}

# Spawn position of the bounding box. Guideline: pieces spawn in rows 21-22
# counted 1-indexed from the bottom (rows 18-19 in this 0-indexed 40-row
# field), fully above the visible playfield, and drop into view on the first
# gravity step. Columns: I spans 3-6, O spans 4-5, JLSTZ "lean left" on 3-5.
SPAWN_Y = 18
SPAWN_X: dict[PieceType, int] = {
    PieceType.I: 3,
    PieceType.O: 4,
    PieceType.J: 3,
    PieceType.L: 3,
    PieceType.S: 3,
    PieceType.T: 3,
    PieceType.Z: 3,
}


def _rotate_cw(cells: tuple[tuple[int, int], ...], box: int) -> tuple[tuple[int, int], ...]:
    # Clockwise rotation inside an n x n box, +y down: (x, y) -> (n-1-y, x).
    return tuple(sorted((box - 1 - y, x) for x, y in cells))


def _build_rotations() -> dict[PieceType, tuple[tuple[tuple[int, int], ...], ...]]:
    out: dict[PieceType, tuple[tuple[tuple[int, int], ...], ...]] = {}
    for piece, spawn in SPAWN_CELLS.items():
        box = BOX_SIZE[piece]
        rots = [spawn]
        for _ in range(3):
            rots.append(_rotate_cw(rots[-1], box))
        out[piece] = tuple(rots)
    return out


# PIECE_CELLS[piece][rot] -> tuple of (cx, cy) bounding-box coordinates.
PIECE_CELLS = _build_rotations()

# Precomputed per-row masks for fast collision checks:
# PIECE_ROWS[piece][rot] -> tuple of (dy, mask) covering the piece's occupied
# rows within its bounding box.
PIECE_ROWS: dict[PieceType, tuple[tuple[tuple[int, int], ...], ...]] = {}
for _p, _rots in PIECE_CELLS.items():
    _rows_per_rot = []
    for _cells in _rots:
        _by_dy: dict[int, int] = {}
        for _x, _y in _cells:
            _by_dy[_y] = _by_dy.get(_y, 0) | (1 << _x)
        _rows_per_rot.append(tuple(sorted(_by_dy.items())))
    PIECE_ROWS[_p] = tuple(_rows_per_rot)

# ---------------------------------------------------------------------------
# SRS wall kicks — verbatim from https://harddrop.com/wiki/SRS
# Offsets are (dx, dy) with **+y = up** (wiki convention). Convert with
# screen_dy = -dy when applying.
# O piece never kicks (its rotation is a no-op in this implementation).
# ---------------------------------------------------------------------------

KICKS_JLSTZ: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {
    # 0->R
    (0, 1): ((0, 0), (-1, 0), (-1, +1), (0, -2), (-1, -2)),
    # R->0
    (1, 0): ((0, 0), (+1, 0), (+1, -1), (0, +2), (+1, +2)),
    # R->2
    (1, 2): ((0, 0), (+1, 0), (+1, -1), (0, +2), (+1, +2)),
    # 2->R
    (2, 1): ((0, 0), (-1, 0), (-1, +1), (0, -2), (-1, -2)),
    # 2->L
    (2, 3): ((0, 0), (+1, 0), (+1, +1), (0, -2), (+1, -2)),
    # L->2
    (3, 2): ((0, 0), (-1, 0), (-1, -1), (0, +2), (-1, +2)),
    # L->0
    (3, 0): ((0, 0), (-1, 0), (-1, -1), (0, +2), (-1, +2)),
    # 0->L
    (0, 3): ((0, 0), (+1, 0), (+1, +1), (0, -2), (+1, -2)),
}

KICKS_I: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {
    # 0->R
    (0, 1): ((0, 0), (-2, 0), (+1, 0), (-2, -1), (+1, +2)),
    # R->0
    (1, 0): ((0, 0), (+2, 0), (-1, 0), (+2, +1), (-1, -2)),
    # R->2
    (1, 2): ((0, 0), (-1, 0), (+2, 0), (-1, +2), (+2, -1)),
    # 2->R
    (2, 1): ((0, 0), (+1, 0), (-2, 0), (+1, -2), (-2, +1)),
    # 2->L
    (2, 3): ((0, 0), (+2, 0), (-1, 0), (+2, +1), (-1, -2)),
    # L->2
    (3, 2): ((0, 0), (-2, 0), (+1, 0), (-2, -1), (+1, +2)),
    # L->0
    (3, 0): ((0, 0), (+1, 0), (-2, 0), (+1, -2), (-2, +1)),
    # 0->L
    (0, 3): ((0, 0), (-1, 0), (+2, 0), (-1, +2), (+2, -1)),
}

# 180-degree rotation kicks (SRS+ table, as used by Jstris' 180 option and
# TETR.IO — osk's published "180 KICK TABLE"). Standard SRS has no 180
# rotation; modern games add it with this universal table (applies to every
# piece, I included). Offsets (dx, dy), **+y = up** like the tables above:
#   N->S "off-the-floor", S->N "off-the-ceiling", E/W->W/E horizontal unhooks
#   and 2-up "jumps" (there are no downward 180 kicks on horizontal turns).
KICKS_180: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {
    # 0->2 (North->South)
    (0, 2): ((0, 0), (0, +1), (+1, +1), (-1, +1), (+1, 0), (-1, 0)),
    # 2->0 (South->North)
    (2, 0): ((0, 0), (0, -1), (-1, -1), (+1, -1), (-1, 0), (+1, 0)),
    # 1->3 (East->West): off-the-wall, tall jump, constricted jump,
    #   tall zipper (2 up), constricted zipper (1 up)
    (1, 3): ((0, 0), (+1, 0), (+1, +2), (+1, +1), (0, +2), (0, +1)),
    # 3->1 (West->East): mirror of East->West
    (3, 1): ((0, 0), (-1, 0), (-1, +2), (-1, +1), (0, +2), (0, +1)),
}

# T-spin mini/full detection data (Tetris-Friends ruleset, as used by Jstris).
# The 3x3 box corners, indexed 0..3: 0=(0,0) top-left, 1=(2,0) top-right,
# 2=(0,2) bottom-left, 3=(2,2) bottom-right (box coords, +y down).
T_BOX_CORNERS: tuple[tuple[int, int], ...] = ((0, 0), (2, 0), (0, 2), (2, 2))
# "Front" corners = the two diagonal cells adjacent to the pointing side.
# rot 0 points up -> top corners; rot 1 right -> right corners; etc.
T_FRONT_CORNERS: dict[int, tuple[int, int]] = {0: (0, 1), 1: (1, 3), 2: (2, 3), 3: (0, 2)}

# Kick tests whose use "upgrades" a mini T-spin to a full one (Tetris
# Friends / Jstris: the T-Spin Triple kick — the last kick test, applied on
# the 0=>R and 2=>L transitions).
TSPIN_UPGRADE_KICKS: dict[tuple[int, int], frozenset[int]] = {
    (0, 1): frozenset({4}),
    (2, 3): frozenset({4}),
}

# Renderer palette (plain RGB tuples; no pygame types here).
PIECE_COLORS: dict[PieceType, tuple[int, int, int]] = {
    PieceType.I: (49, 199, 239),
    PieceType.J: (90, 101, 173),
    PieceType.L: (227, 91, 2),
    PieceType.O: (247, 211, 8),
    PieceType.S: (66, 182, 66),
    PieceType.T: (173, 77, 156),
    PieceType.Z: (239, 32, 41),
}
GARBAGE_COLOR = (118, 124, 134)
