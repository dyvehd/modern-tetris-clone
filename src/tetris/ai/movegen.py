"""Movement-graph placement enumeration (movegen) for the AI layer.

Given a playfield and a piece, enumerate every resting placement that is
actually reachable by legal inputs — shifts, SRS rotations (kicks
included, optionally 180) and sonic drops (infinite soft drop) — starting
from the standard spawn state. Spin-ins are found because this is a BFS
over movement states, not a drop-column enumeration: a T-spin slot that no
fall can enter shows up here with its spin classification, and nothing
unreachable ever does.

Finite soft drop (stopping mid-column) is deliberately NOT an edge. Nearly
all competitive play uses infinite SDF, and placements that require a
mid-air stop are rare and generally poor moves — the cost of losing them
is far smaller than the cost of carrying finite-SDF timing/replay edge
cases through the whole AI stack. Every T-spin entry survives the cut:
the pre-rotation rest is always a column ghost (sonic-reachable), the slot
entry is the kick, and the lock is a zero-distance hard drop.

The engine is the rule authority. This module mirrors ``Game._try_shift``
/ ``_try_rotate`` and the infinite-SDF branch of ``_apply_gravity``
exactly (same kick tables, only the first fitting kick applies, O never
rotates) and mirrors the engine's spin bookkeeping: a placement only
counts as a spin when the piece's final movement into the resting
position was a rotation. Hard drop preserves that (``Game._hard_drop``
never touches ``last_action``), so any state above the free-fall ghost
carries its arrival class onto the ghost placement; shifts and gravity
fall cancel it.

Movement states are tiny and closed: at most 4 rotations x 13 x-values x
~26 rows, deduped by a visited set — there is no state explosion to
contain. A full enumeration costs on the order of a thousand collision
checks.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from ..engine import board as B
from ..engine.constants import (
    FIELD_H,
    FIELD_W,
    KICKS_180,
    KICKS_I,
    KICKS_JLSTZ,
    PIECE_CELLS,
    SPAWN_X,
    SPAWN_Y,
    T_BOX_CORNERS,
    T_FRONT_CORNERS,
    TSPIN_UPGRADE_KICKS,
    PieceType,
)
from ..engine.game import Action

# Arrival classes mirror the engine's last_action / last_kick_index state
# that feeds spin detection at lock time.
CLS_PLAIN = 0  # arrived by a shift or soft drop — never a spin
CLS_ROT = 1  # arrived by a rotation — spin-eligible
CLS_UPGRADE = 2  # arrived by an upgrading rotation (TST kick): always full

_CLS_BIT = {CLS_PLAIN: 1, CLS_ROT: 2, CLS_UPGRADE: 4}

_ROT_DELTAS = (1, -1, 2)  # CW, CCW, 180


@dataclass(frozen=True)
class Placement:
    """One reachable resting placement, in ActivePiece coordinates."""

    piece: PieceType
    rot: int
    x: int
    y: int
    spin: str  # "none" | "mini" | "full" — the engine's verdict at lock
    cells: tuple[tuple[int, int], ...]  # absolute (row, col), sorted

    @property
    def key(self) -> tuple:
        """Identity after locking: cell set + spin (I/S/Z/O rotations that
        occupy identical cells are the same placement)."""
        return (self.cells, self.spin)


def spin_class(rows: list[int], piece: PieceType, rot: int, x: int, y: int, cls: int) -> str:
    """Spin verdict for a resting piece at (x, y, rot) whose final movement
    into the position had arrival class ``cls``. Mirrors ``Game._detect_tspin``
    (Tetris-Friends ruleset, as Jstris: T only, 3-corner rule, front corners
    split mini/full, TST-kick upgrade overrides both)."""
    if piece is not PieceType.T or cls == CLS_PLAIN:
        return "none"
    occupied = []
    for ox, oy in T_BOX_CORNERS:
        gx, gy = x + ox, y + oy
        occupied.append(
            gx < 0 or gx >= FIELD_W or gy < 0 or gy >= FIELD_H or (rows[gy] >> gx) & 1
        )
    if cls == CLS_UPGRADE:
        return "full"  # the TST kick upgrades regardless of the corner split
    if sum(occupied) >= 3:
        f1, f2 = T_FRONT_CORNERS[rot]
        if occupied[f1] and occupied[f2]:
            return "full"
        return "mini"
    return "none"


def _rotation_target(
    rows: list[int], piece: PieceType, rot: int, x: int, y: int, delta: int, allow_180: bool
) -> tuple[int, int, int, int] | None:
    """Result state of a rotation, or None. Mirrors ``Game._try_rotate``:
    only the first fitting kick applies, O never rotates."""
    if piece is PieceType.O:
        return None
    if delta == 2 and not allow_180:
        return None
    new_rot = (rot + delta) % 4
    if delta == 2:
        table = KICKS_180
    elif piece is PieceType.I:
        table = KICKS_I
    else:
        table = KICKS_JLSTZ
    for i, (kx, ky) in enumerate(table.get((rot, new_rot), ((0, 0),))):
        nx, ny = x + kx, y - ky  # kick tables use +y = up (wiki convention)
        if not B.collides(rows, piece, new_rot, nx, ny):
            upgrade = i in TSPIN_UPGRADE_KICKS.get((rot, new_rot), frozenset())
            return (new_rot, nx, ny, CLS_UPGRADE if upgrade else CLS_ROT)
    return None


def successors(
    rows: list[int], piece: PieceType, rot: int, x: int, y: int, allow_180: bool
) -> list[tuple[int, int, int, int, Action]]:
    """All movement edges from a state as (rot, x, y, arrival_cls, action).
    The downward edge is a sonic drop: all the way to the column rest in
    one input (the infinite-SDF branch of ``Game._apply_gravity``)."""
    out = []
    for dx, action in ((-1, Action.LEFT), (1, Action.RIGHT)):
        nx = x + dx
        if not B.collides(rows, piece, rot, nx, y):
            out.append((rot, nx, y, CLS_PLAIN, action))
    ny = B.ghost_y(rows, piece, rot, x, y)
    if ny > y:
        out.append((rot, x, ny, CLS_PLAIN, Action.SOFT_DROP))
    deltas: tuple[tuple[int, Action], ...] = (
        (1, Action.ROT_CW), (-1, Action.ROT_CCW), (2, Action.ROT_180)
    )
    for delta, action in (deltas if allow_180 else deltas[:2]):
        t = _rotation_target(rows, piece, rot, x, y, delta, allow_180)
        if t is not None:
            out.append((*t, action))
    return out


def _movement_masks(
    rows: list[int], piece: PieceType, allow_180: bool
) -> dict[tuple[int, int, int], int]:
    """BFS from spawn over (rot, x, y) states; value = bitmask of the
    arrival classes the state is reachable with. A state is re-enqueued
    when it gains a class so the new class keeps propagating (successor
    classes depend only on the edge, never on the parent's class)."""
    if B.collides(rows, piece, 0, SPAWN_X[piece], SPAWN_Y):
        return {}  # block out: the piece cannot spawn at all
    start = (0, SPAWN_X[piece], SPAWN_Y)
    masks: dict[tuple[int, int, int], int] = {start: _CLS_BIT[CLS_PLAIN]}
    queue = deque([start])
    while queue:
        state = queue.popleft()
        rot, x, y = state
        for n_rot, nx, ny, cls, _action in successors(rows, piece, rot, x, y, allow_180):
            key = (n_rot, nx, ny)
            bit = _CLS_BIT[cls]
            if masks.get(key, 0) & bit:
                continue
            masks[key] = masks.get(key, 0) | bit
            queue.append(key)
    return masks


def enumerate_placements(
    rows: list[int], piece: PieceType, *, allow_180: bool = True
) -> list[Placement]:
    """All reachable resting placements of ``piece`` on ``rows``.

    Sorted by (y, x, rot); duplicates that lock identical cells with the
    same spin (I/O/S/Z 180-symmetric rotations) are collapsed to one entry.
    The reported spin is the best class the position admits.
    """
    masks = _movement_masks(rows, piece, allow_180)
    if not masks:
        return []

    # Free-fall ghost per (rot, x) column, from the highest visited state:
    # every state at or above it hard-drops onto the ghost rest.
    col_top: dict[tuple[int, int], int] = {}
    for rot, x, y in masks:
        key = (rot, x)
        if key not in col_top or y < col_top[key]:
            col_top[key] = y
    col_ghost = {
        key: B.ghost_y(rows, piece, key[0], key[1], y) for key, y in col_top.items()
    }
    col_bits: dict[tuple[int, int], int] = {key: 0 for key in col_ghost}
    for (rot, x, y), bits in masks.items():
        if y <= col_ghost[(rot, x)]:
            col_bits[(rot, x)] |= bits

    out: list[Placement] = []
    seen: set[tuple] = set()
    for rot, x, y in sorted(masks):
        if not B.collides(rows, piece, rot, x, y + 1):
            continue  # not resting — just a waypoint state
        bits = masks[(rot, x, y)]
        if y == col_ghost[(rot, x)]:
            bits |= col_bits[(rot, x)]  # hard-drop preserves the arrival class
        cls = (
            CLS_UPGRADE
            if bits & _CLS_BIT[CLS_UPGRADE]
            else CLS_ROT
            if bits & _CLS_BIT[CLS_ROT]
            else CLS_PLAIN
        )
        spin = spin_class(rows, piece, rot, x, y, cls)
        cells = tuple(sorted((y + cy, x + cx) for cx, cy in PIECE_CELLS[piece][rot]))
        key = (cells, spin)
        if key in seen:
            continue
        seen.add(key)
        out.append(Placement(piece, rot, x, y, spin, cells))
    out.sort(key=lambda p: (p.y, p.x, p.rot))
    return out
