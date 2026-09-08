"""Movegen: exhaustive reachable-placement enumeration.

Empty-board counts are checked against shapes derived analytically from
PIECE_CELLS (each distinct shape contributes 11 - width columns), which
validates reachability + dedup without hand-counting. The TSD fixture
validates the headline feature: a spin-in placement that no drop-column
enumeration could ever find, with its engine-matching spin verdict.
"""

from tetris.ai import enumerate_placements
from tetris.ai.movegen import CLS_PLAIN, CLS_ROT, CLS_UPGRADE, spin_class
from tetris.engine import board as B
from tetris.engine.constants import PIECE_CELLS, PieceType

EMPTY = [0] * 40

# Classic T-spin double slot: notch over rows 38-39 with a single
# overhang block at (37, 3). A vertical T (nub toward the open side)
# soft-drops down the col-4 shaft to rest, then rotates CW with kick 0
# into the slot — the position is below the free-fall ghost, so no drop
# or shift ever reaches it. (from_ascii art must be flush-left: leading
# spaces are empty cells, not indentation.)
TSD_BOARD = B.from_ascii(
    """...#......
###...####
####.#####"""
)


def _empty_board_expected(piece: PieceType) -> int:
    """On an empty board every column hosts exactly one placement per
    distinct (x-, y-normalized) shape: 11 - width resting columns."""
    shapes = {
        tuple(sorted((x - min(cx for cx, _ in c), y - min(cy for _, cy in c)) for x, y in c))
        for c in PIECE_CELLS[piece]
    }
    return sum(11 - (max(x for x, _ in s) + 1) for s in shapes)


def test_empty_board_counts_match_piece_shapes():
    for piece in PieceType:
        placements = enumerate_placements(EMPTY, piece)
        assert len(placements) == _empty_board_expected(piece), piece.name
        # everything rests on the floor and nothing spins on a flat field
        assert all(max(r for r, _ in p.cells) == 39 for p in placements)
        assert all(p.spin == "none" for p in placements)


def test_expected_empty_board_counts_pinned():
    # analytic ground truth, pinned so a table regression is loud:
    # I 17 (7 flat + 10 vertical), O 9, T 34, S/Z 17 (2 distinct shapes),
    # J/L 34 (4 distinct shapes each).
    pinned = {PieceType.I: 17, PieceType.O: 9, PieceType.T: 34, PieceType.S: 17,
              PieceType.Z: 17, PieceType.J: 34, PieceType.L: 34}
    for piece, count in pinned.items():
        assert _empty_board_expected(piece) == count, piece.name
        assert len(enumerate_placements(EMPTY, piece)) == count, piece.name


def test_dedup_180_symmetric_rotations():
    # I/S/Z rot0 and rot2 (and all O rotations) lock identical cells; each
    # cell set must appear exactly once.
    for piece in (PieceType.I, PieceType.O, PieceType.S, PieceType.Z):
        placements = enumerate_placements(EMPTY, piece)
        keys = [p.key for p in placements]
        assert len(keys) == len(set(keys))


def test_deterministic_sorted_output():
    for piece in PieceType:
        a = enumerate_placements(TSD_BOARD, piece)
        b = enumerate_placements(list(TSD_BOARD), piece)
        assert a == b
        assert a == sorted(a, key=lambda p: (p.y, p.x, p.rot))


def test_tsd_spin_in_placement_is_found_and_classified_full():
    placements = enumerate_placements(TSD_BOARD, PieceType.T)
    slot = [p for p in placements if p.cells == ((38, 3), (38, 4), (38, 5), (39, 4))]
    assert len(slot) == 1
    assert slot[0].rot == 2 and slot[0].x == 3 and slot[0].y == 37
    assert slot[0].spin == "full"


def test_tsd_board_still_lists_plain_drops_above_the_overhang():
    placements = enumerate_placements(TSD_BOARD, PieceType.T)
    # spin classification spans the full spectrum here: plain drops, a mini
    # (flat T over the shaft entered by rotation), and the full TSD — the
    # engine really would score each of them that way at lock.
    assert {p.spin for p in placements} == {"none", "mini", "full"}
    # a flat drop resting on the row-37 overhang exists (not only spin-ins)
    assert any(p.rot == 0 and p.y == 35 and p.spin == "none" for p in placements)


def test_blocked_spawn_yields_no_placements():
    full = [(1 << 10) - 1] * 40
    assert enumerate_placements(full, PieceType.T) == []
    # spawn row itself blocked is enough (rows 18-19 host the spawn box)
    top_filled = [0] * 18 + [(1 << 10) - 1] * 22
    assert enumerate_placements(top_filled, PieceType.T) == []


def test_allow_180_flag():
    # on an open board 180 rotation unlocks nothing new
    for piece in PieceType:
        with_180 = enumerate_placements(EMPTY, piece, allow_180=True)
        without = enumerate_placements(EMPTY, piece, allow_180=False)
        assert {p.key for p in with_180} == {p.key for p in without}


def test_spin_class_mirrors_engine_rules():
    # T at (3, 37) rot 0 (nub up): box corners (37,3) (37,5) (39,3) (39,5);
    # front corners for rot 0 are the two top ones.
    rows = [0] * 40
    rows[39] |= 0b101 << 3  # bottom corners filled (cols 3 and 5)
    rows[37] |= 1 << 3  # one top corner -> 3 total, front split
    assert spin_class(rows, PieceType.T, 0, 3, 37, CLS_PLAIN) == "none"
    assert spin_class(rows, PieceType.T, 0, 3, 37, CLS_ROT) == "mini"
    assert spin_class(rows, PieceType.T, 0, 3, 37, CLS_UPGRADE) == "full"
    rows[37] |= 1 << 5  # both top corners -> 4 total, front both filled
    assert spin_class(rows, PieceType.T, 0, 3, 37, CLS_ROT) == "full"
    # upgrade kick wins even with under 3 corners
    rows[39] = 0
    rows[37] = 0
    assert spin_class(rows, PieceType.T, 0, 3, 37, CLS_ROT) == "none"
    assert spin_class(rows, PieceType.T, 0, 3, 37, CLS_UPGRADE) == "full"
    # non-T pieces never spin
    rows[39] |= 0b101 << 3
    assert spin_class(rows, PieceType.J, 0, 3, 37, CLS_ROT) == "none"
