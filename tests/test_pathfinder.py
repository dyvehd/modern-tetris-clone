"""Pathfinder: target placement -> Action sequence, verified by replaying
through the real engine.

The triangle test is the heart: for every placement the movegen enumerates
(movegen x pathfinder x engine), a path must exist, replaying it through a
0G ``Game`` (infinite SDF, so ``Action.SOFT_DROP`` is a sonic drop) must
lock the piece exactly as predicted, and the engine's own spin detection
at lock must match ``Placement.spin``. Movegen and pathfinder share their
movement-graph code, so this cross-validates the AI layer against the
engine's independent implementation of the same physics — the engine is
the rule authority.
"""

import math

import pytest
from conftest import art_rows, make_game

from tetris.ai import enumerate_placements, find_path
from tetris.ai.movegen import Placement
from tetris.engine.constants import FULL_ROW, PIECE_CELLS, PieceType
from tetris.engine.game import Action

TSD_BOARD = art_rows(
    """...#......
###...####
####.#####"""
)

# a pile with holes and ledges — movement paths must dodge it
MESSY_BOARD = art_rows(
    """..........
..........
..........
..........
.....#....
..#..#....
.##.###...
##.####.#.
#.####.#..
#.#####..#"""
)

BOARDS = {"empty": art_rows(""), "tsd": TSD_BOARD, "messy": MESSY_BOARD}


def _expected_lines(rows: list[int], placement: Placement) -> int:
    """Rows the placement completes, by direct board arithmetic."""
    merged = list(rows)
    for cx, cy in PIECE_CELLS[placement.piece][placement.rot]:
        merged[placement.y + cy] |= 1 << (placement.x + cx)
    return sum(1 for r in merged if r == FULL_ROW)


def _assert_replay(rows: list[int], piece: PieceType, placement: Placement, path: list[Action]) -> None:
    """Play ``path`` on a fresh 0G infinite-SDF game and demand the lock
    matches the prediction exactly (cells, lines, spin class)."""
    game = make_game(soft_drop_factor=math.inf)
    game.set_rows(list(rows))
    game.spawn_forced(piece)
    for action in path:
        game.tick([action])
    assert game.pieces_placed == 1, (piece, placement, path)
    assert not game.over, (piece, placement, path)  # low boards: no lockout

    lines = _expected_lines(rows, placement)
    ev = next((e for e in game.events if e["kind"] in ("clear", "tspin")), None)
    if lines == 0 and placement.spin == "none":
        assert ev is None, (placement, ev)
        pre = {(i, x) for i, row in enumerate(rows) for x in range(10) if row >> x & 1}
        post = {
            (i, x) for i, row in enumerate(game.rows) for x in range(10) if row >> x & 1
        }
        assert post - pre == set(placement.cells), (placement, path)
        return
    assert ev is not None, (placement, path)
    if ev["kind"] == "tspin":
        assert lines == 0 and placement.spin != "none", (placement, ev)
    else:
        assert ev["lines"] == lines, (placement, ev)

    label = ev["label"]
    if placement.spin == "full":
        assert label.startswith("T-SPIN") and "MINI" not in label, (placement, label)
    elif placement.spin == "mini":
        if lines >= 2:  # Jstris: mini doubles and up count as full
            assert label.startswith("T-SPIN") and "MINI" not in label, (placement, label)
        else:
            assert label.startswith("T-SPIN MINI"), (placement, label)
    else:
        assert "T-SPIN" not in label, (placement, label)
    assert game.tspins == (placement.spin != "none"), (placement, label)


@pytest.mark.parametrize("board_name", list(BOARDS))
@pytest.mark.parametrize("piece", list(PieceType))
def test_triangle_every_placement_replays_exactly(board_name, piece):
    rows = BOARDS[board_name]
    placements = enumerate_placements(list(rows), piece)
    assert placements, (board_name, piece)
    for p in placements:
        path = find_path(list(rows), piece, p)
        assert path is not None, (board_name, piece, p)
        assert path[-1] is Action.HARD_DROP
        _assert_replay(rows, piece, p, path)


def test_spawn_column_target_is_bare_hard_drop():
    # the flat placement straight below spawn: the spawn state's ghost
    # already equals the target rest, so the shortest path is one input
    rows = BOARDS["empty"]
    placements = enumerate_placements(rows, PieceType.T)
    flat = next(p for p in placements if p.rot == 0 and p.x == 3)
    assert find_path(rows, PieceType.T, flat) == [Action.HARD_DROP]
    # a shift target is reached by shifting at spawn height, then dropping
    one_left = next(p for p in placements if p.rot == 0 and p.x == 2)
    assert find_path(rows, PieceType.T, one_left) == [Action.LEFT, Action.HARD_DROP]


def test_paths_are_deterministic_and_hold_free():
    rows = MESSY_BOARD
    for piece in PieceType:
        for p in enumerate_placements(rows, piece):
            a = find_path(rows, piece, p)
            b = find_path(list(rows), piece, p)
            assert a == b
            assert a is not None and Action.HOLD not in a
            assert len(a) <= 40, (piece, p)


def test_unreachable_target_returns_none():
    # a hand-crafted placement inside a sealed pocket: no path exists
    sealed = art_rows(
        """#........#
#........#
#........#
#........#
##########"""
    )
    bogus = Placement(
        piece=PieceType.O,
        rot=0,
        x=2,
        y=35,
        spin="none",
        cells=((35, 2), (35, 3), (36, 2), (36, 3)),
    )
    assert find_path(sealed, PieceType.O, bogus) is None


def test_tsd_path_is_spin_entry():
    rows = TSD_BOARD
    placements = enumerate_placements(rows, PieceType.T)
    fulls = [p for p in placements if p.spin == "full"]
    # two legitimate spin-ins here: the TSD itself (rot 2) and a left-facing
    # spin (rot 3) whose cells form a vertical line at the shaft wall —
    # both are real engine-verified placements, the rot 3 one is simply a
    # poorer move (it fills the shaft's side column)
    tsd = next(p for p in fulls if p.rot == 2)
    assert tsd.cells == ((38, 3), (38, 4), (38, 5), (39, 4))
    assert len(fulls) == 2
    path = find_path(rows, PieceType.T, tsd)
    assert path is not None
    # a modern T-spin ends with rotation -> hard drop; the sonic descent to
    # the pre-rotation rest is the entry
    assert path[-2] is Action.ROT_CW
    assert Action.SOFT_DROP in path
    _assert_replay(rows, PieceType.T, tsd, path)
    game = make_game(soft_drop_factor=math.inf)
    game.set_rows(list(rows))
    game.spawn_forced(PieceType.T)
    for action in path:
        game.tick([action])
    ev = next(e for e in game.events if e["kind"] == "clear")
    assert ev["label"] == "T-SPIN DOUBLE"
    assert ev["lines"] == 2


def test_allow_180_flag():
    rows = MESSY_BOARD
    placements = enumerate_placements(rows, PieceType.T)
    used_180 = False
    for p in placements:
        path = find_path(rows, PieceType.T, p, allow_180=False)
        if path is not None:
            assert Action.ROT_180 not in path
            game = make_game(soft_drop_factor=math.inf)
            game.set_rows(list(rows))
            game.spawn_forced(PieceType.T)
            for action in path:
                game.tick([action])
            assert game.pieces_placed == 1
        full = find_path(rows, PieceType.T, p)
        assert full is not None
        if Action.ROT_180 in full:
            used_180 = True
    # with 180 enabled, flipping orientation costs one input, so at least
    # one rot-2/rot-0 target genuinely uses it
    assert used_180
