"""Piece rotation matrices, spawn states, and SRS kick tables.

The kick tables here are transcribed verbatim from
https://harddrop.com/wiki/SRS (offsets (dx, dy) with +y = UP).
"""

from tetris.engine.constants import (
    KICKS_I,
    KICKS_JLSTZ,
    PIECE_CELLS,
    SPAWN_X,
    SPAWN_Y,
    PieceType,
)


def cells(piece, rot):
    return set(PIECE_CELLS[PieceType[piece]][rot])


# --- rotation matrices (SRS diagrams, box coords, +y down) -----------------

def test_t_rotations():
    assert cells("T", 0) == {(1, 0), (0, 1), (1, 1), (2, 1)}
    assert cells("T", 1) == {(1, 0), (1, 1), (2, 1), (1, 2)}
    assert cells("T", 2) == {(0, 1), (1, 1), (2, 1), (1, 2)}
    assert cells("T", 3) == {(1, 0), (0, 1), (1, 1), (1, 2)}


def test_i_rotations():
    assert cells("I", 0) == {(0, 1), (1, 1), (2, 1), (3, 1)}
    assert cells("I", 1) == {(2, 0), (2, 1), (2, 2), (2, 3)}
    assert cells("I", 2) == {(0, 2), (1, 2), (2, 2), (3, 2)}
    assert cells("I", 3) == {(1, 0), (1, 1), (1, 2), (1, 3)}


def test_o_is_rotation_invariant():
    assert cells("O", 0) == cells("O", 1) == cells("O", 2) == cells("O", 3)
    assert cells("O", 0) == {(0, 0), (1, 0), (0, 1), (1, 1)}


def test_jlstz_spawn_states():
    assert cells("J", 0) == {(0, 0), (0, 1), (1, 1), (2, 1)}
    assert cells("L", 0) == {(2, 0), (0, 1), (1, 1), (2, 1)}
    assert cells("S", 0) == {(1, 0), (2, 0), (0, 1), (1, 1)}
    assert cells("Z", 0) == {(0, 0), (1, 0), (1, 1), (2, 1)}


def test_jlstz_180_states():
    assert cells("J", 2) == {(0, 1), (1, 1), (2, 1), (2, 2)}
    assert cells("L", 2) == {(0, 1), (1, 1), (2, 1), (0, 2)}
    assert cells("S", 2) == {(1, 1), (2, 1), (0, 2), (1, 2)}
    assert cells("Z", 2) == {(0, 1), (1, 1), (1, 2), (2, 2)}


# --- spawn states -----------------------------------------------------------

def test_spawn_positions():
    # Guideline: rows 21-22 counted from the bottom (here 0-indexed rows
    # 18-19 of the 40-row field), fully above the visible playfield.
    for piece in PieceType:
        assert SPAWN_X[piece] in (3, 4)
    game_active = cells("I", 0)
    assert SPAWN_X[PieceType.I] == 3 and SPAWN_Y == 18
    # I spawns on the lower spawn row (row 21 from bottom = row index 19)
    assert all(y == 1 for _, y in game_active)
    # JLSTZ lean left (columns 3-5), O central (columns 4-5)
    assert SPAWN_X[PieceType.O] == 4
    assert all(SPAWN_X[p] == 3 for p in PieceType if p not in (PieceType.I, PieceType.O))


# --- SRS kick tables (verbatim wiki data) -----------------------------------

WIKI_JLSTZ = {
    (0, 1): ((0, 0), (-1, 0), (-1, +1), (0, -2), (-1, -2)),
    (1, 0): ((0, 0), (+1, 0), (+1, -1), (0, +2), (+1, +2)),
    (1, 2): ((0, 0), (+1, 0), (+1, -1), (0, +2), (+1, +2)),
    (2, 1): ((0, 0), (-1, 0), (-1, +1), (0, -2), (-1, -2)),
    (2, 3): ((0, 0), (+1, 0), (+1, +1), (0, -2), (+1, -2)),
    (3, 2): ((0, 0), (-1, 0), (-1, -1), (0, +2), (-1, +2)),
    (3, 0): ((0, 0), (-1, 0), (-1, -1), (0, +2), (-1, +2)),
    (0, 3): ((0, 0), (+1, 0), (+1, +1), (0, -2), (+1, -2)),
}

WIKI_I = {
    (0, 1): ((0, 0), (-2, 0), (+1, 0), (-2, -1), (+1, +2)),
    (1, 0): ((0, 0), (+2, 0), (-1, 0), (+2, +1), (-1, -2)),
    (1, 2): ((0, 0), (-1, 0), (+2, 0), (-1, +2), (+2, -1)),
    (2, 1): ((0, 0), (+1, 0), (-2, 0), (+1, -2), (-2, +1)),
    (2, 3): ((0, 0), (+2, 0), (-1, 0), (+2, +1), (-1, -2)),
    (3, 2): ((0, 0), (-2, 0), (+1, 0), (-2, -1), (+1, +2)),
    (3, 0): ((0, 0), (+1, 0), (-2, 0), (+1, -2), (-2, +1)),
    (0, 3): ((0, 0), (-1, 0), (+2, 0), (-1, +2), (+2, -1)),
}


def test_jlstz_kick_table_matches_wiki():
    assert KICKS_JLSTZ == WIKI_JLSTZ


def test_i_kick_table_matches_wiki():
    assert KICKS_I == WIKI_I


def test_every_transition_has_five_tests():
    for table in (KICKS_JLSTZ, KICKS_I):
        assert len(table) == 8
        for tests in table.values():
            assert len(tests) == 5
            assert tests[0] == (0, 0)
