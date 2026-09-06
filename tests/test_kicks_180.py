"""180-degree rotation kicks (SRS+ table, as in Jstris/TETR.IO).

The table is universal (same kicks for every piece; the official diagram
illustrates with a T). Kick order matters and is pinned here:
    N->S  (0,0) (0,+1) (+1,+1) (-1,+1) (+1,0) (-1,0)
    S->N  (0,0) (0,-1) (-1,-1) (+1,-1) (-1,0) (+1,0)
    E->W  (0,0) (+1,0) (+1,+2) (+1,+1) (0,+2) (0,+1)
    W->E  (0,0) (-1,0) (-1,+2) (-1,+1) (0,+2) (0,+1)
Kicks are asserted in engine coordinates (+y = DOWN), converted from the
stored wiki convention (+y = UP).
"""

import pytest

from conftest import make_game, place

from tetris.engine import board as B
from tetris.engine.constants import PieceType
from tetris.engine.game import Action


@pytest.mark.parametrize(
    "piece",
    [
        PieceType.T,
        PieceType.J,
        PieceType.L,
        PieceType.S,
        PieceType.Z,
        PieceType.I,
    ],
)
def test_off_the_floor_up_kick(piece: PieceType):
    # Every piece flat on the floor: the 180 target orientation holds a cell
    # one row below the old body, so the pivot kick collides with the floor
    # and kick test 2 ("off-the-floor", wiki (0,+1) = UP 1) raises the piece.
    game = make_game()
    place(game, piece, x=3, y=38, rot=0)
    game.tick([Action.ROT_180])
    p = game.active
    assert (p.rot, p.x, p.y) == (2, 3, 37)
    assert game.last_kick_index == 1


def test_o_rotation_180_is_noop():
    game = make_game()
    place(game, PieceType.O, x=4, y=25, rot=0)
    game.tick([Action.ROT_180])
    p = game.active
    assert (p.rot, p.x, p.y) == (0, 4, 25)
    assert game.last_action is None


def test_east_west_tall_jump():
    # T pointing east: the in-place cell and the (+1,0) "off-the-wall" cell
    # are both blocked, so kick test 3 ("tall jump", wiki (+1,+2) = right 1,
    # UP 2) applies — horizontal 180s jump up, never down.
    game = make_game()
    game.rows[37] |= 1 << 6  # blocks the in-place rotation cell
    game.rows[38] |= 1 << 8  # blocks the (+1,0) off-the-wall cell
    place(game, PieceType.T, x=6, y=36, rot=1)
    game.tick([Action.ROT_180])
    p = game.active
    assert (p.rot, p.x, p.y) == (3, 7, 34)
    assert game.last_kick_index == 2


def test_west_east_tall_zipper_before_constricted():
    # T pointing west with every earlier kick blocked: the stack plugs the
    # column the -1 x-kicks would land in and the cells (0,+1) needs, so the
    # FIRST fitting kick is (0,+2) "tall zipper" — pins the kick order.
    game = make_game()
    game.rows[36] |= (1 << 7) | (1 << 9)
    game.rows[37] |= 1 << 9  # blocks the in-place rotation cell
    game.rows[38] |= 1 << 7
    place(game, PieceType.T, x=7, y=36, rot=3)
    game.tick([Action.ROT_180])
    p = game.active
    assert (p.rot, p.x, p.y) == (1, 7, 34)
    assert game.last_kick_index == 4


def test_west_east_unhook_left_at_right_wall():
    # The reported case: T against the right wall, in-place 180 blocked by a
    # stack cell in the column the rotated nub needs; kick test 2 (-1, 0)
    # slides the piece LEFT away from the wall.
    game = make_game()
    game.rows[38] |= 1 << 9
    place(game, PieceType.T, x=7, y=37, rot=3)
    game.tick([Action.ROT_180])
    p = game.active
    assert (p.rot, p.x, p.y) == (1, 6, 37)
    assert game.last_kick_index == 1


def test_south_north_off_the_ceiling():
    # T pointing down directly under a solid row: kick test 2 of 2=>0
    # ("off-the-ceiling", wiki (0,-1) = DOWN 1) drops it out of the ceiling.
    game = make_game()
    game.rows[36] = 0b1111111111
    place(game, PieceType.T, x=3, y=36, rot=2)
    game.tick([Action.ROT_180])
    p = game.active
    assert (p.rot, p.x, p.y) == (0, 3, 37)
    assert game.last_kick_index == 1


def test_tspin_via_180_counts():
    # 180 into a notch and hard drop: 3 corners filled (walls count), both
    # front corners filled -> full T-spin. Pins that a 180 keeps the last
    # movement "rotate" (T-spin eligibility) like any other rotation.
    game = make_game()
    game.rows[36] |= (1 << 3) | (1 << 5)
    game.rows[38] |= (1 << 3) | (1 << 5)
    place(game, PieceType.T, x=3, y=36, rot=2)
    game.tick([Action.ROT_180])  # 2->0 in place: only (4,36) was free
    p = game.active
    assert p.rot == 0
    game.tick([Action.HARD_DROP])
    assert game.tspins == 1
    assert game.score == 400  # full T-spin, no lines


def test_180_fails_when_no_kick_fits():
    # T flat on the floor with the cells beside its nub plugged: every kick
    # collides (the in-place and x-kick targets fall below the floor, the
    # up-kick targets hit the stack), so the rotation is canceled entirely.
    game = make_game()
    game.rows[38] |= (1 << 3) | (1 << 5)
    place(game, PieceType.T, x=3, y=38, rot=0)
    game.tick([Action.ROT_180])
    p = game.active
    assert (p.rot, p.x, p.y) == (0, 3, 38)
    assert game.last_action is None
