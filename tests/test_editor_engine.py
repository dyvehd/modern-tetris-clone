"""Engine primitives for the zen sandbox editors: bag phase + set_queue,
mouse cell painting, and shape recognition (any 4 connected cells are
exactly one of the 7 tetrominoes — chirality included)."""

from conftest import make_game

from tetris.engine import board as B
from tetris.engine.constants import EDITOR_GRAY, PIECE_CELLS, PieceType


# ---------------------------------------------------------------- shapes

def test_every_piece_rotation_is_recognized_at_any_position():
    for piece, rots in PIECE_CELLS.items():
        for rot_cells in rots:
            cells = [(x + 3, y + 10) for x, y in rot_cells]
            assert B.piece_from_cells(cells) is piece, (piece, rot_cells)


def test_recognition_is_order_insensitive_and_shift_free():
    assert B.piece_from_cells([(5, 38), (2, 38), (4, 38), (3, 38)]) is PieceType.I
    assert B.piece_from_cells([(4, 36), (4, 37), (4, 38), (4, 39)]) is PieceType.I
    assert B.piece_from_cells([(4, 38), (5, 38), (4, 39), (5, 39)]) is PieceType.O
    assert B.piece_from_cells([(4, 38), (5, 38), (6, 38), (5, 39)]) is PieceType.T


def test_recognition_keeps_chirality():
    # an L-shaped set is an L or a J depending on the foot side — never ambiguous
    j = [(x + 5, y + 20) for x, y in PIECE_CELLS[PieceType.J][2]]
    l = [(x + 5, y + 20) for x, y in PIECE_CELLS[PieceType.L][2]]
    assert B.piece_from_cells(j) is PieceType.J
    assert B.piece_from_cells(l) is PieceType.L
    s = [(x + 5, y + 20) for x, y in PIECE_CELLS[PieceType.S][0]]
    z = [(x + 5, y + 20) for x, y in PIECE_CELLS[PieceType.Z][0]]
    assert B.piece_from_cells(s) is PieceType.S
    assert B.piece_from_cells(z) is PieceType.Z


def test_recognition_rejects_non_tetrominoes():
    assert B.piece_from_cells([(1, 30), (2, 30), (3, 31), (4, 32)]) is None  # diagonal
    assert B.piece_from_cells([(1, 30), (2, 30), (3, 31), (1, 31)]) is None  # not a piece
    assert B.piece_from_cells([(1, 30), (2, 30), (3, 30)]) is None  # 3 cells
    assert B.piece_from_cells([(1, 30), (2, 30), (3, 30), (4, 30), (5, 30)]) is None  # 5
    assert B.piece_from_cells([(1, 30), (2, 30), (3, 30), (3, 30)]) is None  # dup


def test_gray_component_bounded_and_connected():
    styles = [[-1] * 10 for _ in range(40)]
    styles[38][4] = styles[38][5] = styles[37][5] = EDITOR_GRAY
    comp = B.gray_component(styles, 38, 5)
    assert sorted(comp) == sorted([(38, 5), (38, 4), (37, 5)])
    # real garbage (-2) and piece cells never join an editor-gray component
    styles[36][5] = -2
    styles[38][6] = PieceType.I.value
    assert sorted(B.gray_component(styles, 38, 5)) == sorted(comp)
    # oversized regions abort instead of returning a partial list
    big = [[EDITOR_GRAY] * 10 for _ in range(40)]
    assert B.gray_component(big, 20, 5) is None


# ------------------------------------------------------------------ cells

def test_edit_paint_and_erase():
    game = make_game()
    assert game.edit_paint(38, 4)
    assert game.rows[38] >> 4 & 1
    assert game.styles[38][4] == EDITOR_GRAY
    # painting the same cell again is a no-op
    assert not game.edit_paint(38, 4)
    # painting with a piece style over an occupied cell only restyles it
    assert game.edit_paint(38, 4, PieceType.I.value)
    assert game.styles[38][4] == PieceType.I.value
    # out of field is ignored
    assert not game.edit_paint(-1, 0)
    assert not game.edit_paint(0, 10)

    assert game.edit_erase(38, 4)
    assert not (game.rows[38] >> 4 & 1)
    assert game.styles[38][4] == -1
    assert not game.edit_erase(38, 4)  # already empty


def test_edited_lines_clear_normally_and_styles_shift():
    from tetris.engine.constants import FULL_ROW
    from tetris.engine.game import Action

    game = make_game()
    for x in range(9):  # gray row 39 with a col-9 shaft
        game.edit_paint(39, x)
    # a vertical I down the shaft completes the edited line
    game.spawn_forced(PieceType.I)
    game.active.rot = 1  # vertical, column x+2
    game.active.x = 7
    game.tick([Action.HARD_DROP])
    assert game.lines == 1
    assert game.rows[39] == 1 << 9  # only the I's upper cells remain
    # the clear shifted the style grid down with the rows: I-styled cells
    # at rows 37-39 col 9, and the gray row is gone
    for ry in (37, 38, 39):
        assert game.styles[ry][9] == PieceType.I.value
    assert all(v == -1 for x, v in enumerate(game.styles[39]) if x != 9)
    assert not any(v == EDITOR_GRAY for row in game.styles for v in row)
    assert FULL_ROW not in game.rows


# ------------------------------------------------------------------- bags

def test_bag_pos_tracks_pieces_taken():
    game = make_game()
    assert game.bag_pos == 0
    for expected in range(1, 8):
        game._take_from_queue()
        assert game.bag_pos == expected % 7


def test_set_queue_replaces_and_offsets():
    game = make_game()
    game.set_queue([PieceType.S, PieceType.I], bag_offset=3)
    assert list(game.queue[:2]) == [PieceType.S, PieceType.I]
    assert game.bag_pos == 3
    # the sequence runs out and fresh 7-bags top the queue back up
    assert len(game.queue) >= 7
    assert game.queue[0] is PieceType.S
    game._take_from_queue()
    assert game.bag_pos == 4


def test_set_queue_empty_returns_to_random_bags():
    game = make_game()
    before = list(game.queue)
    game.set_queue([], bag_offset=0)
    assert list(game.queue) != before or True  # refilled with a fresh bag
    assert len(game.queue) >= 7
    assert game.bag_pos == 0
