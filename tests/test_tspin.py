"""T-spin detection: 3-corner rule, front-corner mini rule, TST upgrade,
last-movement rule, wall counting. Sources:
https://harddrop.com/wiki/T-Spin (Tetris-Friends ruleset, used by Jstris)."""

from conftest import hard_drop, make_game, place

from tetris.engine.constants import PieceType
from tetris.engine.game import Action


def _prep(game, rows, rot, x, y):
    game.set_rows(rows)
    place(game, PieceType.T, x=x, y=y, rot=rot)


# --- unit tests of the detector ---------------------------------------------


def _detect(game, rot, from_rot, to_rot, kick_index, last_action="rotate"):
    game.last_action = last_action
    game.last_rot_from = from_rot
    game.last_rot_to = to_rot
    game.last_kick_index = kick_index
    return game._detect_tspin(game.active)


def test_full_when_both_front_corners_filled():
    game = make_game()
    # T rot 0 at (4, 36); fill all four box corners except one front is
    # enough for full when both front corners are set.
    rows = [0] * 40
    rows[36] |= (1 << 4) | (1 << 6)  # top corners (front for rot 0)
    rows[38] |= (1 << 4) | (1 << 6)  # bottom corners
    _prep(game, rows, rot=0, x=4, y=36)
    assert _detect(game, 0, 0, 1, 0) == "full"


def test_mini_when_only_one_front_corner_filled():
    game = make_game()
    rows = [0] * 40
    rows[36] |= 1 << 4  # front-left filled, front-right open
    rows[38] |= (1 << 4) | (1 << 6)  # both back corners
    _prep(game, rows, rot=0, x=4, y=36)
    assert _detect(game, 0, 0, 1, 0) == "mini"


def test_two_corners_is_not_a_spin():
    game = make_game()
    rows = [0] * 40
    rows[38] |= (1 << 4) | (1 << 6)  # back corners only
    _prep(game, rows, rot=0, x=4, y=36)
    assert _detect(game, 0, 0, 1, 0) == "none"


def test_out_of_bounds_counts_as_filled_corner():
    game = make_game()
    rows = [0] * 40
    # T rot 3 (pointing left) at x=0: box corners at x=0 are "wall side";
    # fill the two right corners and use the floor for one more.
    rows[37] |= 1 << 2
    rows[39] |= (1 << 0) | (1 << 2)
    _prep(game, rows, rot=3, x=0, y=37)
    # corners: (0,37) empty, (2,37) filled, (0,39) filled, (2,39) filled
    assert _detect(game, 3, 0, 3, 0) == "mini"


def test_tst_kick_upgrades_mini_to_full():
    game = make_game()
    rows = [0] * 40
    rows[36] |= 1 << 4  # one front corner
    rows[38] |= (1 << 4) | (1 << 6)  # back corners
    _prep(game, rows, rot=0, x=4, y=36)
    assert _detect(game, 0, 0, 1, 4) == "full"  # kick test 5 on 0=>R


def test_tst_kick_only_on_0r_and_2l():
    game = make_game()
    rows = [0] * 40
    rows[36] |= 1 << 4
    rows[38] |= (1 << 4) | (1 << 6)
    _prep(game, rows, rot=0, x=4, y=36)
    # Same corners, but the 5th test on 0=>L is not a TST kick: still mini.
    assert _detect(game, 0, 0, 3, 4) == "mini"


def test_last_action_must_be_rotation():
    game = make_game()
    rows = [0] * 40
    rows[36] |= (1 << 4) | (1 << 6)
    rows[38] |= (1 << 4) | (1 << 6)
    _prep(game, rows, rot=0, x=4, y=36)
    assert _detect(game, 0, 0, 1, 0, last_action="shift") == "none"
    assert _detect(game, 0, 0, 1, 0, last_action="fall") == "none"
    assert _detect(game, 0, 0, 1, 0, last_action=None) == "none"


def test_non_t_pieces_never_spin():
    game = make_game()
    rows = [0] * 40
    rows[36] |= 0b1111111111
    place(game, PieceType.J, x=4, y=37, rot=0)
    assert _detect(game, 0, 0, 1, 0) == "none"


# --- integration: real rotations into real slots ----------------------------


def test_wall_mini_tspin_single():
    # Classic left-wall mini TSS: T rotated CCW against the wall with a
    # floor row beneath. Board (rows 37-39):
    #   37: ..#......        (back corner stack)
    #   39: #.#######        (row clears after the T fills col 1)
    game = make_game()
    rows = [0] * 40
    rows[37] |= 1 << 2
    rows[39] |= (1 << 0) | 0b1111111100  # cols 0, 2..9; col 1 open for the T
    _prep(game, rows, rot=0, x=0, y=37)  # T pointing up beside the wall
    game.tick([Action.ROT_CCW])  # -> rot L at (0,37), kick test 1
    p = game.active
    assert (p.rot, p.x, p.y) == (3, 0, 37)
    # A blocked shift must NOT cancel the spin (no movement happened).
    game.tick([Action.RIGHT])
    hard_drop(game)  # 0 travel: rotation remains the last movement
    assert game.lines == 1
    assert game.tspins == 1
    assert game.score == 200  # mini T-spin single
    assert game.attack_sent == 0
    assert game.b2b_chain == 1  # mini single keeps the chain
    assert game.combo == 1


def test_wall_full_tspin_single():
    # Same shape with both front corners filled -> full TSS (800, attack 2).
    game = make_game()
    rows = [0] * 40
    rows[37] |= (1 << 0) | (1 << 2)
    rows[39] |= (1 << 0) | 0b1111111100
    _prep(game, rows, rot=0, x=0, y=37)
    game.tick([Action.ROT_CCW])
    hard_drop(game)
    assert game.lines == 1
    assert game.tspins == 1
    assert game.score == 800
    assert game.attack_sent == 2
    assert game.b2b_chain == 1


def test_successful_shift_after_rotation_cancels_spin():
    # Unit-level guarantee (see test_last_action_must_be_rotation) plus a
    # sanity check that a *failed* shift (no movement) keeps the spin.
    game = make_game()
    rows = [0] * 40
    rows[37] |= (1 << 0) | (1 << 2)
    rows[39] |= (1 << 0) | 0b1111111100
    _prep(game, rows, rot=0, x=0, y=37)
    game.tick([Action.ROT_CCW])
    game.tick([Action.RIGHT])  # wedged: shift fails, nothing moves
    assert game.last_action == "rotate"
    hard_drop(game)
    assert game.tspins == 1
    assert game.lines == 1
