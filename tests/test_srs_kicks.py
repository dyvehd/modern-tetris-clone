"""Functional SRS kick scenarios — these pin down the (dx, dy) sign
convention (engine positions use +y = DOWN; wiki tables use +y = UP)."""

from conftest import hard_drop, make_game, place

from tetris.engine import board as B
from tetris.engine.constants import PieceType


def test_t_floor_kick_moves_piece_up():
    # T pointing up on the floor, rotating CCW: basic rotation and test 2
    # collide with the floor; test 3 (+1, +1 wiki = right 1, UP 1) succeeds.
    game = make_game()
    place(game, PieceType.T, x=4, y=38, rot=0)
    game.tick([])
    game.tick([game_action_ccw()])
    p = game.active
    assert (p.rot, p.x, p.y) == (3, 5, 37)  # moved right 1, UP 1


def game_action_ccw():
    from tetris.engine.game import Action

    return Action.ROT_CCW


def test_tst_kick_left_down():
    # Only the 5th SRS test (0=>R: -1,-2 = left 1, DOWN 2) fits; every other
    # test is blocked by the stack. Also exercises the TST T-spin upgrade.
    game = make_game()
    game.set_rows(
        B.from_ascii(
            "\n".join(
                [
                    "          ",  # 20
                    "          ",
                    "          ",
                    "          ",
                    "          ",
                    "          ",
                    "          ",
                    "          ",
                    "          ",
                    "          ",
                    "          ",
                    "          ",
                    "          ",
                    "          ",
                    "....#.....",  # 34
                    "....#.....",  # 35
                    "          ",  # 36
                    "...#.#....",  # 37
                    "          ",  # 38
                    "...#......",  # 39
                ]
            )
        )
    )
    place(game, PieceType.T, x=4, y=35, rot=0)
    from tetris.engine.game import Action

    game.tick([Action.ROT_CW])
    p = game.active
    assert (p.rot, p.x, p.y) == (1, 3, 37)  # left 1, down 2
    assert game.last_kick_index == 4

    # Lock in place (0 travel): 3 corners filled but only one front corner,
    # so without the TST-kick upgrade this would be a mini — the kick
    # upgrades it to a full T-spin.
    hard_drop(game)
    assert game.tspins == 1
    assert game.score == 400  # full T-spin, 0 lines
    assert any(ev["kind"] == "tspin" for ev in game.events)


def test_i_wall_kick_right():
    # I vertical (rot R), rotating back to horizontal with (4,37) blocked:
    # test 2 of R=>0 (+2, 0) moves it RIGHT 2.
    game = make_game()
    game.set_rows(
        B.from_ascii(
            "\n".join(
                ["          "] * 17
                + [
                    "          ",  # 37: (4,37) filled below
                    "          ",
                    "          ",
                ]
            )
        )
    )
    game.rows[37] |= 1 << 4
    place(game, PieceType.I, x=4, y=36, rot=1)
    from tetris.engine.game import Action

    game.tick([Action.ROT_CCW])
    p = game.active
    assert (p.rot, p.x, p.y) == (0, 6, 36)  # right 2, no vertical shift
    assert game.last_kick_index == 1


def test_no_kick_when_basic_rotation_fits():
    from tetris.engine.game import Action

    game = make_game()
    place(game, PieceType.T, x=3, y=25, rot=0)
    game.tick([Action.ROT_CW])
    p = game.active
    assert (p.rot, p.x, p.y) == (1, 3, 25)
    assert game.last_kick_index == 0


def test_o_rotation_is_noop():
    from tetris.engine.game import Action

    game = make_game()
    place(game, PieceType.O, x=4, y=25, rot=0)
    game.tick([Action.ROT_CW])
    p = game.active
    assert (p.rot, p.x, p.y) == (0, 4, 25)


def test_blocked_rotation_fails():
    from tetris.engine.game import Action

    game = make_game()
    # Solid rows above and below leave the T nowhere to kick (tests 1-5 all
    # collide), so the rotation must fail entirely.
    game.rows[23] = 0b1111111111
    game.rows[26] = 0b1111111111
    game.rows[27] = 0b1111111111
    place(game, PieceType.T, x=3, y=24, rot=0)
    game.tick([Action.ROT_CW])
    p = game.active
    assert (p.rot, p.x, p.y) == (0, 3, 24)
    assert game.last_action is None


def test_i_on_floor_rotates_via_up_kick():
    # I horizontal on the floor rotating CW: every test that keeps it at the
    # same height or lower collides with the floor, so SRS applies the 5th
    # test (+1, +2) — the I pops UP and over one column. This is standard.
    from tetris.engine.game import Action

    game = make_game()
    place(game, PieceType.I, x=3, y=38, rot=0)
    game.tick([Action.ROT_CW])
    p = game.active
    assert (p.rot, p.x, p.y) == (1, 4, 36)  # right 1, UP 2
    assert game.last_kick_index == 4
