"""Lock delay, move reset (15), step-down counter reset, hard drop."""

from conftest import make_game, place

from tetris.engine.constants import PieceType
from tetris.engine.game import Action, GameConfig


def test_lock_after_500ms_on_floor():
    game = make_game(gravity_g=1.0, lock_delay_ms=500)  # 30 ticks at 60 Hz
    place(game, PieceType.T, x=3, y=38, rot=0)  # resting on the floor
    for _ in range(29):
        game.tick([])
    assert game.active is not None
    assert game.lock_timer == 29
    game.tick([])  # 30th grounded tick -> lock
    assert game.pieces_placed == 1


def test_move_resets_lock_timer():
    game = make_game(gravity_g=1.0, lock_delay_ms=500)
    place(game, PieceType.T, x=3, y=38, rot=0)
    game.tick([])  # grounded, timer = 1
    game.tick([Action.LEFT])  # successful move: timer back to 0 (+1 grounded)
    assert game.active.x == 2
    assert game.move_resets == 1
    assert game.lock_timer == 1  # restarted, then one grounded tick
    # 28 more idle ticks should NOT lock (30-tick window restarted)
    for _ in range(28):
        game.tick([])
    assert game.pieces_placed == 0
    game.tick([])
    assert game.pieces_placed == 1


def test_16th_move_does_not_reset():
    game = make_game(gravity_g=1.0, lock_delay_ms=500)
    place(game, PieceType.T, x=3, y=38, rot=0)
    game.tick([])  # grounded
    # 15 alternating successful moves, each on its own tick
    for i in range(15):
        game.tick([Action.LEFT if i % 2 == 0 else Action.RIGHT])
        assert game.active is not None
    assert game.move_resets == 15
    ticks_after_15th = game.tick_count
    game.tick([Action.RIGHT])  # 16th move succeeds but must not reset
    assert game.active.x == 3
    # The window ended with the 15th move: the piece locks 29 ticks later.
    for _ in range(27):
        game.tick([])
    assert game.pieces_placed == 0
    game.tick([])
    assert game.pieces_placed == 1
    assert game.tick_count - ticks_after_15th == 29


def test_new_lowest_row_resets_counter():
    game = make_game(gravity_g=1.0, lock_delay_ms=500)
    # platform at cols 0-4, rows 38-39
    for row in (38, 39):
        game.rows[row] = 0b0000011111
    place(game, PieceType.T, x=0, y=36, rot=0)  # resting on the platform
    game.tick([])  # grounded
    for i in range(15):
        game.tick([Action.RIGHT if i % 2 == 0 else Action.LEFT])
    assert game.move_resets == 15
    # slide off the platform edge (x=5 is clear of the cols 0-4 platform)
    for _ in range(4):
        game.tick([Action.RIGHT])  # x: 1 -> 5; grounded moves consume nothing
    for _ in range(6):
        game.tick([])  # gravity drops the T to the floor
    assert game.move_resets == 0  # new lowest row reset the counter
    assert game.active.y == 38
    # and the budget is available again
    game.tick([Action.RIGHT])
    assert game.move_resets == 1


def test_hard_drop_locks_instantly():
    game = make_game(gravity_g=1.0, lock_delay_ms=500)
    place(game, PieceType.T, x=3, y=20, rot=0)
    game.tick([Action.HARD_DROP])
    assert game.pieces_placed == 1
    assert game.score == 2 * 18  # 18 rows fallen at 2 points each


def test_airborne_moves_do_not_consume_resets():
    game = make_game(gravity_g=0.0, lock_delay_ms=500)
    place(game, PieceType.T, x=3, y=25, rot=0)  # floating in mid-air
    for _ in range(20):
        game.tick([Action.RIGHT, Action.LEFT])
    assert game.move_resets == 0
    assert game.active is not None
