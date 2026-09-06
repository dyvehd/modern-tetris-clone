"""Garbage: delay, rise, cancel, per-spawn cap, overflow top-out."""

from conftest import hard_drop, make_game, place

from tetris.engine.constants import FIELD_H, FULL_ROW, PieceType


def test_garbage_rises_on_next_spawn():
    game = make_game(gravity_g=0.0, garbage_delay_ms=0)
    game.spawn_forced(PieceType.T)
    game.add_garbage(4)
    assert sum(len(b.rows) for b in game.garbage_queue) == 4
    # not applied while the current piece is alive
    game.tick([])
    assert all(row == 0 for row in game.rows)
    hard_drop(game)  # T locks on the floor; next spawn pushes garbage in
    bottom = game.rows[-4:]
    assert all(row != 0 for row in bottom)
    assert all(row != FULL_ROW for row in bottom)  # each row has a hole
    # the locked T was pushed up by 4 rows (bottom row 39 -> 35)
    assert game.rows[34] == 0b00010000  # T top cell (col 4)
    assert game.rows[35] == 0b00111000  # T bar (cols 3-5)
    assert sum(len(b.rows) for b in game.garbage_queue) == 0


def test_garbage_delay_respected():
    game = make_game(gravity_g=0.0, garbage_delay_ms=500)  # 30 ticks
    game.spawn_forced(PieceType.T)
    game.add_garbage(1)
    hard_drop(game)  # lock + spawn at tick ~1: not due yet
    assert game.rows[39] == 0b00111000  # the locked T's bar, still on floor
    for _ in range(40):
        game.tick([])  # delay elapses while the next piece is alive
    pre = list(game.rows)
    hard_drop(game)  # piece 2 locks; the next spawn delivers the garbage
    # stack shifted up one row, garbage entered at the bottom
    assert game.rows[38] == pre[39]
    assert game.rows[39] not in (0, FULL_ROW)


def test_attack_cancels_pending_garbage():
    game = make_game(gravity_g=0.0)
    game.spawn_forced(PieceType.T)
    game.add_garbage(3)
    # double-clear setup: rows 38-39 open at cols 4-5; junk in the buffer
    # so the clear is NOT a perfect clear
    game.rows[10] |= 1
    game.rows[38] = FULL_ROW & ~((1 << 4) | (1 << 5))
    game.rows[39] = FULL_ROW & ~((1 << 4) | (1 << 5))
    place(game, PieceType.O, x=4, y=37)
    hard_drop(game)  # O fills cols 4-5 -> double -> attack 1
    assert game.attack_sent == 0  # cancelled 1 of the 3 pending rows
    assert sum(len(b.rows) for b in game.garbage_queue) == 2


def test_garbage_cap_per_rise():
    game = make_game(gravity_g=0.0, garbage_delay_ms=0, garbage_cap_per_rise=2)
    game.spawn_forced(PieceType.T)
    game.add_garbage(5)
    hard_drop(game)
    # the locked T was pushed up 2 rows; exactly 2 garbage rows entered
    assert game.rows[36] == 0b00010000  # T top cell
    assert game.rows[37] == 0b00111000  # T bar
    assert all(row not in (0, FULL_ROW) for row in game.rows[-2:])
    assert sum(len(b.rows) for b in game.garbage_queue) == 3


def test_overflow_tops_out():
    game = make_game(gravity_g=0.0, garbage_delay_ms=0)
    game.rows[:] = [FULL_ROW] * (FIELD_H - 2) + [0, 0]
    game.add_garbage(2)
    game.spawn_forced(PieceType.Z)  # spawn applies due garbage -> overflow
    assert game.over
