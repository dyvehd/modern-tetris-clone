"""Cheese (dig) race: starting stack, messiness, refill trigger, goal cap.

Mechanics: Jstris-style dirty cheese (adjacent holes never align) with a
per-row hole-change chance ("messiness", TETR.IO's term), topped back up
only when a placement clears nothing (Jstris) or after every placement
(TETR.IO). Cross-checked against four-tris' cheese code and Techmino's
parts/eventsets/dig_*.lua + getHolePos, 2026-09-07.
"""

from conftest import hard_drop, make_game, place

from tetris.config import AppConfig, make_mode_config
from tetris.engine.constants import FIELD_H, FIELD_W, FULL_ROW, PieceType


def hole_of(row: int) -> int:
    holes = [x for x in range(FIELD_W) if not (row >> x) & 1]
    assert len(holes) == 1
    return holes[0]


def test_cheese_starts_with_a_full_stack():
    game = make_game(cheese_rows=9, goal_lines=10)
    assert game.cheese_on_board == 9
    assert all(row == 0 for row in game.rows[:-9])
    for row in game.rows[-9:]:
        assert bin(row).count("1") == FIELD_W - 1  # one hole per row
    assert all(v == -2 for line in game.styles[-9:] for v in line)


def test_non_cheese_game_has_no_cheese():
    game = make_game()
    assert game.cfg.cheese_rows == 0
    assert game.cheese_on_board == 0
    assert all(row == 0 for row in game.rows)


def test_messiness_100_holes_never_align_adjacent_rows():
    game = make_game(cheese_rows=9, goal_lines=None)
    holes = [hole_of(row) for row in game.rows[-9:]]
    holes += [game._next_cheese_hole() for _ in range(40)]
    assert all(a != b for a, b in zip(holes, holes[1:]))


def test_messiness_0_pins_a_single_column():
    game = make_game(cheese_rows=9, goal_lines=None, cheese_messiness=0.0)
    holes = [hole_of(row) for row in game.rows[-9:]]
    holes += [game._next_cheese_hole() for _ in range(40)]
    assert len(set(holes)) == 1


def test_messiness_is_a_per_row_column_change_chance():
    game = make_game(cheese_rows=9, goal_lines=None, cheese_messiness=50.0)
    holes = [game._next_cheese_hole() for _ in range(300)]
    changes = sum(a != b for a, b in zip(holes, holes[1:]))
    # 50% chance to move per row: ~150 of 299 transitions (seed-pinned)
    assert 80 < changes < 220


def test_clearing_a_line_refills_the_stack():
    game = make_game(cheese_rows=9, goal_lines=10)
    hole = hole_of(game.rows[-1])
    place(game, PieceType.I, x=hole - 2, y=36, rot=1)  # vertical I down the shaft
    hard_drop(game)
    # the I completes the bottom row — plus any run rows sharing its hole
    # (clearing a whole shaft in one drop is the point of hole runs)
    assert 1 <= game.lines <= 4
    # dug rows are refilled back up to the goal-capped target
    assert game.cheese_on_board == min(9, 10 - game.lines)
    assert bin(game.rows[39]).count("1") == FIELD_W - 1  # bottom row is cheese


def test_no_clear_means_no_refill():
    game = make_game(cheese_rows=9, goal_lines=10)
    place(game, PieceType.O, x=0, y=0)  # ghost-drops from the top
    hard_drop(game)
    assert game.lines == 0
    assert game.cheese_on_board == 9  # unchanged: nothing was dug
    # the O rests directly on top of the 9-row cheese stack (rows 31-39)
    assert game.rows[29] == 0b0000000011
    assert game.rows[30] == 0b0000000011


def test_refill_capped_at_lines_still_needed():
    game = make_game(cheese_rows=9, goal_lines=10)
    game.rows[:] = [0] * FIELD_H
    game.lines = 8
    game.cheese_on_board = 0
    game._cheese_refill()
    assert game.cheese_on_board == 2  # min(9, 10 - 8)
    assert all(row == 0 for row in game.rows[:-2])
    assert bin(game.rows[39]).count("1") == FIELD_W - 1


def test_reaching_the_goal_wins():
    game = make_game(cheese_rows=9, goal_lines=10)
    game.rows[:] = [0] * (FIELD_H - 1) + [FULL_ROW & ~(1 << 3)]
    game.styles[:] = [[-1] * FIELD_W for _ in range(FIELD_H)]
    game.styles[-1] = [-2] * FIELD_W
    game.cheese_on_board = 1
    game.lines = 9
    place(game, PieceType.I, x=1, y=36, rot=1)  # fills the col-3 shaft
    hard_drop(game)
    assert game.lines == 10
    assert game.over and game.won
    # the last clear was counted as dug, and the refill target had hit 0,
    # so no fresh cheese rose after it
    assert game.cheese_on_board == 0


def test_infinite_cheese_refills_forever():
    game = make_game(cheese_rows=9, goal_lines=None)
    game.cheese_on_board = 2
    game._cheese_refill()
    assert game.cheese_on_board == 9
    assert not game.over


# --- mode presets / config ---------------------------------------------------

def test_cheese_mode_presets():
    base = AppConfig()
    rules, trainer = make_mode_config("Cheese 10", base)
    assert rules.cheese_rows == 9 and rules.goal_lines == 10 and not trainer
    rules, _ = make_mode_config("Cheese ∞", base)
    assert rules.cheese_rows == 9 and rules.goal_lines is None


def test_user_cheese_rows_override_and_no_leak():
    base = AppConfig()
    base.rules.cheese_rows = 10  # four-tris/Techmino stack size
    rules, _ = make_mode_config("Cheese 10", base)
    assert rules.cheese_rows == 10
    # ...and it must not turn non-cheese modes into cheese
    for mode in ("Marathon", "Sprint 40 Lines", "Zen", "VS Sandbox"):
        rules, _ = make_mode_config(mode, base)
        assert rules.cheese_rows == 0, mode
