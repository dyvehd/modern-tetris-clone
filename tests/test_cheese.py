"""Cheese (dig) race: starting stack, hole runs, refill, goal cap, win.

Mechanics follow four-tris' cheese mode (top-up after placements, hole runs
of 1/1/2/2/4/5 rows) with Jstris' 9-row stack, and Techmino's dig_100l rule
of capping the refill at the lines still needed (verified against
Techmino's parts/eventsets/dig_*.lua, 2026-09-07).
"""

from conftest import hard_drop, make_game, place

from tetris.config import AppConfig, make_mode_config
from tetris.engine.constants import FIELD_H, FIELD_W, FULL_ROW, PieceType

RUN_POOL = (1, 1, 2, 2, 4, 5)  # four-tris' default GARBAGE=1,1,2,2,4,5


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


def test_holes_form_runs_that_move_column():
    game = make_game(cheese_rows=9, goal_lines=10)
    holes = [hole_of(row) for row in game.rows[-9:]]
    runs = []  # (column, length) per run
    for hole in holes:
        if runs and runs[-1][0] == hole:
            runs[-1] = (hole, runs[-1][1] + 1)
        else:
            runs.append((hole, 1))
    # every run length comes from the pool, and the hole changes column
    # between runs (never a second shaft right next to / on the old one)
    assert all(length in RUN_POOL for _, length in runs)
    assert len(runs) >= 2
    assert all(a != b for (a, _), (b, _) in zip(runs, runs[1:]))


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


def test_custom_hole_run_pool():
    game = make_game(cheese_rows=6, goal_lines=None, cheese_hole_runs=(6,))
    holes = [hole_of(row) for row in game.rows[-6:]]
    assert len(set(holes)) == 1  # a 6-row run: one straight shaft
    game.cheese_on_board = 5  # pretend one row was dug; refill draws a new run
    game._cheese_refill()
    assert hole_of(game.rows[-1]) != holes[0]  # the hole moves to a new column


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
