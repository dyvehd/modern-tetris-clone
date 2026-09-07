"""Zen undo (TETR.IO-style Ctrl+Z): one snapshot per spawned piece, undo
steps back to the spawn of the piece you are controlling — the last placed
piece returns to your hands, the board/queue/score revert with it."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pytest  # noqa: E402

pygame = pytest.importorskip("pygame")

from tetris.app import App  # noqa: E402
from tetris.engine.game import Btn  # noqa: E402


def _app_for(mode: str):
    from tetris.app import App
    from tetris.config import AppConfig, DebugConfig

    app = App(AppConfig(debug=DebugConfig(log_input=False)))
    app.mode_idx = app.modes.index(mode)
    app.start_game()
    app.logic_tick()  # first spawn (snapshot of the empty board)
    return app


def _hard_drop(app) -> None:
    app.controller.press(Btn.HARD)
    app.logic_tick()
    app.controller.release(Btn.HARD)
    app.logic_tick()  # spawn tick: the undo snapshot for the next piece


def _undo_key(app) -> None:
    # KEYDOWN carries the modifier state; fake a Ctrl+Z event
    event = type("FakeKeydown", (), {"mod": pygame.KMOD_CTRL})()
    app.handle_key(pygame.K_z, event)


def test_undo_restores_board_piece_and_stats():
    app = _app_for("Zen")
    assert app.undo_enabled
    _hard_drop(app)
    after_one = (list(app.game.rows), app.game.pieces_placed, app.game.score)
    placed_type = app.game.active.type
    _hard_drop(app)
    assert app.game.pieces_placed == 2
    assert app.game.rows != after_one[0]

    _undo_key(app)
    assert app.game.pieces_placed == 1
    assert app.game.score == after_one[2]
    assert list(app.game.rows) == after_one[0]
    # the dropped piece is back in your hands, back at spawn state
    assert app.game.active is not None
    assert app.game.active.type is placed_type
    assert app.game.active.rot == 0


def test_undo_walks_back_to_the_empty_board():
    app = _app_for("Zen")
    for _ in range(3):
        _hard_drop(app)
    assert app.game.pieces_placed == 3
    for placed in (2, 1, 0):
        _undo_key(app)
        assert app.game.pieces_placed == placed
    assert all(row == 0 for row in app.game.rows)
    # empty history: further undo presses are a no-op, game stays playable
    _undo_key(app)
    assert app.game.pieces_placed == 0
    assert app.game.over is False


def test_undo_after_hold_restores_hold_state():
    app = _app_for("Zen")
    held_type = app.game.active.type
    app.controller.press(Btn.HOLD)
    app.logic_tick()
    app.controller.release(Btn.HOLD)
    app.logic_tick()
    assert app.game.hold_type is held_type
    _hard_drop(app)
    _undo_key(app)
    # hold state reverts with the snapshot
    assert app.game.hold_type is held_type
    assert app.game.can_hold is False  # the hold had already been used


def test_undone_piece_returns_to_hands_not_queue():
    app = _app_for("Zen")
    _hard_drop(app)
    in_hand = app.game.active.type
    queue_before = list(app.game.queue)
    rows_before = list(app.game.rows)
    _hard_drop(app)
    _undo_key(app)
    # board is back, the dropped piece is the active one again, and the
    # queue is exactly as it was before the placement
    assert list(app.game.rows) == rows_before
    assert app.game.active.type is in_hand
    assert list(app.game.queue) == queue_before


def test_undo_disabled_outside_zen_modes():
    app = _app_for("Sprint 40 Lines")
    assert not app.undo_enabled
    _hard_drop(app)
    placed = app.game.pieces_placed
    _undo_key(app)
    assert app.game.pieces_placed == placed
    assert app._undo_stack == []


def test_undo_reaches_into_the_previous_game_after_restart():
    app = _app_for("Zen")
    _hard_drop(app)
    after_one = list(app.game.rows)
    _hard_drop(app)
    app.restart()
    assert app.game.pieces_placed == 0
    _undo_key(app)
    # Ctrl+Z after R revives the game you just left: its board after the
    # first placement, with that game's second piece back in your hands
    assert app.game.pieces_placed == 1
    assert list(app.game.rows) == after_one
    # the revived game keeps playing and taking new snapshots
    _hard_drop(app)
    assert app.game.pieces_placed == 2
    _undo_key(app)
    assert app.game.pieces_placed == 1
    assert list(app.game.rows) == after_one


def test_undo_after_top_out_from_the_over_screen():
    app = _app_for("Zen")
    _hard_drop(app)
    after_one = list(app.game.rows)
    _hard_drop(app)
    app.game._game_over(won=False)
    app.state = App.STATE_OVER
    app.render()  # the over screen (with the undo hint) draws fine
    _undo_key(app)
    assert app.state == App.STATE_PLAY
    assert app.game.over is False
    assert app.game.pieces_placed == 1
    assert list(app.game.rows) == after_one


def test_history_does_not_leak_into_non_undo_modes():
    app = _app_for("Zen")
    _hard_drop(app)
    assert app._undo_stack
    app.mode_idx = app.modes.index("Sprint 40 Lines")
    app.start_game()
    assert app._undo_stack == []
    _undo_key(app)  # a no-op: sprint has no undo
    assert app.game.pieces_placed == 0
