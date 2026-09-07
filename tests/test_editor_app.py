"""Zen sandbox mouse editor (four-tris inspired): paint/erase with drag,
4-cell strokes auto-color as their tetromino, edits are Ctrl+Z undoable,
and clicking the next box opens the queue editor dialog."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pytest  # noqa: E402

pygame = pytest.importorskip("pygame")

from tetris.app import App  # noqa: E402
from tetris.engine import board as B  # noqa: E402
from tetris.engine.constants import EDITOR_GRAY, PieceType  # noqa: E402
from tetris.engine.game import Btn  # noqa: E402

from test_undo import _undo_key  # noqa: E402


def _app_for(mode: str = "Zen"):
    from tetris.config import AppConfig, DebugConfig

    app = App(AppConfig(debug=DebugConfig(log_input=False)))
    app.mode_idx = app.modes.index(mode)
    app.start_game()
    app.logic_tick()  # first spawn
    return app


def _cell_pos(app, ry: int, x: int):
    """Center of a field cell in screen coordinates."""
    r = app.renderer
    return (r.field_x + x * r.cell + r.cell // 2,
            r.field_y + (ry - 20) * r.cell + r.cell // 2)


def _click(app, ry, x, button=1, shift=False):
    app.handle_mouse_buttondown(_cell_pos(app, ry, x), button, shift)
    app.handle_mouse_buttonup(button)


def _drag(app, cells, button=1, shift=False):
    """A drag through the given (ry, x) cells, in order."""
    app.handle_mouse_buttondown(_cell_pos(app, *cells[0]), button, shift)
    buttons = (1, 0, 0) if button == 1 else (0, 0, 1)
    for cell in cells[1:]:
        app.handle_mouse_motion(_cell_pos(app, *cell), buttons)
    app.handle_mouse_buttonup(button)


def _style(app, ry, x):
    return app.game.styles[ry][x]


def _occupied(app, ry, x):
    return app.game.rows[ry] >> x & 1


# ------------------------------------------------------------------ painting

def test_left_click_paints_gray_right_and_shift_click_erase():
    app = _app_for()
    assert app.edit_enabled
    _click(app, 38, 4)
    assert _occupied(app, 38, 4) and _style(app, 38, 4) == EDITOR_GRAY
    _click(app, 38, 4, button=3)
    assert not _occupied(app, 38, 4) and _style(app, 38, 4) == -1
    _click(app, 38, 4)
    _click(app, 38, 4, shift=True)
    assert not _occupied(app, 38, 4)
    # erase of an empty cell is a no-op that pushes no undo snapshot
    stack = len(app._undo_stack)
    _click(app, 30, 0, button=3)
    assert len(app._undo_stack) == stack


def test_drag_paints_a_line_with_no_gaps():
    app = _app_for()
    _drag(app, [(38, 2), (38, 6)])
    for x in range(2, 7):
        assert _occupied(app, 38, x) and _style(app, 38, x) == EDITOR_GRAY
    assert not _occupied(app, 38, 1) and not _occupied(app, 38, 7)


def test_drag_over_existing_piece_cells_restyles_them():
    app = _app_for()
    # a real piece locks on the floor with piece styles
    app.game.spawn_forced(PieceType.O)
    app.game.tick([])
    from tetris.engine.game import Action

    app.game.tick([Action.HARD_DROP])
    assert any(_style(app, 39, x) == PieceType.O.value for x in range(10))
    _drag(app, [(39, 4), (39, 5)])
    assert _style(app, 39, 4) == EDITOR_GRAY  # overwritten by the stroke


def test_clicks_outside_the_field_do_nothing():
    app = _app_for()
    stack = len(app._undo_stack)
    app.handle_mouse_buttondown((100, 100), 1, False)
    app.handle_mouse_buttonup(1)
    app.handle_mouse_motion((100, 100), (1, 0, 0))
    assert len(app._undo_stack) == stack
    assert all(row == 0 for row in app.game.rows)


# ------------------------------------------------------------- recognition

def test_four_cell_drag_is_colored_as_the_tetromino():
    app = _app_for()
    # straight line -> I (cyan)
    _drag(app, [(38, 2), (38, 3), (38, 4), (38, 5)])
    for x in range(2, 6):
        assert _style(app, 38, x) == PieceType.I.value
    # L shape -> J or L by chirality (cells are (row, col))
    dragged = [(37, 7), (38, 7), (39, 7), (39, 8)]
    piece = B.piece_from_cells(dragged)
    _drag(app, dragged)
    for ry, x in dragged:
        assert _style(app, ry, x) == piece.value


def test_five_cell_drag_leaves_everything_gray():
    app = _app_for()
    _drag(app, [(38, 2), (38, 3), (38, 4), (38, 5), (38, 6)])
    for x in range(2, 7):
        assert _style(app, 38, x) == EDITOR_GRAY  # the 4-coloring reverted


def test_four_disconnected_cells_stay_gray():
    app = _app_for()
    _drag(app, [(38, 0), (38, 2), (38, 4), (38, 6)])  # gaps between cells
    for x in (0, 2, 4, 6):
        assert _style(app, 38, x) == EDITOR_GRAY


def test_component_recolor_across_separate_strokes():
    # four-tris AutoColor: a fresh paint that completes a gray component of
    # exactly 4 colors the whole component, even cells from earlier strokes
    app = _app_for()
    _click(app, 38, 2)
    _click(app, 38, 3)
    _click(app, 38, 4)
    assert _style(app, 38, 4) == EDITOR_GRAY
    _click(app, 38, 5)  # 4th cell of the connected component
    for x in range(2, 6):
        assert _style(app, 38, x) == PieceType.I.value


def test_oversized_component_never_recolors():
    app = _app_for()
    # a 5-cell DRAG: the 4th cell colors as I, the 5th reverts it to gray
    _drag(app, [(38, 0), (38, 1), (38, 2), (38, 3), (38, 4)])
    for x in range(5):
        assert _style(app, 38, x) == EDITOR_GRAY
    # separate 1-cell strokes completing a 4-component DO recolor
    # (four-tris AutoColor across strokes)
    _click(app, 38, 6)
    _click(app, 38, 7)
    _click(app, 38, 8)
    _click(app, 38, 9)
    for x in range(6, 10):
        assert _style(app, 38, x) == PieceType.I.value


# ------------------------------------------------------------------- undo

def test_edit_stroke_is_undoable():
    app = _app_for()
    _hard_drop = lambda a: (a.controller.press(Btn.HARD), a.logic_tick(),
                            a.controller.release(Btn.HARD), a.logic_tick())
    _hard_drop(app)
    before = list(app.game.rows), [list(r) for r in app.game.styles]
    _drag(app, [(38, 2), (38, 3), (38, 4), (38, 5)])
    assert any(_occupied(app, 38, x) for x in range(2, 6))
    _undo_key(app)
    # the board reverts to exactly the pre-stroke state (placed piece kept)
    assert (list(app.game.rows), [list(r) for r in app.game.styles]) == before
    assert app.game.active is not None


# -------------------------------------------------------------- mode gating

def test_mouse_editing_disabled_outside_zen_modes():
    app = _app_for("Sprint 40 Lines")
    assert not app.edit_enabled
    _click(app, 38, 4)
    assert all(row == 0 for row in app.game.rows)
    assert app._undo_stack == []
    # the queue box click also does nothing
    app.renderer.next_rect = pygame.Rect(0, 0, 50, 50)
    app.handle_mouse_buttondown((10, 10), 1, False)
    assert app.queue_edit is None


def test_hover_tracking_and_dialog_blocks_editing():
    app = _app_for()
    app.handle_mouse_motion(_cell_pos(app, 35, 3), (0, 0, 0))
    assert app._hover_cell == (35, 3)
    app.handle_mouse_motion((10, 10), (0, 0, 0))
    assert app._hover_cell is None
    # while the queue dialog is open the board ignores mouse edits
    app.renderer.next_rect = pygame.Rect(0, 0, 100, 100)
    app.handle_mouse_buttondown((10, 10), 1, False)
    assert app.queue_edit is not None
    _click(app, 38, 4)
    assert all(row == 0 for row in app.game.rows)
    # and the game is frozen while it is open
    before = app.game.tick_count
    app.logic_tick()
    assert app.game.tick_count == before
    # typing into the dialog is swallowed, not routed to the game
    from tetris.config import Keybinds

    event = type("FakeKeydown", (), {"mod": 0})()
    app.handle_key(pygame.K_i, event)
    assert app.queue_edit["seq"].endswith("I")
    assert app.game.active.rot == 0
