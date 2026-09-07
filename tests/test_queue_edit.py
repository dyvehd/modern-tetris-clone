"""Queue editor dialog (four-tris BagSet-style): click the next box, type a
piece sequence + a 7-bag offset, ENTER applies — the queue is replaced and
followed by fresh shuffled bags. The offset sets the bag phase (separator
positions in the preview). Ctrl+Z reverts a queue edit."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pytest  # noqa: E402

pygame = pytest.importorskip("pygame")

from tetris.app import App  # noqa: E402
from tetris.engine.constants import PIECE_LETTERS, PieceType  # noqa: E402

from test_editor_app import _app_for  # noqa: E402
from test_undo import _undo_key  # noqa: E402


def _event(mod=0):
    return type("FakeKeydown", (), {"mod": mod})()


def _click_next_box(app) -> None:
    """A left click inside the next-queue box (needs a rendered frame so the
    box rect exists)."""
    app.render()
    box = app.renderer.next_rect
    assert box is not None
    app.handle_mouse_buttondown(box.center, 1, False)
    app.handle_mouse_buttonup(1)


def _type(app, text: str) -> None:
    for ch in text:
        # pygame defines single letters lowercase
        code = getattr(pygame, f"K_{ch.lower()}", None)
        if code is None:
            code = getattr(pygame, f"K_{ch.upper()}")
        app.handle_key(code, _event())


def _key(app, name: str) -> None:
    code = getattr(pygame, f"K_{name}")
    app.handle_key(code, _event())


def test_clicking_next_box_opens_dialog_prefilled_with_queue():
    app = _app_for()
    _click_next_box(app)
    assert app.queue_edit is not None
    expected = "".join(PIECE_LETTERS[p] for p in app.game.queue)
    assert app.queue_edit["seq"] == expected
    assert app.queue_edit["off"] == "0"
    assert app.queue_edit["field"] == 0


def test_typing_and_applying_replaces_queue():
    app = _app_for()
    _click_next_box(app)
    # clear the prefill, type a fresh sequence
    for _ in range(20):
        _key(app, "BACKSPACE")
    _type(app, "SIOLTZJ")
    _key(app, "RETURN")
    assert app.queue_edit is None
    want = [PieceType.S, PieceType.I, PieceType.O, PieceType.L,
            PieceType.T, PieceType.Z, PieceType.J]
    assert list(app.game.queue[:7]) == want
    assert len(app.game.queue) >= 7  # random bags topped it up
    # the sequence is dealt in order
    for piece in want:
        assert app.game._take_from_queue() is piece


def test_offset_sets_bag_phase():
    app = _app_for()
    _click_next_box(app)
    _key(app, "TAB")  # to the offset field
    _type(app, "3")
    _key(app, "TAB")  # back to sequence
    for _ in range(20):
        _key(app, "BACKSPACE")
    _type(app, "TI")
    _key(app, "RETURN")
    assert app.game.bag_pos == 3
    assert list(app.game.queue[:2]) == [PieceType.T, PieceType.I]
    # bag boundaries in the preview fall after 7-offset pieces
    assert (app.game.bag_pos + 0 + 1) % 7 != 0  # after piece 0: mid-bag
    app.game._take_from_queue()  # deal the T: 4 pieces dealt of this bag
    assert app.game.bag_pos == 4


def test_invalid_letters_are_rejected_with_error():
    app = _app_for()
    _click_next_box(app)
    for _ in range(20):
        _key(app, "BACKSPACE")
    _type(app, "SIX")  # X is not a piece letter
    assert "letters" in app.queue_edit["error"]
    assert app.queue_edit["seq"] == "SI"  # the X never landed
    # an error does not block applying the valid part
    _key(app, "RETURN")
    assert list(app.game.queue[:2]) == [PieceType.S, PieceType.I]


def test_escape_cancels_without_applying():
    app = _app_for()
    before = list(app.game.queue)
    _click_next_box(app)
    for _ in range(20):
        _key(app, "BACKSPACE")
    _type(app, "ZZZZ")
    _key(app, "ESCAPE")
    assert app.queue_edit is None
    assert list(app.game.queue) == before


def test_empty_sequence_returns_to_random_bags():
    app = _app_for()
    _click_next_box(app)
    for _ in range(20):
        _key(app, "BACKSPACE")
    assert app.queue_edit["seq"] == ""
    _key(app, "RETURN")
    assert len(app.game.queue) >= 7  # refilled by the randomizer


def test_queue_edit_is_undoable():
    app = _app_for()
    before = list(app.game.queue)
    _click_next_box(app)
    for _ in range(20):
        _key(app, "BACKSPACE")
    _type(app, "JJJ")
    _key(app, "RETURN")
    assert list(app.game.queue[:3]) == [PieceType.J] * 3
    _undo_key(app)
    assert list(app.game.queue) == before


def test_offset_only_change_is_undoable_and_applies():
    app = _app_for()
    pre_bag_pos = app.game.bag_pos
    _click_next_box(app)
    _key(app, "TAB")
    _type(app, "6")
    _key(app, "RETURN")
    assert app.game.bag_pos == 6
    _undo_key(app)
    assert app.game.bag_pos == pre_bag_pos


def test_queue_dialog_only_in_edit_modes():
    app = _app_for("Sprint 40 Lines")
    app.render()
    box = app.renderer.next_rect
    app.handle_mouse_buttondown(box.center, 1, False)
    app.handle_mouse_buttonup(1)
    assert app.queue_edit is None


def test_dialog_renders():
    app = _app_for()
    _click_next_box(app)
    app.render()  # the dialog overlay draws without errors
    _type(app, "T")
    app.render()
    assert app.queue_edit["seq"].endswith("T")
