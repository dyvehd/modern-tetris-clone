"""Input event log: documented format, spawn dedup, app wiring end-to-end."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pytest  # noqa: E402

from tetris.engine.constants import PieceType  # noqa: E402
from tetris.engine.game import Btn  # noqa: E402
from tetris.input.logger import InputLogger  # noqa: E402


class FakePiece:
    def __init__(self, piece_type):
        self.type = piece_type


def texts_of(path):
    """Log lines with the timestamp column stripped."""
    return [line.split(maxsplit=1)[1] for line in path.read_text().splitlines()]


def test_log_matches_the_documented_format(tmp_path):
    path = tmp_path / "input.log"
    log = InputLogger(path)
    log.key(Btn.LEFT, down=True)
    log.key(Btn.LEFT, down=False)
    log.key(Btn.HARD, down=True)
    log.check_piece(FakePiece(PieceType.L))
    log.key(Btn.HARD, down=False)
    log.close()
    assert texts_of(path) == [
        "Left (key down)",
        "Left (key up)",
        "Harddrop (key down)",
        "L piece",
        "Harddrop (key up)",
    ]


def test_timestamps_are_seconds_from_session_start(tmp_path):
    path = tmp_path / "input.log"
    log = InputLogger(path)
    for _ in range(5):
        log.key(Btn.RIGHT, True)
    log.close()
    stamps = [float(line.split(maxsplit=1)[0]) for line in path.read_text().splitlines()]
    assert len(stamps) == 5
    assert all(s >= 0.0 for s in stamps)
    assert stamps == sorted(stamps)


def test_piece_logged_once_per_spawn(tmp_path):
    path = tmp_path / "input.log"
    log = InputLogger(path)
    piece = FakePiece(PieceType.T)
    log.check_piece(piece)
    log.check_piece(piece)  # same object while falling: no repeat line
    log.check_piece(None)  # ARE gap between pieces: nothing logged
    log.check_piece(FakePiece(PieceType.S))  # next spawn
    log.close()
    assert texts_of(path) == ["T piece", "S piece"]


def test_file_created_lazily_and_closed_logger_is_noop(tmp_path):
    path = tmp_path / "sub" / "input.log"
    log = InputLogger(path)
    assert not path.exists()  # construction alone touches nothing
    log.note("hello")
    assert path.exists()
    log.close()
    log.key(Btn.HOLD, True)  # after close: dropped, not raised
    log.note("late")
    assert texts_of(path) == ["hello"]


def test_app_logs_keys_pieces_and_state(tmp_path):
    pytest.importorskip("pygame")
    import pygame

    from tetris.app import App
    from tetris.config import AppConfig

    cfg = AppConfig()
    cfg.debug.input_log_dir = str(tmp_path)
    cfg.rules.are_ms = 0
    cfg.rules.line_clear_delay_ms = 0
    cfg.rules.gravity_curve = False
    cfg.rules.gravity_g = 0.0
    app = App(cfg)
    assert app.keylog is not None

    app.start_game()
    app.handle_key(pygame.K_LEFT, None)  # raw event unused beyond .repeat
    app.logic_tick()  # piece 1 spawns; the left press is applied
    app.handle_keyup(pygame.K_LEFT)
    app.handle_key(pygame.K_SPACE, None)  # hard drop piece 1
    app.logic_tick()  # locks; piece 2 spawns immediately (ARE 0)
    app.handle_keyup(pygame.K_SPACE)
    app.handle_key(pygame.K_ESCAPE, None)  # pause
    app.keylog.close()

    log_file = next(tmp_path.glob("input_*.log"))
    texts = texts_of(log_file)
    header = texts[0]
    assert header.startswith("#")
    assert "game start: Marathon" in header and "seed" in header and "das" in header
    assert "Left (key down)" in texts
    assert "Left (key up)" in texts
    assert "paused" in texts
    # a new piece line lands between hard-drop down and its key-up,
    # exactly like the documented example
    hd = texts.index("Harddrop (key down)")
    hu = texts.index("Harddrop (key up)")
    piece_rows = [i for i, t in enumerate(texts) if t.endswith(" piece")]
    assert any(hd < i < hu for i in piece_rows)


def test_disabled_logger_leaves_app_fully_functional(tmp_path):
    pytest.importorskip("pygame")

    from tetris.app import App
    from tetris.config import AppConfig, DebugConfig

    app = App(AppConfig(debug=DebugConfig(log_input=False)))
    assert app.keylog is None
    app.start_game()
    for _ in range(5):
        app.logic_tick()
    assert app.game is not None and app.game.pieces_placed == 0


def _game_app(**rule_overrides):
    """A play-state app with a still field (0G) for input regression tests."""
    pytest.importorskip("pygame")

    from tetris.app import App
    from tetris.config import AppConfig, DebugConfig

    cfg = AppConfig(debug=DebugConfig(log_input=False))
    cfg.rules.gravity_g = 0.0
    cfg.rules.gravity_curve = False
    for key, value in rule_overrides.items():
        setattr(cfg.rules, key, value)
    app = App(cfg)
    app.start_game()
    app.handle_events()  # pump any startup events
    return app


def test_os_key_repeat_never_reaches_the_game(tmp_path):
    """Regression (user log 2026-09-07, Sprint): holding the hard-drop key
    dropped a piece every ~50 ms — pygame-ce's synthetic repeat KEYDOWNs
    were reaching gameplay because they carry no usable .repeat flag. A
    KEYDOWN for a key that is already down must be dropped."""
    pygame = pytest.importorskip("pygame")

    app = _game_app()
    pygame.key.set_repeat(300, 50)  # exactly the app's setting

    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
    app.handle_events()
    app.logic_tick()  # spawn piece 1, hard drop it
    assert app.game.pieces_placed == 1

    # same physical key held: 20 auto-repeat KEYDOWNs, one per frame
    for _ in range(20):
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
        app.handle_events()
        app.logic_tick()
    assert app.game.pieces_placed == 1  # exactly one drop from one press

    # a real release then a real press still works
    pygame.event.post(pygame.event.Event(pygame.KEYUP, key=pygame.K_SPACE))
    app.handle_events()
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
    app.handle_events()
    app.logic_tick()
    assert app.game.pieces_placed == 2


def test_held_rotate_key_does_not_pre_rotate_spawns(tmp_path):
    """Regression (user log 2026-09-07, Sprint): with CW held across a hard
    drop, every spawned piece entered with 1 CW rotation (the engine's old
    always-on IRS). Only the piece you actually rotated may be rotated."""
    pygame = pytest.importorskip("pygame")

    app = _game_app()

    # the random seed may deal an O piece first — O rotation is a no-op, so
    # force a T to make the rotation assertion deterministic
    app.game.spawn_forced(PieceType.T)
    app.game.tick([])
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP))
    app.handle_events()
    app.logic_tick()  # spawn piece 1, rotate it CW
    assert app.game.active.rot == 1
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
    app.handle_events()
    app.logic_tick()  # drop piece 1 with the CW key still held
    assert app.game.active.rot == 0  # piece 2 spawns unrotated

    # repeat KEYDOWNs of the still-held CW key must not rotate piece 2
    for _ in range(5):
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP))
        app.handle_events()
        app.logic_tick()
    assert app.game.active.rot == 0


def test_window_focus_lost_releases_held_keys(tmp_path):
    """Keys held when the window loses focus never get KEYUPs; their stale
    entries must be cleared or the next real press of that key is eaten."""
    pygame = pytest.importorskip("pygame")

    app = _game_app()

    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFT))
    app.handle_events()
    app.logic_tick()
    assert pygame.K_LEFT in app._keys_down
    assert Btn.LEFT in app.controller._dir_stack

    pygame.event.post(pygame.event.Event(pygame.WINDOWFOCUSLOST))
    app.handle_events()
    assert pygame.K_LEFT not in app._keys_down  # next press will register
    assert Btn.LEFT not in app.controller._dir_stack  # no stuck DAS


def test_menu_still_auto_repeats(tmp_path):
    """The menu keeps hold-to-scroll: synthetic repeats of a held DOWN key
    must keep moving the mode selection (dedup exempts menu/settings)."""
    pygame = pytest.importorskip("pygame")

    from tetris.app import App
    from tetris.config import AppConfig, DebugConfig

    app = App(AppConfig(debug=DebugConfig(log_input=False)))
    pygame.key.set_repeat(300, 50)
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DOWN))
    app.handle_events()
    idx = app.mode_idx
    for _ in range(3):
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DOWN))
        app.handle_events()
    assert app.mode_idx == idx + 3  # first press + 3 repeats all navigate
