"""In-game settings screen: handling config, key remapping, persistence."""

import math
import os
import tomllib

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pytest  # noqa: E402

pygame = pytest.importorskip("pygame")  # noqa: E402

from tetris.app import App, parse_key_names  # noqa: E402
from tetris.config import (  # noqa: E402
    AppConfig,
    DebugConfig,
    InputConfig,
    Keybinds,
    load_config,
    save_config,
)


def make_event(key: int, repeat: bool = False):
    """A real pygame event, like the ones pygame.event.get() delivers.

    Normal keydowns carry NO .repeat attribute (accessing it raises
    AttributeError) — regression guard for the settings-screen crash.
    """
    if repeat:
        return pygame.event.Event(pygame.KEYDOWN, key=key, repeat=True)
    return pygame.event.Event(pygame.KEYDOWN, key=key)


def fresh_app(tmp_path):
    return App(config=AppConfig(debug=DebugConfig(log_input=False)),
               settings_path=tmp_path / "settings.toml")


# --- config persistence ------------------------------------------------------


def test_save_config_round_trip(tmp_path):
    cfg = AppConfig()
    cfg.input.das_ms = 133.0
    cfg.input.arr_ms = 0.0
    cfg.input.sdf = math.inf
    cfg.keys.left = "g"
    path = tmp_path / "settings.toml"
    save_config(cfg, path)

    data = tomllib.loads(path.read_text())
    assert data["input"]["das_ms"] == 133.0
    assert data["input"]["arr_ms"] == 0.0
    assert data["input"]["sdf"] == math.inf  # TOML `inf` round-trips
    assert data["keys"]["left"] == "g"


def test_save_config_preserves_other_sections(tmp_path):
    path = tmp_path / "settings.toml"
    path.write_text(
        "[rules]\nlock_delay_ms = 400\n\n"
        "[input]\ndas_ms = 200.0\n"
    )
    cfg = AppConfig()
    cfg.input.das_ms = 100.0
    save_config(cfg, path)

    data = tomllib.loads(path.read_text())
    assert data["rules"]["lock_delay_ms"] == 400  # hand-edited section survived
    assert data["input"]["das_ms"] == 100.0
    assert "keys" in data


def test_saved_settings_are_loaded_on_startup(tmp_path):
    path = tmp_path / "settings.toml"
    cfg = AppConfig()
    cfg.input.das_ms = 120.0
    cfg.keys.hard_drop = "f"
    save_config(cfg, path)

    import dataclasses

    loaded = AppConfig()
    data = tomllib.loads(path.read_text())
    names = {f.name for f in dataclasses.fields(loaded.input)}
    for key, value in data["input"].items():
        assert key in names
    assert data["keys"]["hard_drop"] == "f"


def test_load_config_still_works_with_defaults(monkeypatch, tmp_path):
    import tetris.config

    # hermetic: point the loader at an empty search path
    monkeypatch.setattr(tetris.config, "_SETTINGS_PATHS", (tmp_path / "none.toml",))
    cfg = load_config()
    assert cfg.input.das_ms == 167.0
    assert cfg.keys.hard_drop == "space"


# --- settings screen flow ----------------------------------------------------


def test_settings_open_adjust_and_capture(tmp_path):
    app = fresh_app(tmp_path)
    app.handle_key(pygame.K_s, make_event(pygame.K_s))
    assert app.state == App.STATE_SETTINGS

    # first selectable row is DAS: RIGHT adds 5 ms
    app.handle_key(pygame.K_RIGHT, make_event(pygame.K_RIGHT))
    assert app.cfg.input.das_ms == 172.0

    # DOWN to ARR, LEFT twice lowers it toward 0 (floored)
    app.handle_key(pygame.K_DOWN, make_event(pygame.K_DOWN))
    app.handle_key(pygame.K_LEFT, make_event(pygame.K_LEFT))
    app.handle_key(pygame.K_LEFT, make_event(pygame.K_LEFT))
    assert app.cfg.input.arr_ms == 23.0

    # DOWN to SDF and walk it all the way to infinity
    app.handle_key(pygame.K_DOWN, make_event(pygame.K_DOWN))
    for _ in range(40):
        app.handle_key(pygame.K_RIGHT, make_event(pygame.K_RIGHT))
    assert math.isinf(app.cfg.input.sdf)
    app.handle_key(pygame.K_LEFT, make_event(pygame.K_LEFT))
    assert app.cfg.input.sdf == 40.0

    # capture: one more DOWN lands on the first bind row (Move left)
    app.handle_key(pygame.K_DOWN, make_event(pygame.K_DOWN))
    app.handle_key(pygame.K_RETURN, make_event(pygame.K_RETURN))
    assert app.capture_field == "left"
    app.handle_key(pygame.K_g, make_event(pygame.K_g))
    assert app.capture_field is None
    assert app.cfg.keys.left == "g"

    # ESC saves and returns to the menu
    app.handle_key(pygame.K_ESCAPE, make_event(pygame.K_ESCAPE))
    assert app.state == App.STATE_MENU

    data = tomllib.loads((tmp_path / "settings.toml").read_text())
    assert data["input"]["das_ms"] == 172.0
    assert data["keys"]["left"] == "g"


def test_rebound_key_routes_in_game(tmp_path):
    app = fresh_app(tmp_path)
    app.open_settings(App.STATE_MENU)
    # selectable rows: das(0), arr(1), sdf(2), Move left(3), ...
    app.settings_idx = 3
    app.handle_key(pygame.K_RETURN, make_event(pygame.K_RETURN))
    app.handle_key(pygame.K_g, make_event(pygame.K_g))
    assert app.cfg.keys.left == "g"
    app.close_settings()

    # start a game: pressing g must move the piece left
    app.start_game()
    app.game.tick([])  # first tick performs the spawn
    x0 = app.game.active.x
    app.handle_key(pygame.K_g, make_event(pygame.K_g))
    actions, held = app.controller.update()
    app.game.tick(actions, held)
    assert app.game.active.x == x0 - 1
    # the old binding no longer fires
    x1 = app.game.active.x
    app.handle_key(pygame.K_LEFT, make_event(pygame.K_LEFT))
    actions, held = app.controller.update()
    app.game.tick(actions, held)
    assert app.game.active.x == x1


def test_rebinding_via_modifier_key_round_trips(tmp_path):
    app = fresh_app(tmp_path)
    app.open_settings(App.STATE_MENU)
    app.settings_idx = 3  # Move left
    app.handle_key(pygame.K_RETURN, make_event(pygame.K_RETURN))
    app.handle_key(pygame.K_LSHIFT, make_event(pygame.K_LSHIFT))
    assert app.cfg.keys.left == "lshift"  # round-trips through the parser
    assert pygame.K_LSHIFT in parse_key_names(app.cfg.keys.left)


def test_backspace_clears_binding(tmp_path):
    app = fresh_app(tmp_path)
    app.open_settings(App.STATE_MENU)
    app.settings_idx = 3
    app.handle_key(pygame.K_RETURN, make_event(pygame.K_RETURN))
    app.handle_key(pygame.K_BACKSPACE, make_event(pygame.K_BACKSPACE))
    assert app.cfg.keys.left == ""
    app.render()  # renders "-" for the empty bind


def test_reset_defaults(tmp_path):
    app = fresh_app(tmp_path)
    app.cfg.input.das_ms = 50.0
    app.cfg.keys.left = "g"
    app.open_settings(App.STATE_MENU)
    app.handle_key(pygame.K_r, make_event(pygame.K_r))
    assert app.cfg.input.das_ms == 167.0
    assert app.cfg.keys.left == "left"


def test_escape_cancels_capture_without_binding(tmp_path):
    app = fresh_app(tmp_path)
    app.open_settings(App.STATE_MENU)
    app.settings_idx = 3
    app.handle_key(pygame.K_RETURN, make_event(pygame.K_RETURN))
    app.handle_key(pygame.K_ESCAPE, make_event(pygame.K_ESCAPE))
    assert app.capture_field is None
    assert app.cfg.keys.left == "left"  # unchanged


def test_normal_keydown_events_have_no_repeat_attribute(tmp_path):
    """Regression: the settings screen crashed with AttributeError because
    pygame-ce keydown events only carry .repeat on auto-repeat events. Pump
    real events through handle_events() exactly like the main loop does."""
    app = fresh_app(tmp_path)
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_s))
    assert app.handle_events() is True
    assert app.state == App.STATE_SETTINGS

    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFT))
    assert app.handle_events() is True  # used to crash right here
    assert app.cfg.input.das_ms == 162.0  # LEFT = -5 ms

    # auto-repeat events (which DO have .repeat=True) also adjust while held
    for _ in range(3):
        pygame.event.post(
            pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFT, repeat=True)
        )
    app.handle_events()
    assert app.cfg.input.das_ms == 147.0

    # and a real KEYUP for the released arrow is harmless in settings
    pygame.event.post(pygame.event.Event(pygame.KEYUP, key=pygame.K_LEFT))
    app.handle_events()


def test_settings_screen_renders(tmp_path):
    app = fresh_app(tmp_path)
    app.open_settings(App.STATE_MENU)
    app.handle_key(pygame.K_RETURN, make_event(pygame.K_RETURN))  # capture on
    app.render()
    app.handle_key(pygame.K_ESCAPE, make_event(pygame.K_ESCAPE))
    app.render()
