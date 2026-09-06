"""App smoke test with SDL dummy video driver + engine-level gameplay drive."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pytest  # noqa: E402

from tetris.engine import board as B  # noqa: E402
from tetris.engine.constants import FULL_ROW, PieceType  # noqa: E402
from tetris.engine.env import Action, TetrisEnv  # noqa: E402


def test_full_game_headless():
    env = TetrisEnv(seed=2024)
    placed = 0
    for piece_idx in range(300):
        # steer each piece to a different column so the stack stays flat
        target = piece_idx % 10
        for _ in range(20):
            piece = env.observe().piece
            if piece is None:
                break
            x = piece[2]
            if x < target:
                env.step(Action.RIGHT)
            elif x > target:
                env.step(Action.LEFT)
            else:
                break
        obs, _ = env.step(Action.HARD_DROP)
        if obs.over:
            break
        placed = obs.pieces_placed
    assert placed > 20  # the engine plays a full game without crashing


def test_sprint_goal_finishes_game():
    cfg_rows = [0] * 40
    # board one quad away from 40 lines... simpler: goal_lines=1 with a quad
    from tetris.engine.game import Game, GameConfig

    cfg = GameConfig(gravity_g=0.0, lock_delay_ms=100_000, goal_lines=1)
    game = Game(cfg, seed=1)
    for row in (36, 37, 38, 39):
        game.rows[row] = FULL_ROW & ~1  # col 0 open
    game.spawn_forced(PieceType.I)
    game.active.rot = 1  # vertical I through the col-0 shaft
    game.active.x = -2
    game.active.y = 36
    game.tick([Action.HARD_DROP])
    assert game.lines == 4
    assert game.over and game.won


def test_parse_key_names_resolves_all_default_keybinds():
    """Regression: pygame's K_* constants are uppercase except single letters,
    so lowercase names from settings.toml must resolve case-insensitively."""
    pygame = pytest.importorskip("pygame")
    import dataclasses

    from tetris.app import parse_key_names
    from tetris.config import Keybinds

    for field in dataclasses.fields(Keybinds()):
        spec = getattr(Keybinds(), field.name)
        codes = parse_key_names(spec)
        assert codes, f"keybind '{field.name}' ('{spec}') resolved to no keys"
    # spot-check the ones that broke before (uppercase constants)
    assert pygame.K_LEFT in parse_key_names("left")
    assert pygame.K_ESCAPE in parse_key_names("escape")
    assert pygame.K_SPACE in parse_key_names("space")
    assert pygame.K_DOWN in parse_key_names("down")
    assert pygame.K_UP in parse_key_names("up")
    # single letters exist lowercase in pygame
    assert pygame.K_z in parse_key_names("z")
    # cached: identical result on repeat calls
    assert parse_key_names("up,x") is parse_key_names("up,x")


def test_pygame_app_smoke(monkeypatch, tmp_path):
    pytest.importorskip("pygame")
    from tetris.app import App
    from tetris.config import AppConfig, DebugConfig

    app = App(AppConfig(debug=DebugConfig(log_input=False)))
    app.state = App.STATE_PLAY
    app.start_game()
    assert app.game is not None

    # drive ~5 seconds of game time with some inputs
    from tetris.engine.game import Btn

    app.controller.press(Btn.LEFT)
    for i in range(400):
        if i % 60 == 10:
            app.controller.press(Btn.HARD)
        if i % 60 == 11:
            app.controller.release(Btn.HARD)
        if i % 60 == 20:
            app.controller.press(Btn.ROT_CW)
        if i % 60 == 21:
            app.controller.release(Btn.ROT_CW)
        app.logic_tick()
    assert app.game.pieces_placed >= 4

    # render a few frames (also exercise menu + game over paths)
    app.render()
    app.state = App.STATE_MENU
    app.render()
    app.game.over = True
    app.state = App.STATE_OVER
    app.render()

    # save a screenshot for manual inspection
    pygame = pytest.importorskip("pygame")
    out = tmp_path / "smoke.png"
    pygame.image.save(app.screen, str(out))
    assert out.exists() and out.stat().st_size > 1000
    pygame.quit()
