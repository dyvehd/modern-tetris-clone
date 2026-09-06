"""Headless env: determinism, cloning, placement search, engine purity."""

import subprocess
import sys
import sysconfig

from tetris.engine import board as B
from tetris.engine.constants import PieceType
from tetris.engine.env import Action, TetrisEnv, reachable_placements


def script(actions, length=200):
    """Pseudo-random but fixed action script exercising the whole API."""
    import random

    rng = random.Random(actions)
    out = []
    for _ in range(length):
        out.append(rng.choice(list(Action)))
    return out


def test_same_seed_same_game():
    env_a = TetrisEnv(seed=123)
    env_b = TetrisEnv(seed=123)
    for action in script(1):
        obs_a, _ = env_a.step(action)
        obs_b, _ = env_b.step(action)
        assert obs_a == obs_b
        if obs_a.over:
            break
    assert obs_a.over or obs_a.pieces_placed > 20


def test_different_seeds_diverge():
    env_a = TetrisEnv(seed=1)
    env_b = TetrisEnv(seed=2)
    for action in script(2)[:100]:
        obs_a, _ = env_a.step(action)
        obs_b, _ = env_b.step(action)
        if obs_a != obs_b:
            return
    raise AssertionError("games with different seeds never diverged")


def test_clone_is_independent():
    env = TetrisEnv(seed=9)
    for _ in range(30):
        env.step(Action.HARD_DROP)
        if env.game.over:
            env.reset(seed=9)
    env.reset(seed=9)  # known clean state
    clone = env.clone()
    clone.step(Action.HARD_DROP)
    assert clone.observe() != env.observe()
    clone.game.rows[39] |= 1
    assert env.game.rows[39] == 0


def test_observation_shape():
    env = TetrisEnv(seed=5)
    env.step(Action.NOOP)  # first tick performs the initial spawn
    obs = env.observe()
    assert len(obs.rows) == 40
    assert len(obs.next) == 5
    assert obs.piece is not None and len(obs.piece) == 4
    assert obs.hold is None and obs.can_hold is True


def test_reachable_placements_empty_board():
    rows = [0] * 40
    placements = reachable_placements(rows, PieceType.T)
    # rot 0/2 span 3 columns: x = 0..7; rot 1 can hang off the left wall
    # (x = -1..7); rot 3 off the right (x = 0..8) -> 8+9+8+9 = 34
    assert len(placements) == 34
    # rot 0 rests on its bar (bottom = box row 1 -> y=38); the other three
    # rotations reach one row lower (bottom = box row 2 -> y=37)
    assert all(y == (38 if rot == 0 else 37) for _, rot, y in placements)


def test_reachable_placements_respect_stack():
    rows = [0] * 40
    for row in (39,):
        rows[row] = 0b1111110111  # col 3 open
    placements = reachable_placements(rows, PieceType.I)
    ys = {(x, rot): y for x, rot, y in placements}
    # vertical I in the col-3 shaft lands lower than on top of the stack
    x, rot, y = next(p for p in placements if p[1] == 1 and p[0] == 1)
    assert y == 36  # I vertical occupies rows 36-39 in the shaft


def test_env_step_soft_drop_and_garbage():
    from tetris.engine.game import GameConfig

    env = TetrisEnv(GameConfig(garbage_delay_ms=5000), seed=11)
    env.add_garbage(2)
    for _ in range(120):
        env.step(Action.SOFT_DROP)
    obs = env.observe()
    assert obs.pending_garbage == 2  # not due yet, still queued


def test_engine_has_no_pygame_dependency():
    code = (
        "import sys; import tetris.engine; import tetris.input; import tetris.config;"
        "assert 'pygame' not in sys.modules, 'engine imported pygame!'; print('clean')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"),
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


def test_engine_sources_never_import_pygame():
    import pathlib
    import re

    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "tetris"
    pattern = re.compile(r"^\s*(import pygame|from pygame)", re.MULTILINE)
    for path in list(src.rglob("*.py")):
        if "render" in path.parts or path.name == "app.py":
            continue
        text = path.read_text()
        assert not pattern.search(text), f"{path} imports pygame"
