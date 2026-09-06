"""Shared test helpers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from tetris.engine import board as B  # noqa: E402
from tetris.engine.constants import PieceType  # noqa: E402
from tetris.engine.game import Action, Btn, Game, GameConfig  # noqa: E402


def make_game(**overrides) -> Game:
    """Deterministic 0G test game: pieces never fall, lock delay is huge."""
    cfg = GameConfig(
        gravity_g=0.0,
        lock_delay_ms=100_000,
        are_ms=0,
        line_clear_delay_ms=0,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return Game(cfg, seed=1)


def place(game: Game, piece: PieceType, x: int, y: int, rot: int = 0) -> None:
    """Force-spawn a piece at an exact position (no collision check)."""
    game.spawn_forced(piece)
    active = game.active
    assert active is not None
    active.rot = rot
    active.x = x
    active.y = y


def force_queue(game: Game, pieces: list[PieceType]) -> None:
    """Pin the upcoming pieces (testing)."""
    game.queue.clear()
    game.queue.extend(pieces)


def hard_drop(game: Game, held: frozenset[Btn] = frozenset()) -> None:
    game.tick([Action.HARD_DROP], held)


def art_rows(art: str) -> list[int]:
    return B.from_ascii(art)
