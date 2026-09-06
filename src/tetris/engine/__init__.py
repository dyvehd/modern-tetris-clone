"""Pure-Python Tetris engine: deterministic, dependency-free, AI-ready.

Nothing in this package may import pygame (enforced by a unit test).
"""

from .constants import (
    FIELD_H,
    FIELD_W,
    FULL_ROW,
    VISIBLE_H,
    VISIBLE_TOP,
    PieceType,
)
from .game import Action, ActivePiece, Btn, Game, GameConfig
from .scoring import evaluate_clear

__all__ = [
    "Action",
    "ActivePiece",
    "Btn",
    "Game",
    "GameConfig",
    "PieceType",
    "FIELD_H",
    "FIELD_W",
    "FULL_ROW",
    "VISIBLE_H",
    "VISIBLE_TOP",
    "evaluate_clear",
]
