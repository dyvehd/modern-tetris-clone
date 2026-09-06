"""Headless Tetris environment for AI development.

Design goals:
- ``TetrisEnv`` is a thin, gym-style wrapper around ``Game``: one ``step`` =
  one logic tick (1/60 s), fully deterministic under a seed.
- The board is exposed as plain integers (row bitmasks) so search/learning
  code (and a future Numba/Cython port) can operate without engine objects.
- ``reachable_placements`` enumerates where the active piece can be dropped,
  the standard interface for placement-selection AIs.

Example:
    env = TetrisEnv(seed=42)
    obs = env.reset()
    obs, info = env.step(Action.HARD_DROP)
"""

from __future__ import annotations

from dataclasses import dataclass

from .board import collides, ghost_y
from .constants import PieceType, SPAWN_X, SPAWN_Y
from .game import Action, Btn, Game, GameConfig

__all__ = ["Action", "Btn", "Game", "GameConfig", "TetrisEnv", "Observation", "reachable_placements"]


@dataclass
class Observation:
    """Full state snapshot for AI consumption."""

    rows: tuple[int, ...]  # FIELD_H row bitmasks (row 0 = top)
    piece: tuple[PieceType, int, int, int] | None  # (type, rot, x, y)
    next: tuple[PieceType, ...]
    hold: PieceType | None
    can_hold: bool
    combo: int
    b2b: int
    score: int
    lines: int
    level: int
    attack_sent: int
    pending_garbage: int
    pieces_placed: int
    tick: int
    over: bool
    won: bool

    def visible_rows(self) -> tuple[int, ...]:
        return self.rows[-20:]


class TetrisEnv:
    def __init__(self, config: GameConfig | None = None, seed: int | None = None) -> None:
        self.game = Game(config or GameConfig(), seed)

    # ------------------------------------------------------------- lifecycle

    def reset(self, seed: int | None = None) -> Observation:
        self.game = Game(self.game.cfg, seed)
        return self.observe()

    def step(self, action: Action, held: frozenset[Btn] = frozenset()) -> tuple[Observation, dict]:
        """Advance one logic tick with a single action."""
        self.game.tick([action], held)
        return self.observe(), self.info()

    def step_tick(self, actions: list[Action] = (), held: frozenset[Btn] = frozenset()) -> tuple[Observation, dict]:
        """Advance one logic tick with several simultaneous actions."""
        self.game.tick(list(actions), held)
        return self.observe(), self.info()

    def add_garbage(self, rows: int) -> None:
        """Simulate an opponent sending garbage."""
        self.game.add_garbage(rows)

    # ----------------------------------------------------------------- views

    def observe(self) -> Observation:
        g = self.game
        return Observation(
            rows=tuple(g.rows),
            piece=(g.active.type, g.active.rot, g.active.x, g.active.y) if g.active else None,
            next=tuple(g.queue[:5]),
            hold=g.hold_type,
            can_hold=g.can_hold,
            combo=g.combo,
            b2b=g.b2b_chain,
            score=g.score,
            lines=g.lines,
            level=g.level,
            attack_sent=g.attack_sent,
            pending_garbage=sum(len(b.rows) for b in g.garbage_queue),
            pieces_placed=g.pieces_placed,
            tick=g.tick_count,
            over=g.over,
            won=g.won,
        )

    def info(self) -> dict:
        return {"events": list(self.game.events)}

    # ----------------------------------------------------------- AI helpers

    def reachable_placements(self) -> list[tuple[int, int, int]]:
        """All (x, rot, y) hard-drop landing spots for the active piece,
        via BFS over (x, y, rot) with shifts and rotations (no gravity)."""
        g = self.game
        if g.active is None:
            return []
        piece = g.active.type
        seen: set[tuple[int, int, int]] = set()
        frontier: list[tuple[int, int, int]] = []
        start = (g.active.x, g.active.y, g.active.rot)
        frontier.append(start)
        seen.add(start)
        landings: dict[tuple[int, int], tuple[int, int, int]] = {}
        while frontier:
            x, y, rot = frontier.pop()
            gy = ghost_y(g.rows, piece, rot, x, y)
            landing = (x, rot, gy)
            key = (x, rot, gy)
            if key not in landings:
                landings[key] = landing
            moves = (
                (x - 1, y, rot),
                (x + 1, y, rot),
                (x, y, (rot + 1) % 4),
                (x, y, (rot - 1) % 4),
                (x, y, (rot + 2) % 4),
            )
            for nx, ny, nrot in moves:
                state = (nx, ny, nrot)
                if state in seen:
                    continue
                if collides(g.rows, piece, nrot, nx, ny):
                    continue
                seen.add(state)
                frontier.append(state)
        return sorted(landings.values())

    def clone(self) -> TetrisEnv:
        env = TetrisEnv.__new__(TetrisEnv)
        env.game = self.game.clone()
        return env


def reachable_placements(rows: list[int], piece: PieceType) -> list[tuple[int, int, int]]:
    """Standalone placement search from the spawn position (module-level
    helper for batch simulation / feature engineering)."""
    seen: set[tuple[int, int, int]] = {(SPAWN_X[piece], SPAWN_Y, 0)}
    frontier = [(SPAWN_X[piece], SPAWN_Y, 0)]
    landings: dict[tuple[int, int], tuple[int, int, int]] = {}
    while frontier:
        x, y, rot = frontier.pop()
        gy = ghost_y(rows, piece, rot, x, y)
        landings.setdefault((x, rot), (x, rot, gy))
        for nx, ny, nrot in (
            (x - 1, y, rot),
            (x + 1, y, rot),
            (x, y, (rot + 1) % 4),
            (x, y, (rot - 1) % 4),
            (x, y, (rot + 2) % 4),
        ):
            state = (nx, ny, nrot)
            if state in seen or collides(rows, piece, nrot, nx, ny):
                continue
            seen.add(state)
            frontier.append(state)
    return sorted(landings.values())
