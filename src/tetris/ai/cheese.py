"""Cheese-race episode harness — the evaluation environment for the AI stack.

The harness wraps the pure engine in an episode protocol set at the AI
stack's natural seam: an agent *predicts placements* (which resting position
the current — or held — piece should lock in), and the harness *navigates*
(converting the placement into inputs via movegen + pathfinder). That is the
position-prediction / navigation decoupling made literal: a placement-
predicting agent never emits an input, and every agent — random, search, or
learned — shares the same navigation layer.

Two application modes, chosen per run:

- ``navigate=True`` — the placement is converted to its shortest input
  sequence (:func:`find_path`) and replayed through :meth:`Game.tick`, one
  action per tick. The fully faithful path: the engine's own spin detection,
  lock, line clears, cheese refill and goal bookkeeping all run on the real
  input stream, and the result carries the per-piece input log for later
  humanized playback.
- ``navigate=False`` — the piece is teleported to the target position and
  hard-dropped (two ticks per piece). Line clears depend only on the merged
  cells, so every cheese-relevant outcome (dug lines, refill, win) is
  identical to the navigated run; only cosmetic spin labels are skipped
  (locks are recorded as plain placements). The fast path for large batches
  — weight tuning, RL rollouts.

The episode runs at 0G with infinite SDF, ARE 0 and a lock delay that never
fires: timing is a non-factor by design, so pieces-per-dug-line — the
cheese-race objective — is the only measured skill. An agent error
(illegal or unreachable placement) raises :class:`InvalidDecision` rather
than silently failing the episode.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from statistics import fmean, stdev

from ..engine import board as B
from ..engine.constants import FIELD_H, FIELD_W, PieceType
from ..engine.game import Action, Game, GameConfig
from .movegen import Placement, enumerate_placements
from .pathfinder import find_path


class InvalidDecision(Exception):
    """The agent returned a decision the harness cannot legally apply."""


@dataclass(frozen=True)
class CheeseEnv:
    """Episode parameters: what to dig, how the cheese behaves, what the
    agent may see. ``level`` cheese lines dug wins the episode; the stack is
    kept at ``stack`` rows (Jstris keeps 9 for every goal)."""

    level: int = 10  # cheese lines to dig (the goal)
    stack: int = 9  # cheese rows held on the board (Jstris: 9)
    messiness: float = 100.0  # hole-movement chance, % (Jstris cheese: 100)
    refill_on_clear: bool = False  # False = Jstris (a combo keeps the field down)
    hold_enabled: bool = True
    previews: int = 5  # upcoming pieces the agent may see (queue[0] is next)
    allow_180: bool = True
    piece_cap: int = 400  # placements before an unfinished episode is a failure

    def __post_init__(self) -> None:
        if self.level < 1:
            raise ValueError("level must be >= 1")
        if not 1 <= self.stack <= 20:
            raise ValueError("stack must be within 1..20")
        if not 0.0 <= self.messiness <= 100.0:
            raise ValueError("messiness must be within 0..100")
        if self.previews < 1:
            raise ValueError("previews must be >= 1")
        if self.piece_cap < 1:
            raise ValueError("piece_cap must be >= 1")

    def game_config(self) -> GameConfig:
        # 0G + infinite SDF + ARE 0: placement quality is the only measured
        # skill. The huge lock delay matches the movement model's "timing is
        # a non-factor" assumption (see the pathfinder docs).
        return GameConfig(
            gravity_g=0.0,
            soft_drop_factor=math.inf,
            lock_delay_ms=100_000,
            are_ms=0,
            line_clear_delay_ms=0,
            goal_lines=self.level,
            cheese_rows=self.stack,
            cheese_messiness=self.messiness,
            cheese_refill_on_clear=self.refill_on_clear,
            hold_enabled=self.hold_enabled,
        )

    @property
    def desc(self) -> str:
        refill = "jstris" if not self.refill_on_clear else "tetrio"
        return (
            f"L={self.level} stack={self.stack} mess={self.messiness:g}% "
            f"refill={refill} hold={'on' if self.hold_enabled else 'off'}"
        )


@dataclass(frozen=True)
class Obs:
    """A snapshot of everything the agent may know at a decision point:
    the board, the active piece, hold, the next ``queue`` pieces (already
    trimmed to the env's preview count), and the cheese counters. Frozen
    and copied — an agent cannot mutate the engine through it."""

    rows: tuple[int, ...]
    active: PieceType
    hold: PieceType | None
    can_hold: bool  # hold exists AND is usable right now (can_hold flags fold in hold_enabled)
    queue: tuple[PieceType, ...]
    cheese_on_board: int
    cheese_dug: int
    goal: int
    pieces_placed: int
    allow_180: bool

    def placements(self) -> list[Placement]:
        """Every reachable resting placement of the active piece."""
        return enumerate_placements(list(self.rows), self.active, allow_180=self.allow_180)

    @property
    def hold_piece(self) -> PieceType | None:
        """The piece a hold would bring out: the held piece, or the next
        from the queue when hold is empty."""
        return self.hold if self.hold is not None else (self.queue[0] if self.queue else None)

    def hold_placements(self) -> list[Placement]:
        """Placements of the piece a hold would bring out (empty when there
        is no piece to hold into)."""
        piece = self.hold_piece
        if piece is None:
            return []
        return enumerate_placements(list(self.rows), piece, allow_180=self.allow_180)


@dataclass(frozen=True)
class Decision:
    """One placement decision: lock ``placement`` — holding first when
    ``hold`` is set (the placement must then be for the piece the hold
    brings out, which the observation exposes)."""

    placement: Placement
    hold: bool = False


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    first_piece: PieceType
    won: bool
    pieces: int
    dug: int
    reason: str  # "cleared" | "topout" | "capped"
    decisions: tuple[Decision, ...]
    inputs: tuple[tuple[Action, ...], ...] | None  # navigate mode only; hold included
    ticks: int


def _observe(game: Game, env: CheeseEnv) -> Obs:
    active = game.active
    assert active is not None
    return Obs(
        rows=tuple(game.rows),
        active=active.type,
        hold=game.hold_type,
        can_hold=game.can_hold and game.cfg.hold_enabled,
        queue=tuple(game.queue[: env.previews]),
        cheese_on_board=game.cheese_on_board,
        cheese_dug=game.cheese_dug,
        goal=env.level,
        pieces_placed=game.pieces_placed,
        allow_180=env.allow_180,
    )


def run_episode(agent, env: CheeseEnv, seed: int, *, navigate: bool = True) -> EpisodeResult:
    """Play one seeded cheese episode with ``agent``.

    ``navigate=True`` replays each decision as real inputs through the
    engine; ``navigate=False`` applies placements directly (fast path —
    identical cheese outcomes, spin labels skipped). See the module docs.
    """

    game = Game(env.game_config(), seed=seed)
    first_piece = game.queue[0]
    decisions: list[Decision] = []
    inputs: list[list[Action]] | None = [] if navigate else None

    while not game.over:
        if game.active is None:  # ARE 0: an empty tick spawns the next piece
            game.tick()
            if game.over or game.active is None:
                break
        if game.pieces_placed >= env.piece_cap:
            return _episode_result(game, seed, first_piece, "capped", decisions, inputs)

        obs = _observe(game, env)
        decision = agent.decide(obs)
        piece = decision.placement.piece

        if decision.hold:
            if not (game.cfg.hold_enabled and game.can_hold):
                raise InvalidDecision("hold is not available here")
            expected = game.hold_type if game.hold_type is not None else game.queue[0]
            if piece is not expected:
                raise InvalidDecision(f"hold would bring {expected.name}, not {piece.name}")
            game.tick([Action.HOLD])
            if game.over:  # the held piece itself block out
                break
        else:
            if game.active is None or game.active.type is not piece:
                raise InvalidDecision(f"active piece is not {piece.name}")
        assert game.active is not None and game.active.type is piece

        placement = decision.placement
        placed_before = game.pieces_placed
        if navigate:
            path = find_path(list(game.rows), piece, placement, allow_180=env.allow_180)
            if path is None:
                raise InvalidDecision(f"placement not reachable: {placement}")
            for action in path:
                game.tick([action])
            assert game.pieces_placed == placed_before + 1, "path did not lock the piece"
            inputs.append([Action.HOLD, *path] if decision.hold else path)
        else:
            # fast path: teleport to the target rest and hard-drop it home.
            # Validated as resting so a floating target can never silently
            # diverge from the placement it claims to be.
            if B.collides(game.rows, piece, placement.rot, placement.x, placement.y) or not B.collides(
                game.rows, piece, placement.rot, placement.x, placement.y + 1
            ):
                raise InvalidDecision(f"placement is not a resting position: {placement}")
            active = game.active
            active.rot, active.x, active.y = placement.rot, placement.x, placement.y
            game.last_action = None  # plain arrival — no spin bookkeeping here
            game.tick([Action.HARD_DROP])
            assert game.pieces_placed == placed_before + 1, "hard drop did not lock the piece"
        decisions.append(decision)

    reason = "cleared" if game.won else "topout"
    return _episode_result(game, seed, first_piece, reason, decisions, inputs)


def _episode_result(
    game: Game, seed: int, first_piece: PieceType, reason: str,
    decisions: list[Decision], inputs: list[list[Action]] | None,
) -> EpisodeResult:
    return EpisodeResult(
        seed=seed,
        first_piece=first_piece,
        won=game.won,
        pieces=game.pieces_placed,
        dug=game.cheese_dug,
        reason=reason,
        decisions=tuple(decisions),
        inputs=tuple(tuple(p) for p in inputs) if inputs is not None else None,
        ticks=game.tick_count,
    )


# batch evaluation -------------------------------------------------------------


@dataclass(frozen=True)
class BatchResult:
    agent_name: str
    env: str  # env.desc at run time
    level: int
    seeds: tuple[int, ...]
    pieces: tuple[int, ...]  # per episode, failures included
    reasons: tuple[str, ...]

    @property
    def episodes(self) -> int:
        return len(self.seeds)

    @property
    def wins(self) -> int:
        return sum(1 for r in self.reasons if r == "cleared")

    @property
    def win_rate(self) -> float:
        return self.wins / self.episodes if self.episodes else 0.0

    def _win_pieces(self) -> list[int]:
        return [p for p, r in zip(self.pieces, self.reasons) if r == "cleared"]

    @property
    def mean_pieces(self) -> float | None:
        """Mean pieces over winning episodes (the cheese objective), or None
        when nothing was cleared."""
        wins = self._win_pieces()
        return fmean(wins) if wins else None

    @property
    def ci95_pieces(self) -> float | None:
        wins = self._win_pieces()
        if len(wins) < 2:
            return None
        return 1.96 * stdev(wins) / math.sqrt(len(wins))

    @property
    def mean_pieces_per_line(self) -> float | None:
        mean = self.mean_pieces
        return mean / self.level if mean is not None else None

    def summary(self) -> str:
        head = f"{self.agent_name} | cheese {self.env}"
        counts = Counter(self.reasons)
        fails = " | ".join(f"{k} {v}" for k, v in sorted(counts.items()) if k != "cleared") or "none"
        body = (
            f"episodes {self.episodes} | wins {self.wins} ({100 * self.win_rate:.1f}%)"
            f" | failures: {fails}"
        )
        mean, ci = self.mean_pieces, self.ci95_pieces
        if mean is None:
            stats = "no wins — no pieces-to-clear statistic"
        else:
            ci_txt = f" ± {ci:.2f}" if ci is not None else ""
            stats = (
                f"pieces to clear (wins): {mean:.2f}{ci_txt} (95% CI)"
                f" | pieces/line {self.mean_pieces_per_line:.3f}"
            )
        return f"{head}\n{body}\n{stats}"


def run_batch(
    agent, env: CheeseEnv, n_episodes: int, *, seed0: int = 0, navigate: bool = False
) -> BatchResult:
    """Run ``n_episodes`` seeded episodes (seeds ``seed0..seed0+n-1``) with a
    single agent instance, sequentially. ``navigate=False`` by default: batch
    evaluation is the fast path; the navigated mode is for validation."""
    results = [
        run_episode(agent, env, seed0 + i, navigate=navigate) for i in range(n_episodes)
    ]
    return BatchResult(
        agent_name=getattr(agent, "name", type(agent).__name__),
        env=env.desc,
        level=env.level,
        seeds=tuple(r.seed for r in results),
        pieces=tuple(r.pieces for r in results),
        reasons=tuple(r.reason for r in results),
    )


# board features (shared by agents and, later, search evals) --------------------


def apply_placement(rows: list[int], placement: Placement) -> tuple[list[int], int]:
    """Lock ``placement`` onto a copy of ``rows`` and clear: returns the new
    row list and the number of lines cleared (pure — no engine state)."""
    merged = list(rows)
    B.merge_piece(merged, placement.piece, placement.rot, placement.x, placement.y)
    cleared = B.full_rows(merged)
    if cleared:
        B.clear_rows(merged, cleared)
    return merged, len(cleared)


def count_holes(rows: list[int]) -> int:
    """Empty cells with at least one filled cell above them in the same
    column (covered wells included)."""
    holes = 0
    for x in range(FIELD_W):
        seen = False
        for i in range(FIELD_H):  # top -> bottom
            if rows[i] >> x & 1:
                seen = True
            elif seen:
                holes += 1
    return holes


def stack_height(rows: list[int]) -> int:
    """Occupied rows counted from the floor (0 on an empty board)."""
    for i, row in enumerate(rows):
        if row:
            return FIELD_H - i
    return 0
