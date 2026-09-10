"""Chess-style move annotation for the trainer.

After the player locks a piece, we look their placement up among the
bot's scored candidates and express the quality as:

- ``rank`` — 0 = the bot's top move, 1 = second best, ... "12/38".
- ``loss`` — (best score − player's score) in the backend's native
  units, normalized per-move by the candidate spread so the numbers are
  comparable across backends and positions: ``sigma`` is the standard
  deviation of the candidate scores, and the reported loss is
  (best − player) / sigma (a z-gap). A z-gap of 0 = the top move;
  0.2 = practically equal; 1.0 = a full distribution-width worse.

Labels (deliberately simple, the chess brilliant/…/blunder ladder):

    z < 0.25   "best"     the bot's own choice (or an exact tie)
    z < 1.0    "good"     within the near-tie band
    z < 2.5    "inaccuracy"
    z < 4.0    "mistake"
    else       "blunder"

The candidate list is the backend's own root scoring of its whole action
space (MisaMino: every placement + hold-swap re-scored like its search;
our beam: ``candidate_moves(obs)`` rescored by the eval + win bonus), so
"rank k of n" is a real statement about how many legal alternatives the
bot liked more.

Also tracks the running "accuracy" — mean of min(z, 4) per placed piece
(the capped mean keeps one blunder from erasing a good run) and the
best-move rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean, pstdev

from .backends import BotAdvice, BotCandidate

LABEL_BEST = "best"
LABEL_GOOD = "good"
LABEL_INACCURACY = "inaccuracy"
LABEL_MISTAKE = "mistake"
LABEL_BLUNDER = "blunder"

# z-gap thresholds (see module docs)
Z_BEST = 0.25
Z_GOOD = 1.0
Z_INACCURACY = 2.5
Z_MISTAKE = 4.0
_CAP = 4.0


def label_for(z: float) -> str:
    if z < Z_BEST:
        return LABEL_BEST
    if z < Z_GOOD:
        return LABEL_GOOD
    if z < Z_INACCURACY:
        return LABEL_INACCURACY
    if z < Z_MISTAKE:
        return LABEL_MISTAKE
    return LABEL_BLUNDER


@dataclass(frozen=True)
class MoveQuality:
    """The annotation of one player placement."""

    placed_cells: tuple[tuple[int, int], ...]
    matched: bool  # the placement was among the scored candidates
    hold: bool
    rank: int  # 0-based among the candidates (best-first)
    n_candidates: int
    z_gap: float
    label: str
    best_cells: tuple[tuple[int, int], ...]  # the bot's top move
    best_hold: bool


def annotate(
    advice: BotAdvice,
    placed_cells: tuple[tuple[int, int], ...],
    placed_hold: bool,
) -> MoveQuality | None:
    """Rank ``placed_cells`` (the player's just-locked placement) among
    the bot's scored candidates. Returns None when the bot's candidate
    list is missing (a backend that only reports its top move cannot
    annotate) or the placement isn't in it (an unreachable-for-the-bot
    placement — the player outweirded the bot's movement model)."""
    if not advice.candidates:
        return None
    target = tuple(sorted(placed_cells))
    same_piece = [
        c for c in advice.candidates if c.cells == target and c.hold == placed_hold
    ]
    if not same_piece:
        # hold-agnostic fallback: the placement is the placement
        same_piece = [c for c in advice.candidates if c.cells == target]
        if not same_piece:
            return None
    placed = same_piece[0]
    best = advice.candidates[0]
    scores = [c.score for c in advice.candidates]
    sigma = pstdev(scores) if len(scores) > 1 else 0.0
    gap = best.score - placed.score
    if sigma <= 1e-9:
        z = 0.0 if gap <= 1e-9 else float("inf")
    else:
        z = gap / sigma
    rank = advice.candidates.index(placed)
    return MoveQuality(
        placed_cells=target,
        matched=True,
        hold=placed_hold,
        rank=rank,
        n_candidates=len(advice.candidates),
        z_gap=z,
        label=label_for(z),
        best_cells=best.cells,
        best_hold=best.hold,
    )


@dataclass
class AccuracyStats:
    """Running move-quality statistics for the session."""

    n: int = 0
    best_moves: int = 0
    _zs: list[float] = field(default_factory=list, repr=False)

    def record(self, q: MoveQuality) -> None:
        self.n += 1
        if q.rank == 0:
            self.best_moves += 1
        self._zs.append(min(q.z_gap, _CAP))

    @property
    def accuracy(self) -> float:
        """Capped mean z-gap — lower is better; 0 = always the top move."""
        return fmean(self._zs) if self._zs else 0.0

    @property
    def best_rate(self) -> float:
        return self.best_moves / self.n if self.n else 0.0

    def summary(self) -> str:
        if not self.n:
            return "no placements yet"
        return (
            f"accuracy {self.accuracy:.2f}z | top-move {self.best_rate * 100:.0f}%"
            f" | {self.n} placements"
        )
