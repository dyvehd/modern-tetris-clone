"""7-bag randomizer (Guideline)."""

from __future__ import annotations

import random

from .constants import ALL_PIECES, PieceType


class SevenBag:
    """Deals pieces in shuffled bags of all seven tetrominoes.

    Every window of 7 consecutive deals contains each piece exactly once.
    Fully deterministic for a given seed; state is copyable for AI search.
    """

    __slots__ = ("_rng", "_bag")

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._bag: list[PieceType] = []

    def next_piece(self) -> PieceType:
        if not self._bag:
            self._bag = list(ALL_PIECES)
            self._rng.shuffle(self._bag)
        return self._bag.pop()

    def random(self) -> float:
        return self._rng.random()

    def randrange(self, n: int) -> int:
        return self._rng.randrange(n)

    def peek(self, n: int) -> list[PieceType]:
        """Return the next ``n`` pieces without consuming them."""
        import random as _random

        bag = list(self._bag)
        rng = _random.Random()
        rng.setstate(self._rng.getstate())
        out: list[PieceType] = []
        while len(out) < n:
            if not bag:
                bag = list(ALL_PIECES)
                rng.shuffle(bag)
            out.append(bag.pop())
        return out

    def __deepcopy__(self, memo: dict) -> SevenBag:
        import random as _random

        clone = SevenBag.__new__(SevenBag)
        clone._bag = list(self._bag)
        clone._rng = _random.Random()
        clone._rng.setstate(self._rng.getstate())
        memo[id(self)] = clone
        return clone
