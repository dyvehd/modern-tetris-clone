"""The advisor thread: fresh bot advice for the current piece, off the
game loop.

Every backend is too slow to call inline (MisaMino ~100 ms, our beam
~250 ms, cold-clear and fusion in between) — the game runs at 60 Hz. The
advisor owns a worker thread that re-thinks whenever the engine's "piece
token" changes (a token is the full decision identity: piece, board,
queue, hold — so every placement, hold, undo or restart triggers a
re-think) and publishes the latest advice keyed by that token. Stale
advice (the player placed before the bot answered) never shows against
the next piece: the consumer checks the token.

Each request carries a :meth:`Game.clone` snapshot, so the worker thinks
on an immutable state while the main thread keeps playing. The worker is
the only thread that touches a backend; switching models tears the worker
down and rebuilds it (backends keep state in their own libraries/processes
— a cold swap is the only clean transition).
"""

from __future__ import annotations

import threading

from ..engine.game import Game
from .backends import Backend, BotAdvice


def piece_token(game: Game) -> tuple:
    """Identity of the current decision point. Two states with the same
    token receive the same advice; a token change is the trigger to
    re-think. ``pieces_placed`` leading means undo/restart (same board)
    still counts as a new decision."""
    if game.active is None or game.over:
        return ()
    return (
        game.pieces_placed,
        game.active.type,
        tuple(game.rows),
        tuple(game.queue[:6]),
        game.hold_type,
        game.can_hold,
        game.cheese_dug,
    )


class Advisor:
    """Latest-advice cache for one backend, filled by a worker thread.

    ``request(game)`` from the main thread (cheap: clone + notify);
    ``advice_for(token)`` reads. When a newer request supersedes one still
    queued, the worker skips the stale one — only the latest token thinks.
    """

    def __init__(self, backend: Backend):
        self._backend = backend
        self._cv = threading.Condition()
        self._latest: BotAdvice | None = None
        self._latest_token: tuple | None = None  # token advice was computed for
        self._pending: tuple[tuple, Game] | None = None  # (token, snapshot)
        self._alive = True
        self._worker = threading.Thread(
            target=self._run, name=f"advisor-{backend.name}", daemon=True
        )
        self._worker.start()

    @property
    def backend_name(self) -> str:
        return self._backend.name

    def request(self, game: Game) -> None:
        """Ask for advice on the engine's current decision point. Cheap
        enough to call every logic tick: it is a no-op while the latest
        computed (or pending) token matches."""
        token = piece_token(game)
        if not token:
            return
        with self._cv:
            if token == self._latest_token:
                return
            if self._pending is not None and token == self._pending[0]:
                return
            self._pending = (token, game.clone())
            self._cv.notify_all()

    def advice_for(self, token: tuple) -> BotAdvice | None:
        """The advice for ``token`` if the worker finished it, else None."""
        with self._cv:
            if token == self._latest_token:
                return self._latest
            return None

    def invalidate(self) -> None:
        """Drop cached advice (restart, undo — worlds the advice no longer
        describes). The next request re-thinks from scratch."""
        with self._cv:
            self._latest = None
            self._latest_token = None
            self._pending = None

    def close(self) -> None:
        with self._cv:
            self._alive = False
            self._cv.notify_all()
        self._worker.join(timeout=3.0)
        self._backend.close()

    # ------------------------------------------------------------------ worker

    def _run(self) -> None:
        while True:
            with self._cv:
                while self._alive and self._pending is None:
                    self._cv.wait(timeout=0.25)
                if not self._alive:
                    self._backend.close()
                    return
                token, snapshot = self._pending
                self._pending = None
            try:
                advice = self._backend.think(snapshot)
            except Exception:
                advice = None  # a backend crash never takes the game down
            with self._cv:
                self._latest = advice
                self._latest_token = token
