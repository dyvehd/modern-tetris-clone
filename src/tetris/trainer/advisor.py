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

from ..engine.game import Action, Game
from ..ai.movegen import enumerate_placements
from .backends import Backend, BotAdvice, PlanStep, pin_best


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
        self._pending: tuple[tuple, Game, int] | None = None  # (token, snapshot, plan depth)
        self._alive = True
        # the plan (multi-move lookahead) computed for _plan_token, to the
        # realized depth _plan_target (0 = none computed yet)
        self._plan_token: tuple | None = None
        self._plan_steps: list[PlanStep] = []
        self._plan_depth_claimed = 0
        self._worker = threading.Thread(
            target=self._run, name=f"advisor-{backend.name}", daemon=True
        )
        self._worker.start()

    @property
    def backend_name(self) -> str:
        return self._backend.name

    def request(self, game: Game, depth: int = 1) -> None:
        """Ask for advice (+:meth:`compute_plan` ``depth`` moves of the
        bot's plan) on the engine's current decision point. Cheap enough
        to call every logic tick: it is a no-op while the latest computed
        (or pending) token matches and the plan depth is covered."""
        token = piece_token(game)
        if not token:
            return
        with self._cv:
            p = self._pending
            if p is not None:
                if token == p[0] and depth <= p[2]:
                    return
            elif token == self._latest_token:
                realized = (
                    self._plan_depth_claimed
                    if token == self._plan_token
                    else 0
                )
                if depth <= realized:
                    return
            self._pending = (token, game.clone(), max(1, depth))
            self._cv.notify_all()

    def advice_for(self, token: tuple) -> BotAdvice | None:
        """The advice for ``token`` if the worker finished it, else None."""
        with self._cv:
            if token == self._latest_token:
                return self._latest
            return None

    def plan_for(self, token: tuple) -> list[PlanStep]:
        """The bot's plan for ``token``: steps best-first (step 0 = the
        placement the bot will actually play; step k = its intended
        placement k placements later). Only covered by what has been
        requested — the consumer re-requests with a greater depth."""
        with self._cv:
            if token == self._plan_token:
                return self._plan_steps
            return []

    def invalidate(self) -> None:
        """Drop cached advice and any plan (restart, undo — worlds the
        advice no longer describes). The next request re-thinks."""
        with self._cv:
            self._latest = None
            self._latest_token = None
            self._pending = None
            self._plan_token = None
            self._plan_steps = []
            self._plan_depth_claimed = 0

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
                token, snapshot, depth = self._pending
                self._pending = None
                # claim the job NOW: request() no-ops while this plan is
                # in flight instead of re-queuing the re-think every tick
                self._plan_token = token
                self._plan_steps = []
                self._plan_depth_claimed = depth
            # publish advice first — it is what the player/automove needs;
            # the plan (a tree of re-thinks) may take several backends' worth
            # of think time after it
            try:
                advice = pin_best(self._backend.think(snapshot))
            except Exception:
                advice = None  # a backend crash never takes the game down
            with self._cv:
                self._latest = advice
                self._latest_token = token
            plan: list[PlanStep] = []
            if advice is not None and depth > 1:
                try:
                    plan = self._compute_plan(snapshot, advice, depth)
                except Exception:
                    plan = []
            with self._cv:
                self._plan_steps = plan
                self._plan_depth_claimed = min(depth, max(1, len(plan)))

    def _compute_plan(
        self, snapshot: Game, advice: BotAdvice, depth: int
    ) -> list[PlanStep]:
        """Simulate the bot playing its own advice and record each
        placement. A backend's own plan (Cold Clear) supplies its deeper
        steps for free; everywhere else each step is one think on the
        simulated successor game — an engine replay, so gravity, cheese
        refill, bag and hold semantics are the engine's, never approximated.
        Aborts when a newer decision is pending (the plan for it follows).
        """
        target = min(depth, 6)
        if target <= 1:
            return []
        steps = [PlanStep(advice.piece, tuple(sorted(advice.cells)), advice.hold)]
        supplied = self._backend.plan_steps(advice)
        supplied = list(supplied[1:]) if supplied is not None else None
        # every deeper step is decided on the board AFTER the bot's own
        # move has been played — the simulation starts there
        state = self._successor(snapshot, steps[0])
        if state is None:
            return steps
        for rank in range(1, target):
            with self._cv:
                if self._pending is not None:  # a newer decision arrived
                    return steps
            if supplied is not None and rank - 1 < len(supplied):
                step = supplied[rank - 1]
            else:
                if supplied is not None:
                    supplied = None  # plan exhausted: think from here on
                try:
                    nxt = pin_best(self._backend.think(state))
                except Exception:
                    return steps
                if nxt is None:
                    return steps
                step = PlanStep(nxt.piece, tuple(sorted(nxt.cells)), nxt.hold)
            successor = self._successor(state, step)
            if successor is None:
                return steps
            steps.append(step)
            state = successor
        return steps

    def _successor(self, state: Game, step: PlanStep) -> Game | None:
        """The engine state after the bot plays ``step`` on ``state``
        (navigate the simulated piece to the placement, hard-drop it)."""
        from ..ai.movegen import Placement  # noqa: F401  (type clarity)

        s = state.clone()
        if s.active is None or s.over:
            return None
        actions: list[Action] = []
        if step.hold:
            if not (s.cfg.hold_enabled and s.can_hold):
                return None
            brings = s.hold_type if s.hold_type is not None else (
                s.queue[0] if s.queue else None
            )
            if brings is not step.piece:
                return None
            actions.append(Action.HOLD)
        elif s.active.type is not step.piece:
            return None
        if actions:
            s.tick(actions)
            if s.active is None or s.over:
                return None
        placement = next(
            (
                pl
                for pl in enumerate_placements(list(s.rows), step.piece)
                if set(pl.cells) == set(step.cells)
            ),
            None,
        )
        if placement is None:
            return None
        s.active.rot = placement.rot
        s.active.x = placement.x
        s.active.y = placement.y
        s.tick([Action.HARD_DROP])
        if not s.over:
            s.tick()  # spawn the next piece (gravity handled by the engine)
        return s
