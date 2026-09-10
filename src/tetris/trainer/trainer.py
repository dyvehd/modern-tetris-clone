"""Trainer state machine: everything the 100L cheese trainer mode adds on
top of a running :class:`Game`.

Owned by the app; :meth:`tick` runs once per logic tick from the main
thread, after ``game.tick``. The heavy lifting (bot thinking) lives on the
advisor thread — this module only consumes completed advice.

Features:
- **AI shadows** — the bot's ranked placements drawn as ghost outlines on
  the board: the top move full-strength, the following ``lookahead``
  candidates fading by rank. Visually distinct from the player's own
  ghost (corner-tick outlines vs the solid border).
- **automove** — the bot plays: paced at a fixed PPS, or one move per
  hard-drop key press ("step mode"). Moves are ANIMATED — the path's
  inputs replay one per tick through the real engine, so you watch the
  piece navigate, exactly like the cheese harness's navigate=True mode.
- **live feedback** — with shadows off: after the player locks a piece,
  compare against the bot's placement for that decision point; a
  differing placement is auto-undone (zen-undo puts the piece back in
  their hands) and the annotation flashes.
- **annotation** — every player placement with advice available is ranked
  among the bot's scored candidates (chess-style labels + running
  accuracy; :mod:`tetris.trainer.annotation`).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..ai.movegen import Placement, enumerate_placements
from ..ai.pathfinder import find_path
from ..engine.constants import PieceType
from ..engine.game import Action, Game
from .advisor import Advisor, piece_token
from .annotation import AccuracyStats, MoveQuality, annotate
from .backends import BotAdvice, make_backend


@dataclass
class TrainerConfig:
    """User-facing knobs, all changeable live (see the app's keybinds)."""

    backend: str = "cheese-beam"
    ai_on: bool = True
    shadows_on: bool = True
    lookahead: int = 3  # extra ranked placements drawn after the top move
    live_feedback: bool = False
    automove: bool = False
    automove_pps: float = 2.0
    step_mode: bool = False  # one bot move per hard-drop key press


class Trainer:
    """Trainer state for one running game.

    The app calls :meth:`tick` after every ``game.tick``, passing the
    engine's tick events. The engine's "lock" event (absolute placement
    cells — added for the trainer) drives annotation and live feedback;
    ``self._token`` (the decision point tracked while the piece was live)
    is the key the advice is looked up under.
    """

    def __init__(self, config: TrainerConfig):
        self.cfg = config
        self.advisor: Advisor | None = None
        self.stats = AccuracyStats()
        self.last_quality: MoveQuality | None = None
        self.last_advice: BotAdvice | None = None
        self._token: tuple = ()  # decision point last_advice answers
        # automove state: the pending input path and how far we are
        self._auto_queue: list[Action] = []
        self._auto_accum = 0.0  # ticks since the last move started
        self._stepped = False  # step mode: a press is waiting
        # undo hook (the app owns the zen undo stack): () -> bool
        self._undo_hook = None
        # a player lock whose advice had not arrived yet
        self._pending_token: tuple = ()
        self._pending_cells: tuple[tuple[int, int], ...] = ()
        self._pending_hold: bool = False
        self._player_held = False
        self.set_backend(config.backend)

    # -------------------------------------------------------------- lifecycle

    def set_backend(self, name: str) -> None:
        """Switch bot model (live). Cold swap: the old worker thread is
        torn down, a new one starts — backends keep global state."""
        if self.advisor is not None:
            self.advisor.close()
        self.cfg.backend = name
        self.advisor = Advisor(make_backend(name))
        self.last_advice = None
        self._token = ()  # force a fresh request on the next tick
        self._auto_queue = []
        self._pending_token = ()

    def close(self) -> None:
        if self.advisor is not None:
            self.advisor.close()
            self.advisor = None

    def on_restart(self) -> None:
        """New game: the old world's advice is void."""
        self.stats = AccuracyStats()
        self.last_quality = None
        self.last_advice = None
        self._token = ()
        self._auto_queue = []
        self._auto_accum = 0.0
        self._stepped = False
        self._pending_token = ()
        if self.advisor is not None:
            self.advisor.invalidate()

    def on_undo(self) -> None:
        """An undo restored an earlier state: cancel any in-flight bot
        move; the advice cache is still valid (tokens are content-keyed)."""
        self._auto_queue = []
        self._pending_token = ()

    def set_undo_hook(self, hook) -> None:
        self._undo_hook = hook

    # ------------------------------------------------------------------- tick

    def tick(self, game: Game, events: list[dict]) -> list[dict]:
        """One logic tick of trainer logic, AFTER ``game.tick``.

        Returns extra engine events produced by automove's own ticks (the
        app feeds them to its popup/finish handling).
        """
        if self.advisor is None:
            return []
        extra: list[dict] = []

        # 1. player placements: annotate, maybe auto-undo (live feedback)
        for ev in events:
            if ev.get("kind") == "lock":
                self._on_player_lock(ev)
            elif ev.get("kind") == "hold":
                self._player_held = True

        # 2. keep advice fresh for the current decision point
        token = piece_token(game)
        if token:
            if token != self._token:
                self._token = token
                self.last_advice = None
                self.advisor.request(game)
            elif self.last_advice is None:
                fresh = self.advisor.advice_for(token)
                if fresh is not None:
                    self.last_advice = fresh

        # 3. late advice for a pending lock (the bot answered after the
        # player had already dropped)
        if self._pending_token and self._pending_token != self._token:
            adv = self.advisor.advice_for(self._pending_token)
            if adv is not None:
                self._resolve_lock(adv)

        # 4. automove
        if self.cfg.automove and not game.over:
            extra.extend(self._do_automove(game))

        return extra

    # ------------------------------------------------------------ lock events

    def _on_player_lock(self, ev: dict) -> None:
        cells = tuple(ev["cells"])
        hold = self._player_held
        self._player_held = False
        token = self._token  # the decision point the placement answers
        advice = self.advisor.advice_for(token) if self.advisor else None
        if advice is not None:
            self._pending_token = token
            self._pending_cells = cells
            self._pending_hold = hold
            self._resolve_lock(advice)
        else:
            # advice still computing: keep the lock pending; resolved in
            # step 3 of the next ticks (or dropped on the next lock)
            self._pending_token = token
            self._pending_cells = cells
            self._pending_hold = hold

    def _resolve_lock(self, advice: BotAdvice) -> None:
        """Judge the pending player placement against ``advice``."""
        q = annotate(advice, self._pending_cells, self._pending_hold)
        if q is not None:
            self.last_quality = q
            self.stats.record(q)
        placed = tuple(sorted(self._pending_cells))
        wrong = tuple(sorted(advice.cells)) != placed
        self._pending_token = ()
        if self.cfg.live_feedback and wrong and self._undo_hook is not None:
            self._undo_hook()  # the piece returns to the player's hands

    # --------------------------------------------------------------- automove

    def request_step(self) -> None:
        """Step mode: the hard-drop key was pressed — one bot move."""
        self._stepped = True

    def _do_automove(self, game: Game) -> list[dict]:
        extra: list[dict] = []
        if self._auto_queue:
            # animate: one input per tick through the real engine
            action = self._auto_queue.pop(0)
            game.tick([action])
            extra = list(game.events)
            if game.over:
                self._auto_queue = []
            return extra

        if not self.cfg.ai_on or self.last_advice is None or not self._token:
            # waiting for advice still charges the PPS clock, so the
            # target rate holds once moves chain (advice latency does not
            # compound into the pacing)
            self._auto_accum += 1 / 60.0
            return []
        if self.cfg.step_mode:
            if not self._stepped:
                return []
            self._stepped = False
        else:
            self._auto_accum += 1 / 60.0
            if self._auto_accum < 1.0 / max(self.cfg.automove_pps, 0.05):
                return []
        path = self._path_for(game, self.last_advice)
        self._auto_accum = 0.0
        if not path:
            return []
        self._auto_queue = list(path)
        return []

    def _path_for(self, game: Game, advice: BotAdvice) -> list[Action]:
        """Advice placement -> input sequence ending HARD_DROP (our
        pathfinder). Empty when unreachable (advice outside our movement
        model — never silently replaced with another move)."""
        if game.active is None or game.over:
            return []
        rows = list(game.rows)
        if advice.hold:
            if not (game.cfg.hold_enabled and game.can_hold):
                return []
            brings = game.hold_type if game.hold_type is not None else (
                game.queue[0] if game.queue else None
            )
            if brings is not advice.piece:
                return []
            hold = [Action.HOLD]
        else:
            if game.active.type is not advice.piece:
                return []
            hold = []
        placements = enumerate_placements(rows, advice.piece)
        target = set(advice.cells)
        p = next((pl for pl in placements if set(pl.cells) == target), None)
        if p is None:
            return []
        path = find_path(rows, advice.piece, p)
        if path is None:
            return []
        return hold + list(path)

    # --------------------------------------------------------------- shadows

    def shadow_placements(self, game: Game) -> list[tuple[Placement, int, bool]]:
        """The renderer's shadow list: ``(placement, rank, hold)``.

        Rank 0 is the bot's top move (full-strength outline), rank
        1..lookahead the next ranked candidates (fading by rank). Every
        shadow is a real :class:`Placement` from OUR movegen — what you
        see is a placement the bot's choice navigates to in our movement
        model. The list is empty unless advice for the CURRENT decision
        point is ready."""
        if not self.cfg.ai_on or not self.cfg.shadows_on:
            return []
        if game.active is None or game.over or self.last_advice is None:
            return []
        if piece_token(game) != self._token:
            return []
        advice = self.last_advice
        # the piece each candidate places (hold-consistent with the engine)
        can_hold = game.cfg.hold_enabled and game.can_hold
        brings = None
        if can_hold:
            brings = game.hold_type if game.hold_type is not None else (
                game.queue[0] if game.queue else None
            )
        out: list[tuple[Placement, int, bool]] = []
        seen: set = set()
        limit = 1 + max(0, self.cfg.lookahead)
        for rank, cand in enumerate(advice.candidates[:limit]):
            piece = cand.piece
            hold = cand.hold
            if hold:
                if not can_hold or brings is not cand.piece:
                    continue
            elif game.active.type is not cand.piece:
                continue
            placements = enumerate_placements(list(game.rows), piece)
            p = next((pl for pl in placements if set(pl.cells) == set(cand.cells)), None)
            if p is None or p.key in seen:
                continue
            seen.add(p.key)
            out.append((p, rank, hold))
        return out

    # ------------------------------------------------------------------ HUD

    def hud_lines(self) -> list[tuple[str, str]]:
        """Status rows for the renderer's trainer panel."""
        name = self.cfg.backend
        mode = "off" if not self.cfg.ai_on else (
            "automove" if self.cfg.automove else ("hints" if self.cfg.shadows_on else "watching")
        )
        if self.cfg.automove and self.cfg.step_mode:
            mode = "automove·step"
        rows = [
            ("AI", mode),
            ("MODEL", name),
            ("SHADOWS", f"{1 + max(0, self.cfg.lookahead)}" if self.cfg.shadows_on else "off"),
            ("FEEDBACK", "on" if self.cfg.live_feedback else "off"),
        ]
        if self.cfg.automove:
            rows.append(("PPS", f"{self.cfg.automove_pps:g}"))
        if self.stats.n:
            q = self.last_quality
            last = f"{q.label} ({q.rank + 1}/{q.n_candidates})" if q else "-"
            rows.append(("LAST MOVE", last))
            rows.append(("ACCURACY", f"{self.stats.accuracy:.2f}z / {self.stats.best_rate * 100:.0f}% top"))
        return rows
