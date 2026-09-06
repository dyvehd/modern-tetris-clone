"""The Tetris game state machine.

This is the heart of the engine: a deterministic, fixed-timestep (60 Hz)
simulation of Guideline/Jstris rules with zero rendering dependencies.

One logic tick:
    1. spawn delay (ARE / line-clear delay) countdown, if configured > 0
    2. spawn (garbage rise -> optional IHS/IRS -> block-out check)
    3. input actions, in order: rotations, shifts, hard drop, hold
    4. gravity (with soft-drop factor), lock delay, lock, line clears

Key rule sources:
    - SRS:            https://harddrop.com/wiki/SRS
    - T-spin:         https://harddrop.com/wiki/T-Spin (3-corner + front-corner
                      mini rule + TST-kick upgrade, walls count as filled)
    - Lock delay:     500 ms, move reset, 15-move limit (Jstris/TETR.IO default),
                      counter resets when the piece reaches a new lowest row
    - Scoring/attack: https://harddrop.com/wiki/Jstris (see scoring.py)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import IntEnum

from . import board as B
from .constants import (
    FIELD_H,
    FIELD_W,
    KICKS_180,
    KICKS_I,
    KICKS_JLSTZ,
    PIECE_CELLS,
    PieceType,
    SPAWN_X,
    SPAWN_Y,
    T_BOX_CORNERS,
    T_FRONT_CORNERS,
    TICKS_PER_SEC,
    TSPIN_UPGRADE_KICKS,
    VISIBLE_TOP,
    ms_to_ticks,
)
from .rng import SevenBag
from .scoring import evaluate_clear


class Action(IntEnum):
    """Discrete inputs the engine understands (one logic tick each)."""

    NOOP = 0
    LEFT = 1
    RIGHT = 2
    ROT_CW = 3
    ROT_CCW = 4
    ROT_180 = 5
    SOFT_DROP = 6  # one tick of soft-drop gravity
    HARD_DROP = 7
    HOLD = 8


class Btn(IntEnum):
    """Virtual buttons; the ``held`` set passed to tick() is used for soft
    drop and (if the TGM-style flags are on) initial rotation/hold."""

    LEFT = 0
    RIGHT = 1
    SOFT = 2
    ROT_CW = 3
    ROT_CCW = 4
    ROT_180 = 5
    HARD = 6
    HOLD = 7


@dataclass
class GameConfig:
    lock_delay_ms: int = 500
    max_move_resets: int = 15  # -1 = infinite (Guideline "Infinity")
    are_ms: int = 0  # entry delay
    line_clear_delay_ms: int = 0  # Jstris/TETR.IO default: instant clears
    gravity_g: float = 1.0  # 1G = 60 rows/s = 1 row per tick
    gravity_curve: bool = False  # Guideline marathon curve (ignores gravity_g)
    soft_drop_factor: float = 5.0  # math.inf allowed (instant soft drop)
    hold_enabled: bool = True
    # TGM-style held-at-spawn inputs. Jstris/TETR.IO/Nullpomino/Techmino do
    # not have these (IRS/IHS exist only in the TGM series). Initial DAS, by
    # contrast, is universal — it lives in the input controller, not here.
    irs_enabled: bool = False  # initial rotation: rotate key held at spawn
    ihs_enabled: bool = False  # initial hold: hold key held at spawn
    garbage_delay_ms: int = 500
    garbage_cap_per_rise: int = 8
    garbage_cancel: bool = True
    goal_lines: int | None = None  # sprint/marathon goal; None = endless
    score_level_multiplier: bool = False  # Guideline marathon scoring style
    # Cheese (dig) race: the field starts with — and is topped back up to —
    # this many garbage rows after every piece placement. 0 = off. Jstris
    # keeps 9 cheese rows for every goal (10/18/100/∞); four-tris and
    # Techmino use 10. With a goal, the refill is capped at the lines still
    # needed, so the last clear finishes on an empty board (Techmino's
    # dig_100l rule).
    cheese_rows: int = 0
    # How many consecutive garbage rows share one hole before the hole moves
    # to a different column (four-tris' GARBAGE run-length pool; Jstris
    # cheese plays the same: mostly 1–2 row runs, some longer shafts).
    cheese_hole_runs: tuple[int, ...] = (1, 1, 2, 2, 4, 5)


@dataclass
class ActivePiece:
    type: PieceType
    rot: int
    x: int
    y: int


@dataclass
class GarbageBatch:
    rows: list[int]  # hole column per garbage row
    due_tick: int


@dataclass
class Game:
    cfg: GameConfig
    seed: int | None = None

    # field / pieces -----------------------------------------------------
    rows: list[int] = field(default_factory=lambda: [0] * FIELD_H, init=False)
    # Parallel per-cell style grid for rendering only (piece PieceType value,
    # -1 = empty, -2 = garbage). The AI-facing state is `rows` alone; AIs can
    # ignore this. Kept in sync by _lock/_apply_due_garbage.
    styles: list[list[int]] = field(init=False)
    bag: SevenBag = field(init=False)
    queue: list[PieceType] = field(default_factory=list, init=False)
    active: ActivePiece | None = field(default=None, init=False)
    hold_type: PieceType | None = field(default=None, init=False)
    can_hold: bool = field(default=True, init=False)

    # per-piece state ------------------------------------------------------
    lowest_y: int = field(default=0, init=False)
    move_resets: int = field(default=0, init=False)
    lock_timer: int = field(default=0, init=False)  # ticks grounded
    grounded: bool = field(default=False, init=False)
    gravity_accum: float = field(default=0.0, init=False)
    # last successful movement, for T-spin eligibility:
    #   None, "rotate" (with kick bookkeeping), "shift", "fall"
    last_action: str | None = field(default=None, init=False)
    last_rot_from: int = field(default=0, init=False)
    last_rot_to: int = field(default=0, init=False)
    last_kick_index: int = field(default=0, init=False)

    # spawn/ARE ------------------------------------------------------------
    spawn_delay: int = field(default=0, init=False)

    # garbage ---------------------------------------------------------------
    garbage_queue: list[GarbageBatch] = field(default_factory=list, init=False)
    _last_hole: int = field(default=-1, init=False)

    # cheese (dig) mode -------------------------------------------------------
    cheese_on_board: int = field(default=0, init=False)
    _cheese_hole: int = field(default=-1, init=False)
    _cheese_run: int = field(default=0, init=False)

    # stats -----------------------------------------------------------------
    tick_count: int = field(default=0, init=False)
    score: int = field(default=0, init=False)
    lines: int = field(default=0, init=False)
    level: int = field(default=1, init=False)
    combo: int = field(default=0, init=False)
    b2b_chain: int = field(default=0, init=False)
    b2b_best: int = field(default=0, init=False)
    combo_best: int = field(default=0, init=False)
    pieces_placed: int = field(default=0, init=False)
    attack_sent: int = field(default=0, init=False)
    garbage_received: int = field(default=0, init=False)
    tspins: int = field(default=0, init=False)
    quads: int = field(default=0, init=False)
    perfect_clears: int = field(default=0, init=False)

    over: bool = field(default=False, init=False)
    won: bool = field(default=False, init=False)
    events: list[dict] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.bag = SevenBag(self.seed)
        self.styles = [[-1] * FIELD_W for _ in range(FIELD_H)]
        self._refill_queue()
        self._cheese_refill()  # cheese modes start on a full cheese stack
        self.lock_delay_ticks = max(0, ms_to_ticks(self.cfg.lock_delay_ms))
        self.are_ticks = max(0, ms_to_ticks(self.cfg.are_ms))
        self.clear_delay_ticks = max(0, ms_to_ticks(self.cfg.line_clear_delay_ms))
        self.garbage_delay_ticks = max(0, ms_to_ticks(self.cfg.garbage_delay_ms))

    # ------------------------------------------------------------------ util

    def _refill_queue(self) -> None:
        while len(self.queue) < 7:
            self.queue.append(self.bag.next_piece())

    @property
    def seconds(self) -> float:
        return self.tick_count / TICKS_PER_SEC

    def clone(self) -> Game:
        import copy

        return copy.deepcopy(self)

    # ------------------------------------------------------------------ tick

    def tick(self, actions: list[Action] | tuple[Action, ...] = (), held: frozenset[Btn] = frozenset()) -> None:
        """Advance the simulation by one logic tick (1/60 s)."""
        if self.over:
            return
        self.events = []
        self.tick_count += 1

        if self.active is None:
            if self.spawn_delay > 0:
                self.spawn_delay -= 1
                if self.spawn_delay == 0:
                    self._spawn(held)
                return  # inputs are ignored during ARE/line delay
            self._spawn(held)
            if self.over:
                return

        soft = Btn.SOFT in held or Action.SOFT_DROP in actions
        for action in actions:
            if self.active is None:  # hard drop may have ended the game
                break
            self._apply_action(action, held)

        if self.active is not None:
            self._apply_gravity(soft, held)

    def _apply_action(self, action: Action, held: frozenset[Btn]) -> None:
        piece = self.active
        assert piece is not None
        if action is Action.ROT_CW:
            self._try_rotate(piece, 1)
        elif action is Action.ROT_CCW:
            self._try_rotate(piece, -1)
        elif action is Action.ROT_180:
            self._try_rotate(piece, 2)
        elif action is Action.LEFT:
            self._try_shift(piece, -1)
        elif action is Action.RIGHT:
            self._try_shift(piece, 1)
        elif action is Action.HARD_DROP:
            self._hard_drop(piece, held)
        elif action is Action.HOLD:
            self._do_hold(held)
        # Action.SOFT_DROP / NOOP: soft drop is handled via gravity below.

    # ------------------------------------------------------------- movement

    def _on_successful_move(self) -> None:
        """Move-reset: a successful shift/rotation while grounded restarts the
        lock timer, up to max_move_resets times per piece."""
        if self.grounded and (self.cfg.max_move_resets < 0 or self.move_resets < self.cfg.max_move_resets):
            self.lock_timer = 0
            self.move_resets += 1

    def _try_shift(self, piece: ActivePiece, dx: int) -> bool:
        if B.collides(self.rows, piece.type, piece.rot, piece.x + dx, piece.y):
            return False
        piece.x += dx
        self.last_action = "shift"
        self._on_successful_move()
        return True

    def _try_rotate(self, piece: ActivePiece, delta: int) -> bool:
        """Attempt an SRS rotation. Returns True if the piece moved."""
        if piece.type is PieceType.O:
            return False  # O rotation is a no-op: no kicks, no lock reset
        new_rot = (piece.rot + delta) % 4
        if delta == 2:
            table = KICKS_180
        elif piece.type is PieceType.I:
            table = KICKS_I
        else:
            table = KICKS_JLSTZ
        tests = table.get((piece.rot, new_rot), ((0, 0),))
        for i, (kx, ky) in enumerate(tests):
            nx, ny = piece.x + kx, piece.y - ky  # wiki kicks use +y = up
            if not B.collides(self.rows, piece.type, new_rot, nx, ny):
                self.last_rot_from = piece.rot
                piece.rot, piece.x, piece.y = new_rot, nx, ny
                self.last_rot_to = new_rot
                self.last_kick_index = i
                self.last_action = "rotate"
                self._on_successful_move()
                return True
        return False

    def _hard_drop(self, piece: ActivePiece, held: frozenset[Btn]) -> None:
        gy = B.ghost_y(self.rows, piece.type, piece.rot, piece.x, piece.y)
        dist = gy - piece.y
        piece.y = gy
        self.score += 2 * dist
        # Hard drop locks immediately and does NOT cancel T-spin eligibility
        # (modern games count hard-dropped spins; last movement stays "rotate").
        self._lock(held)

    # -------------------------------------------------------------- gravity

    def _apply_gravity(self, soft: bool, held: frozenset[Btn]) -> None:
        piece = self.active
        assert piece is not None
        g = self._current_gravity_g()
        sdf = self.cfg.soft_drop_factor
        if soft and math.isinf(sdf):
            distance = B.ghost_y(self.rows, piece.type, piece.rot, piece.x, piece.y) - piece.y
        else:
            rate = g * (sdf if soft else 1.0)
            self.gravity_accum += rate
            distance = int(self.gravity_accum)
        if distance > 0:
            self.gravity_accum = max(0.0, self.gravity_accum - distance)
            moved = 0
            for _ in range(distance):
                if B.collides(self.rows, piece.type, piece.rot, piece.x, piece.y + 1):
                    self.gravity_accum = 0.0
                    break
                piece.y += 1
                moved += 1
                self.last_action = "fall"  # gravity fall cancels T-spin eligibility
            if piece.y > self.lowest_y:
                # Reaching a new lowest row resets the move-reset counter.
                self.lowest_y = piece.y
                self.move_resets = 0
            if moved:
                self.grounded = False
                self.lock_timer = 0

        self.grounded = B.collides(self.rows, piece.type, piece.rot, piece.x, piece.y + 1)
        if self.grounded:
            self.lock_timer += 1
            if self.lock_timer >= self.lock_delay_ticks:
                self._lock(held)
        else:
            self.lock_timer = 0

    def _current_gravity_g(self) -> float:
        if not self.cfg.gravity_curve:
            return self.cfg.gravity_g
        # Tetris Worlds / Guideline marathon curve: seconds per row.
        lvl = min(self.level, 20)
        sec_per_row = max((0.8 - 0.007 * (lvl - 1)) ** (lvl - 1), 1 / (60 * 20))
        return 1.0 / (sec_per_row * TICKS_PER_SEC)

    # ----------------------------------------------------------------- lock

    def _detect_tspin(self, piece: ActivePiece) -> str:
        """Returns "none" | "mini" | "full" (Tetris-Friends ruleset, as Jstris)."""
        if piece.type is not PieceType.T or self.last_action != "rotate":
            return "none"
        cx, cy = piece.x, piece.y
        occupied = []
        for ox, oy in T_BOX_CORNERS:
            gx, gy = cx + ox, cy + oy
            occupied.append(
                gx < 0 or gx >= FIELD_W or gy < 0 or gy >= FIELD_H or (self.rows[gy] >> gx) & 1
            )
        total = sum(occupied)
        f1, f2 = T_FRONT_CORNERS[piece.rot]
        transition = (self.last_rot_from, self.last_rot_to)
        if total >= 3:
            if occupied[f1] and occupied[f2]:
                return "full"
            if self.last_kick_index in TSPIN_UPGRADE_KICKS.get(transition, frozenset()):
                return "full"  # TST kick upgrade disregards the corner split
            return "mini"
        if self.last_kick_index in TSPIN_UPGRADE_KICKS.get(transition, frozenset()):
            return "full"  # upgrade applies regardless of corner count
        return "none"

    def _lock(self, held: frozenset[Btn] = frozenset()) -> None:
        piece = self.active
        assert piece is not None
        tspin = self._detect_tspin(piece)
        B.merge_piece(self.rows, piece.type, piece.rot, piece.x, piece.y)
        for cx, cy in PIECE_CELLS[piece.type][piece.rot]:
            px = piece.x + cx
            if 0 <= px < FIELD_W:
                self.styles[piece.y + cy][px] = piece.type.value
        self.pieces_placed += 1
        if tspin != "none":
            self.tspins += 1

        # Lock out: the piece came to rest entirely above the visible field.
        cells = PIECE_CELLS[piece.type][piece.rot]
        lockout = all(piece.y + cy < VISIBLE_TOP for cx, cy in cells)

        cleared = B.full_rows(self.rows)
        n = len(cleared)
        if self.cfg.cheese_rows:
            # a cleared row that contained garbage counts as dug cheese
            for i in cleared:
                if any(v == -2 for v in self.styles[i]):
                    self.cheese_on_board -= 1
        if n:
            B.clear_rows(self.rows, cleared)
            # shift the style grid the same way
            drop = set(cleared)
            kept = [s for i, s in enumerate(self.styles) if i not in drop]
            self.styles[:] = [[-1] * FIELD_W] * (FIELD_H - len(kept)) + kept
            self.lines += n
            if n == 4:
                self.quads += 1
            self.combo += 1
            self.combo_best = max(self.combo_best, self.combo)
        else:
            self.combo = 0
        # A lock that clears no lines never affects the B2B chain
        # (including T-spins with 0 lines, per Guideline).

        pc = n > 0 and all(row == 0 for row in self.rows)
        if pc:
            self.perfect_clears += 1

        if n or tspin != "none":
            outcome = evaluate_clear(
                n,
                tspin,
                self.combo,
                self.b2b_chain,
                pc,
                level_multiplier=self.level if self.cfg.score_level_multiplier else 1,
            )
            if outcome.difficult:
                self.b2b_chain += 1
                self.b2b_best = max(self.b2b_best, self.b2b_chain)
            elif n > 0:
                self.b2b_chain = 0
            self.score += outcome.score
            self._send_attack(outcome.attack)
            if n:
                self.events.append(
                    {
                        "kind": "clear",
                        "lines": n,
                        "label": outcome.label,
                        "b2b": outcome.b2b_extended,
                        "pc": pc,
                        "attack": outcome.attack,
                        "combo": self.combo if self.combo >= 2 else 0,
                    }
                )
            else:
                self.events.append({"kind": "tspin", "label": outcome.label})

        self.level = 1 + self.lines // 10
        self.active = None
        if n and self.clear_delay_ticks > 0:
            self.spawn_delay = max(self.clear_delay_ticks, self.are_ticks)
        elif self.are_ticks > 0:
            self.spawn_delay = self.are_ticks
        else:
            self._spawn(held)  # ARE 0: next piece enters immediately

        if lockout:
            self._game_over(won=False)
        if self.cfg.goal_lines is not None and self.lines >= self.cfg.goal_lines:
            self._game_over(won=True)

    # ---------------------------------------------------------------- spawn

    def _send_attack(self, attack: int) -> None:
        """Outgoing attack cancels pending incoming garbage first (Jstris)."""
        if attack <= 0:
            return
        remaining = attack
        if self.cfg.garbage_cancel:
            while remaining > 0 and self.garbage_queue:
                batch = self.garbage_queue[0]
                if len(batch.rows) <= remaining:
                    remaining -= len(batch.rows)
                    self.garbage_queue.pop(0)
                else:
                    batch.rows = batch.rows[remaining:]
                    remaining = 0
        self.attack_sent += remaining

    def add_garbage(self, count: int) -> None:
        """Queue incoming garbage rows (from an opponent / trainer)."""
        if count <= 0 or self.over:
            return
        holes: list[int] = []
        for _ in range(count):
            if self._last_hole < 0 or self.bag.random() < 0.5:
                self._last_hole = self.bag.randrange(FIELD_W)
            holes.append(self._last_hole)
        self.garbage_queue.append(
            GarbageBatch(rows=holes, due_tick=self.tick_count + self.garbage_delay_ticks)
        )
        self.garbage_received += count

    def _apply_due_garbage(self) -> None:
        """Insert due garbage at the bottom, pushing the stack up."""
        due_now = [b for b in self.garbage_queue if b.due_tick <= self.tick_count]
        if not due_now:
            return
        rows_to_apply: list[int] = []
        cap = self.cfg.garbage_cap_per_rise
        for batch in due_now:
            while batch.rows and len(rows_to_apply) < cap:
                rows_to_apply.append(batch.rows.pop(0))
        self.garbage_queue = [b for b in self.garbage_queue if b.rows]
        if not rows_to_apply:
            return
        n = len(rows_to_apply)
        overflow = any(self.rows[:n])
        garbage_rows = [B.garbage_row(hole) for hole in rows_to_apply]
        # Push the stack up (toward index 0): the top n rows are pushed out
        # of the field, the garbage rows enter at the bottom.
        self.rows[:] = self.rows[n:] + garbage_rows
        garbage_styles = [[-2] * FIELD_W for _ in range(n)]
        self.styles[:] = self.styles[n:] + garbage_styles
        self.events.append({"kind": "garbage", "rows": n})
        if overflow:
            self._game_over(won=False)

    # --------------------------------------------------------------- cheese

    def _next_cheese_hole(self) -> int:
        """Hole column for the next cheese row. One hole per row; the hole
        stays in the same column for a run of consecutive rows, then moves
        to a different column (four-tris' scheme, run lengths drawn from
        ``cheese_hole_runs``)."""
        if self._cheese_run <= 0 or self._cheese_hole < 0:
            prev = self._cheese_hole
            hole = prev
            while hole == prev:
                hole = self.bag.randrange(FIELD_W)
            self._cheese_hole = hole
            runs = self.cfg.cheese_hole_runs or (1,)
            self._cheese_run = runs[self.bag.randrange(len(runs))]
        self._cheese_run -= 1
        return self._cheese_hole

    def _cheese_target(self) -> int:
        """Garbage rows the cheese stack should hold: always ``cheese_rows``
        without a goal (infinite cheese); with a goal, capped at the lines
        still needed so the final clear finishes on an empty board."""
        if self.cfg.goal_lines is None:
            return self.cfg.cheese_rows
        return min(self.cfg.cheese_rows, max(0, self.cfg.goal_lines - self.lines))

    def _cheese_refill(self) -> None:
        """Cheese (dig) mode: rise new garbage rows so the stack holds the
        target number of cheese rows. Called once per spawned piece (and at
        game start), so the stack only shrinks on a tick you actually clear."""
        if not self.cfg.cheese_rows or self.over:
            return
        count = self._cheese_target() - self.cheese_on_board
        if count <= 0:
            return
        overflow = any(self.rows[:count])
        for _ in range(count):
            self.rows[:] = self.rows[1:] + [B.garbage_row(self._next_cheese_hole())]
            self.styles[:] = self.styles[1:] + [[-2] * FIELD_W]
        self.cheese_on_board += count
        self.events.append({"kind": "garbage", "rows": count})
        if overflow:
            self._game_over(won=False)

    def _spawn(
        self,
        held: frozenset[Btn],
        forced_type: PieceType | None = None,
        apply_garbage: bool = True,
        reset_hold: bool = True,
    ) -> None:
        """Spawn the next piece (or a hold swap). Applies garbage first,
        then the optional TGM-style IHS/IRS, then the block-out check."""
        if apply_garbage and not self.over:
            self._apply_due_garbage()
            self._cheese_refill()  # cheese modes: top the stack back up
            if self.over:
                return

        piece_type = forced_type if forced_type is not None else self._take_from_queue()

        # IHS (TGM-style, off by default): hold button held at spawn swaps
        # the incoming piece.
        ihs = (
            self.cfg.ihs_enabled
            and self.cfg.hold_enabled
            and forced_type is None  # a hold swap cannot chain into itself
            and self.can_hold
            and Btn.HOLD in held
        )
        if ihs:
            previous = self.hold_type
            self.hold_type = piece_type
            piece_type = previous if previous is not None else self._take_from_queue()
            self.can_hold = False
            self.events.append({"kind": "hold"})

        piece = ActivePiece(piece_type, 0, SPAWN_X[piece_type], SPAWN_Y)

        # IRS (TGM-style, off by default): rotation button(s) held at spawn
        # rotate the piece as it enters.
        if self.cfg.irs_enabled:
            irs_delta = 0
            if Btn.ROT_180 in held:
                irs_delta = 2
            elif Btn.ROT_CW in held:
                irs_delta = 1
            elif Btn.ROT_CCW in held:
                irs_delta = -1
            if irs_delta and piece.type is not PieceType.O:
                saved = replace(piece)
                if not self._try_rotate(piece, irs_delta):
                    piece.rot, piece.x, piece.y = saved.rot, saved.x, saved.y

        if B.collides(self.rows, piece.type, piece.rot, piece.x, piece.y):
            # Block out: no room at the spawn location.
            self.active = piece
            self._game_over(won=False)
            return

        self.active = piece
        self.lowest_y = piece.y
        self.move_resets = 0
        self.lock_timer = 0
        self.grounded = False
        self.gravity_accum = 0.0
        self.last_action = None
        self.can_hold = reset_hold and not ihs

    def _take_from_queue(self) -> PieceType:
        self._refill_queue()
        return self.queue.pop(0)

    def _do_hold(self, held: frozenset[Btn]) -> None:
        if not self.cfg.hold_enabled or not self.can_hold or self.active is None:
            return
        current = self.active.type
        previous = self.hold_type
        self.hold_type = current
        self.can_hold = False
        self.active = None
        self.events.append({"kind": "hold"})
        # Jstris: hold swaps spawn instantly (no ARE), no garbage rise.
        self._spawn(held, forced_type=previous, apply_garbage=False, reset_hold=False)

    # ----------------------------------------------------------------- end

    def _game_over(self, won: bool) -> None:
        if self.over:
            return
        self.over = True
        self.won = won
        self.active = None
        self.events.append({"kind": "topout" if not won else "finish"})

    # ------------------------------------------------------- test/AI helpers

    def spawn_forced(self, piece_type: PieceType) -> None:
        """Spawn a specific piece next (testing / puzzle setup)."""
        self.spawn_delay = 0
        self.active = None
        self._spawn(frozenset(), forced_type=piece_type)

    def set_rows(self, rows: list[int]) -> None:
        self.rows[:] = rows
        self.styles = [[-1] * FIELD_W for _ in range(FIELD_H)]
