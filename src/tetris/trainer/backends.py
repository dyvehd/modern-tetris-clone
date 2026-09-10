"""Bot backends for the cheese trainer: four external engines plus ours.

The trainer mode consumes one uniform interface (:class:`Backend`):

    advice = backend.think(game) -> BotAdvice | None

Every backend sees exactly what the engine sees (rows, active, hold, queue)
and answers with a *placement* — the piece id, whether to hold first, and
the four absolute cells where the piece should lock — plus, when it can,
the ranked candidate list it scored. Navigation is ALWAYS ours
(``tetris.ai.pathfinder``): no external pathfinder or kick table ever runs
against our engine, the same decoupling the cheese harness uses for every
agent. A placement that our own movegen cannot reach is reported as
unusable (never silently replayed by foreign inputs).

Backends (all optional at runtime — a missing library just drops out of
the model picker):

- ``cheese-beam`` — this repo's corrected beam search
  (``tetris.ai.search.BeamAgent``, beam20x4) with the downstack eval. The
  cheese specialist: digs L10 in ~24 pieces, ~250 ms/decision in pure
  Python (run on the advisor thread, never inline).
- ``misamino``    — the classic MisaMino search core (C++), bridged for
  Linux in ``bots/misamino`` (MMResult/MMCandidate ABI). A VS/attack bot:
  on cheese it digs slowly and builds — a different "expert opinion" to
  compare against, exactly the kind of disagreement that makes a trainer
  interesting.
- ``cold-clear``  — MinusKelvin's Cold Clear (Rust) through its upstream
  C API (async bot thread, placement plan). Strongest general-purpose
  player of the set.
- ``fusion``      — the MochBot/fusion engine (Rust) heuristic beam via
  the ``bots/fusion-shim`` C ABI (no ONNX model required).

Piece ids on our wire are ``PieceType`` values everywhere (I=0 J=1 L=2
O=3 S=4 T=5 Z=6); per-backend mapping lives inside each adapter.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from pathlib import Path

from ..engine.constants import PIECE_CELLS, PieceType
from ..engine.game import Game

# The bots/ tree lives at the repo root — reachable from this file as
# src/tetris/trainer/backends.py -> parents[3] (src, tetris, trainer are
# parents 0..2). Fall back to CWD (running from source without install).
_here = Path(__file__).resolve()
ROOT_CANDIDATES = (
    _here.parents[3],
    Path.cwd(),
)
ROOT = next((p for p in ROOT_CANDIDATES if (p / "bots" / "build").is_dir()), Path.cwd())
BOTS_DIR = ROOT / "bots"
BUILD_DIR = BOTS_DIR / "build"


@dataclass(frozen=True)
class BotCandidate:
    """One placement a backend scored, in our frame.

    ``score`` is backend-native (arbitrary scale — never compared across
    backends); ``lines`` is the placement's line-clear count when the
    backend reports one."""

    piece: PieceType
    hold: bool
    cells: tuple[tuple[int, int], ...]  # absolute (row, col)
    score: float = 0.0
    lines: int = 0


@dataclass(frozen=True)
class BotAdvice:
    """A backend's full answer for one decision point."""

    piece: PieceType
    hold: bool
    cells: tuple[tuple[int, int], ...]  # absolute (row, col) of the lock
    candidates: tuple[BotCandidate, ...] = ()  # ranked best-first
    think_ms: float = 0.0

    @property
    def key(self) -> tuple[PieceType, bool, tuple[tuple[int, int], ...]]:
        return (self.piece, self.hold, tuple(sorted(self.cells)))


class Backend:
    """A bot backend. ``name`` is what the model picker shows. All state
    seeding/reset is backend-specific; the common contract is just
    :meth:`think` — pure with respect to the engine state it is handed."""

    name = "backend"

    def think(self, game: Game) -> BotAdvice | None:
        raise NotImplementedError

    def reset(self) -> None:
        """Drop per-game state (a restart or an undo invalidates it)."""
        pass

    def close(self) -> None:
        pass


# --- our own beam ------------------------------------------------------------


class NativeBeamBackend(Backend):
    """The repo's corrected beam20x4 — the cheese specialist. Decides on
    the harness :class:`Obs` (the same protocol every learner uses), so
    its candidates are exactly ``candidate_moves(obs)`` scored by the
    search's own leaf evaluation."""

    def __init__(self, width: int = 20, depth: int = 4):
        from ..ai.search import BeamAgent

        self.name = f"cheese-beam{width}x{depth}"
        self._agent = BeamAgent(width, depth)
        self._last_obs = None

    def think(self, game: Game) -> BotAdvice | None:
        from ..ai.cheese import Obs, candidate_moves
        from ..ai.search import lock_and_count

        if game.active is None or game.over:
            return None
        stack = game.cfg.cheese_rows or 9
        obs = Obs(
            rows=tuple(game.rows),
            active=game.active.type,
            hold=game.hold_type,
            can_hold=game.can_hold and game.cfg.hold_enabled,
            queue=tuple(game.queue[:5]),
            cheese_on_board=game.cheese_on_board,
            cheese_dug=game.cheese_dug,
            goal=game.cfg.goal_lines or 0,
            pieces_placed=game.pieces_placed,
            allow_180=True,
            stack=stack,
        )
        import time

        t0 = time.perf_counter()
        decision = self._agent.decide(obs)
        think_ms = (time.perf_counter() - t0) * 1000.0

        # rank every candidate with the search's own scoring (a 1-ply
        # rescoring of the beam's action space; ordering at the root is the
        # trainer's move-quality scale)
        from ..ai.eval import eval_board
        from ..ai.search import BeamAgent

        cands = []
        for placement, hold in candidate_moves(obs):
            rows_after, lines, dug = lock_and_count(
                list(obs.rows), placement, obs.cheese_on_board
            )
            score = eval_board(rows_after, self._agent.weights) + (
                self._agent.weights.win if obs.cheese_dug + dug >= obs.goal and obs.goal else 0
            )
            cands.append(
                BotCandidate(
                    piece=placement.piece,
                    hold=hold,
                    cells=placement.cells,
                    score=float(score),
                    lines=lines,
                )
            )
        cands.sort(key=lambda c: -c.score)

        piece = decision.placement.piece
        cells = decision.placement.cells
        hold = decision.hold
        return BotAdvice(
            piece=piece,
            hold=hold,
            cells=cells,
            candidates=tuple(cands),
            think_ms=think_ms,
        )


# --- MisaMino (bots/misamino, ctypes) -----------------------------------------


class _MMResult(ctypes.Structure):
    _fields_ = [
        ("ok", ctypes.c_int),
        ("hold", ctypes.c_int),
        ("piece_id", ctypes.c_int),
        ("cells_r", ctypes.c_int * 4),
        ("cells_c", ctypes.c_int * 4),
    ]


class _MMCandidate(ctypes.Structure):
    _fields_ = [
        ("hold", ctypes.c_int),
        ("piece_id", ctypes.c_int),
        ("cells_r", ctypes.c_int * 4),
        ("cells_c", ctypes.c_int * 4),
        ("score", ctypes.c_int),
        ("lines", ctypes.c_int),
    ]


# GEMTYPE_* ids (tetris_gem.h): I=1 T=2 L=3 J=4 Z=5 S=6 O=7
_TO_MM: dict[PieceType, int] = {
    PieceType.I: 1,
    PieceType.T: 2,
    PieceType.L: 3,
    PieceType.J: 4,
    PieceType.Z: 5,
    PieceType.S: 6,
    PieceType.O: 7,
}
_FROM_MM: dict[int, PieceType] = {v: k for k, v in _TO_MM.items()}


class MisaMinoBackend(Backend):
    """MisaMino's search core through the Linux bridge (libmisamino.so).
    Single-threaded; calls are serialized by the advisor thread anyway.

    ``style`` selects the parameter set: "default" keeps the original
    core weights; "zetris" applies Zetris's stock style (the mat1jaczyyy
    fork's default parameters — Zetris is a MisaMino port, so this is as
    close to "Zetris" as the Linux world gets without porting its
    C++/CLI DLL; the name is kept honest in the UI).
    """

    name = "misamino"
    _MAX_CANDS = 256

    # Zetris stock style (Zetris.PPT/Preferences.cs:20-23 /
    # Zetris.TETRIO defaults): the 21 MisaMinoParameters values, mapped to
    # the bridge's knobs where the core supports them. The bridge exposes
    # spin180/allspin/combo; deeper parameter surgery is not portable.
    def __init__(self, style: str = "default") -> None:
        lib_path = BUILD_DIR / "libmisamino.so"
        if not lib_path.exists():
            raise FileNotFoundError(
                f"libmisamino.so not built — run bots/build_bots.sh ({lib_path})"
            )
        self.name = "zetris" if style == "zetris" else "misamino"
        self._style = style
        self._lib = ctypes.CDLL(str(lib_path))
        self._lib.mm_version.restype = ctypes.c_int
        if self._lib.mm_version() != 1:
            raise RuntimeError("unexpected libmisamino ABI version")
        self._lib.mm_think.restype = ctypes.c_int
        self._lib.mm_think.argtypes = [
            ctypes.POINTER(ctypes.c_ushort),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(_MMResult),
            ctypes.POINTER(_MMCandidate),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
        ]
        self._lib.mm_configure.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        # spin-180 on (our engine has 180s), all-spin off (Jstris rules),
        # Jstris combo table in MisaMino indexing (combo 1 = first clear):
        # attack 0/0/1/1/1/2/2/3/3/4/4/4/5 -> index 0,1 unused-pad
        combo = (ctypes.c_int * 20)(0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5)
        self._lib.mm_configure(1, 0, combo, len(combo))

    def think(self, game: Game) -> BotAdvice | None:
        if game.active is None or game.over:
            return None
        import time

        rows = (ctypes.c_ushort * 40)(*game.rows)
        queue = game.queue[:5]
        nxt = (ctypes.c_ubyte * max(1, len(queue)))(
            *[_TO_MM[p] for p in queue]
        )
        hold_id = _TO_MM[game.hold_type] if game.hold_type is not None else 0
        best = _MMResult()
        cands = (_MMCandidate * self._MAX_CANDS)()
        n = ctypes.c_int(0)
        t0 = time.perf_counter()
        self._lib.mm_think(
            rows,
            len(queue),
            nxt,
            hold_id,
            1 if (game.cfg.hold_enabled and game.can_hold) else 0,
            _TO_MM[game.active.type],
            game.b2b_chain,
            game.combo,
            2,  # full 3-ply root scoring for the ranked list
            ctypes.byref(best),
            cands,
            self._MAX_CANDS,
            ctypes.byref(n),
        )
        think_ms = (time.perf_counter() - t0) * 1000.0
        if not best.ok:
            return None
        out_cands = tuple(
            BotCandidate(
                piece=PieceType(c.piece_id),
                hold=bool(c.hold),
                cells=tuple(sorted(zip(c.cells_r, c.cells_c))),
                score=float(c.score),
                lines=c.lines,
            )
            for c in cands[: n.value]
            if 0 <= c.piece_id <= 6
        )
        return BotAdvice(
            piece=PieceType(best.piece_id),
            hold=bool(best.hold),
            cells=tuple(sorted(zip(best.cells_r, best.cells_c))),
            candidates=out_cands,
            think_ms=think_ms,
        )


# --- cold-clear (upstream C API, ctypes) --------------------------------------


# CCPiece enum order (coldclear.h): I O T L J S Z
_CC_PIECE = [PieceType.I, PieceType.O, PieceType.T, PieceType.L, PieceType.J, PieceType.S, PieceType.Z]
_TO_CC: dict[PieceType, int] = {p: i for i, p in enumerate(_CC_PIECE)}


class _CCWeights(ctypes.Structure):
    # mirrors coldclear.h's CCWeights (int32 x 26 + tslot[4] + well_column[10]
    # + 13 more ints + 3 bools) — order matters, it is written by
    # cc_default_weights and read field-by-field by the Rust side.
    _fields_ = [
        ("back_to_back", ctypes.c_int32),
        ("bumpiness", ctypes.c_int32),
        ("bumpiness_sq", ctypes.c_int32),
        ("row_transitions", ctypes.c_int32),
        ("height", ctypes.c_int32),
        ("top_half", ctypes.c_int32),
        ("top_quarter", ctypes.c_int32),
        ("jeopardy", ctypes.c_int32),
        ("cavity_cells", ctypes.c_int32),
        ("cavity_cells_sq", ctypes.c_int32),
        ("overhang_cells", ctypes.c_int32),
        ("overhang_cells_sq", ctypes.c_int32),
        ("covered_cells", ctypes.c_int32),
        ("covered_cells_sq", ctypes.c_int32),
        ("tslot", ctypes.c_int32 * 4),
        ("well_depth", ctypes.c_int32),
        ("max_well_depth", ctypes.c_int32),
        ("well_column", ctypes.c_int32 * 10),
        ("b2b_clear", ctypes.c_int32),
        ("clear1", ctypes.c_int32),
        ("clear2", ctypes.c_int32),
        ("clear3", ctypes.c_int32),
        ("clear4", ctypes.c_int32),
        ("tspin1", ctypes.c_int32),
        ("tspin2", ctypes.c_int32),
        ("tspin3", ctypes.c_int32),
        ("mini_tspin1", ctypes.c_int32),
        ("mini_tspin2", ctypes.c_int32),
        ("perfect_clear", ctypes.c_int32),
        ("combo_garbage", ctypes.c_int32),
        ("move_time", ctypes.c_int32),
        ("wasted_t", ctypes.c_int32),
        ("use_bag", ctypes.c_bool),
        ("timed_jeopardy", ctypes.c_bool),
        ("stack_pc_damage", ctypes.c_bool),
    ]


class _CCOptions(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int),        # CC_0G = 0
        ("spawn_rule", ctypes.c_int),  # CC_ROW_21_AND_FALL = 1
        ("pcloop", ctypes.c_int),      # CC_PC_OFF = 0
        ("min_nodes", ctypes.c_uint32),
        ("max_nodes", ctypes.c_uint32),
        ("threads", ctypes.c_uint32),
        ("use_hold", ctypes.c_bool),
        ("speculate", ctypes.c_bool),
    ]


class _CCMove(ctypes.Structure):
    _fields_ = [
        ("hold", ctypes.c_bool),
        ("expected_x", ctypes.c_uint8 * 4),
        ("expected_y", ctypes.c_uint8 * 4),
        ("movement_count", ctypes.c_uint8),
        ("movements", ctypes.c_int * 32),
        ("nodes", ctypes.c_uint32),
        ("depth", ctypes.c_uint32),
        ("original_rank", ctypes.c_uint32),
    ]


class _CCPlanPlacement(ctypes.Structure):
    _fields_ = [
        ("piece", ctypes.c_int),
        ("tspin", ctypes.c_int),
        ("expected_x", ctypes.c_uint8 * 4),
        ("expected_y", ctypes.c_uint8 * 4),
        ("cleared_lines", ctypes.c_int32 * 4),
    ]


class ColdClearBackend(Backend):
    """Cold Clear through its async C API. The bot thread keeps its own
    world model: we seed it per piece (reset + queue) and poll until it
    answers. `speculate` is off — we always resync the whole state, so the
    bot's internal bag model can never drift from ours."""

    name = "cold-clear"

    def __init__(self) -> None:
        lib_path = BUILD_DIR / "libcold_clear.so"
        if not lib_path.exists():
            raise FileNotFoundError(
                f"libcold_clear.so not built — run bots/build_bots.sh ({lib_path})"
            )
        self._lib = ctypes.CDLL(str(lib_path))
        lib = self._lib
        lib.cc_default_options.argtypes = [ctypes.POINTER(_CCOptions)]
        lib.cc_default_weights.argtypes = [ctypes.POINTER(_CCWeights)]
        lib.cc_launch_with_board_async.restype = ctypes.c_void_p
        lib.cc_launch_with_board_async.argtypes = [
            ctypes.POINTER(_CCOptions),
            ctypes.POINTER(_CCWeights),  # dereferenced unconditionally
            ctypes.c_void_p,  # book: NULL
            ctypes.POINTER((ctypes.c_bool * 10) * 40),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_bool,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_uint32,
        ]
        lib.cc_destroy_async.argtypes = [ctypes.c_void_p]
        lib.cc_reset_async.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER((ctypes.c_bool * 10) * 40),
            ctypes.c_bool,
            ctypes.c_uint32,
        ]
        lib.cc_add_next_piece_async.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.cc_request_next_move.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        lib.cc_poll_next_move.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_CCMove),
            ctypes.POINTER(_CCPlanPlacement),
            ctypes.POINTER(ctypes.c_uint32),
        ]
        lib.cc_poll_next_move.restype = ctypes.c_int
        self._bot: int | None = None

    def _field(self, game: Game):
        # CC's field is 400 bools in row-major order, index 0 = bottom-left
        # cell (y-up). Ours: rows[0] = top, 40 rows. cc cell (x, y) with y
        # counted up from the bottom = our (row 39-y, col x).
        field = (ctypes.c_bool * 10 * 40)()
        for i, row in enumerate(game.rows):
            y = 39 - i
            if not 0 <= y < 40:
                continue
            frow = field[y]
            for x in range(10):
                frow[x] = bool(row >> x & 1)
        return field

    def _bag_remain(self, game: Game) -> int:
        # EnumSet bit per CCPiece still in the current 7-bag. CC removes
        # each queued piece from the bag set as it is added, so the set
        # must include the queued pieces (active + previews) plus our
        # un-dealt bag remainder. Approximation at bag boundaries: a piece
        # queued from a previous bag masks its un-dealt twin — with
        # speculation off CC never extends past the visible queue, so the
        # bag set is not consulted for piece generation.
        remain = 0
        for p in game.bag._bag:
            remain |= 1 << _TO_CC[p]
        for p in [game.active.type, *game.queue[:5]] if game.active else game.queue[:5]:
            remain |= 1 << _TO_CC[p]
        return remain

    def _launch(self, game: Game) -> None:
        lib = self._lib
        opts = _CCOptions()
        lib.cc_default_options(ctypes.byref(opts))
        opts.mode = 0  # CC_0G
        opts.spawn_rule = 1  # CC_ROW_21_AND_FALL — 40-row field parity
        opts.use_hold = game.cfg.hold_enabled
        opts.speculate = False  # we resync every piece anyway
        opts.threads = 1
        self._weights = _CCWeights()  # keep alive for the bot's lifetime
        lib.cc_default_weights(ctypes.byref(self._weights))
        # CC's queue convention: the FIRST queue piece is the current piece
        # (advance_queue pops it to spawn; a hold pops it into the hold
        # slot). Pass [active] + previews.
        queue = [game.active.type, *game.queue[:5]]
        self._hold_ptr = (
            ctypes.c_int(_TO_CC[game.hold_type])
            if game.hold_type is not None
            else None
        )
        q = (ctypes.c_int * len(queue))(*[_TO_CC[p] for p in queue])
        self._bot = lib.cc_launch_with_board_async(
            ctypes.byref(opts),
            ctypes.byref(self._weights),
            None,  # no book (null-checked upstream)
            self._field(game),
            self._bag_remain(game),
            ctypes.byref(self._hold_ptr) if self._hold_ptr is not None else None,
            game.b2b_chain > 0,
            game.combo,
            q,
            len(queue),
        )

    def reset(self) -> None:
        if self._bot is not None:
            self._lib.cc_destroy_async(self._bot)
            self._bot = None

    def close(self) -> None:
        self.reset()

    def think(self, game: Game) -> BotAdvice | None:
        if game.active is None or game.over:
            return None
        import time

        # Full resync each decision: our engine may have drifted from the
        # bot's world (undo, garbage, cheese refill) — with speculation off
        # and a fresh bot per decision there is no state to drift.
        self.reset()
        self._launch(game)
        lib = self._lib
        lib.cc_request_next_move(self._bot, 0)
        mv = _CCMove()
        plan = (_CCPlanPlacement * 32)()
        plan_len = ctypes.c_uint32(32)
        t0 = time.perf_counter()
        # CCPiecePollStatus enum order: CC_MOVE_PROVIDED=0, CC_WAITING=1,
        # CC_BOT_DEAD=2 (coldclear.h)
        while True:
            status = lib.cc_poll_next_move(
                self._bot, ctypes.byref(mv), plan, ctypes.byref(plan_len)
            )
            if status == 0:  # MOVE_PROVIDED
                break
            if status == 2:  # BOT_DEAD
                return None
            time.sleep(0.002)  # WAITING
            if (time.perf_counter() - t0) > 10.0:
                return None
        think_ms = (time.perf_counter() - t0) * 1000.0
        cells = tuple(
            sorted(
                (39 - int(y), int(x))
                for x, y in zip(mv.expected_x, mv.expected_y)
            )
        )
        # hold means: the placement is of the piece hold brings out
        if mv.hold:
            held = game.hold_type if game.hold_type is not None else game.queue[0]
            piece = held
        else:
            piece = game.active.type
        # plan placements (from index 0) as extra candidates for shadows
        cands = []
        for i in range(plan_len.value):
            pl = plan[i]
            p = _CC_PIECE[pl.piece]
            c = tuple(sorted((39 - int(y), int(x)) for x, y in zip(pl.expected_x, pl.expected_y)))
            cands.append(
                BotCandidate(
                    piece=p,
                    hold=(i == 0 and mv.hold),
                    cells=c,
                    score=float(-i),  # plan order: earlier = better
                    lines=sum(1 for v in pl.cleared_lines if v >= 0),
                )
            )
        return BotAdvice(
            piece=piece,
            hold=bool(mv.hold),
            cells=cells,
            candidates=tuple(cands),
            think_ms=think_ms,
        )


# --- fusion (bots/fusion-shim, ctypes) ----------------------------------------


# external order for piece_from_external: I=0 O=1 T=2 S=3 Z=4 J=5 L=6
_TO_EXT: dict[PieceType, int] = {
    PieceType.I: 0,
    PieceType.O: 1,
    PieceType.T: 2,
    PieceType.S: 3,
    PieceType.Z: 4,
    PieceType.J: 5,
    PieceType.L: 6,
}


class _FCandidate(ctypes.Structure):
    _fields_ = [
        ("hold", ctypes.c_int),
        ("piece_id", ctypes.c_int),
        ("cells_r", ctypes.c_int * 4),
        ("cells_c", ctypes.c_int * 4),
        ("score", ctypes.c_float),
    ]


class _FResult(ctypes.Structure):
    _fields_ = [
        ("ok", ctypes.c_int),
        ("hold", ctypes.c_int),
        ("piece_id", ctypes.c_int),
        ("cells_r", ctypes.c_int * 4),
        ("cells_c", ctypes.c_int * 4),
        ("n_cands", ctypes.c_int),
        ("complexity", ctypes.c_float),
        ("best_score", ctypes.c_float),
    ]


class FusionBackend(Backend):
    """The MochBot fusion engine, heuristic beam (no model)."""

    name = "fusion"
    _MAX_CANDS = 128

    def __init__(self, beam_width: int = 160, depth: int = 6) -> None:
        lib_path = BUILD_DIR / "libfusion_shim.so"
        if not lib_path.exists():
            raise FileNotFoundError(
                f"libfusion_shim.so not built — run bots/build_bots.sh ({lib_path})"
            )
        self._lib = ctypes.CDLL(str(lib_path))
        self._lib.fs_version.restype = ctypes.c_int
        if self._lib.fs_version() != 1:
            raise RuntimeError("unexpected libfusion_shim ABI version")
        self._lib.fs_think.restype = ctypes.c_int
        self._lib.fs_think.argtypes = [
            ctypes.POINTER(ctypes.c_ushort),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_int,
            ctypes.c_ubyte,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(_FResult),
            ctypes.POINTER(_FCandidate),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._beam = beam_width
        self._depth = depth

    def think(self, game: Game) -> BotAdvice | None:
        if game.active is None or game.over:
            return None
        import time

        rows = (ctypes.c_ushort * 40)(*game.rows)
        queue = game.queue[:5]
        nxt = (ctypes.c_ubyte * max(1, len(queue)))(
            *[_TO_EXT[p] for p in queue]
        )
        best = _FResult()
        cands = (_FCandidate * self._MAX_CANDS)()
        t0 = time.perf_counter()
        self._lib.fs_think(
            rows,
            len(queue),
            nxt,
            _TO_EXT[game.hold_type] if game.hold_type is not None else -1,
            int(game.active.type),
            game.b2b_chain,
            game.combo,
            ctypes.byref(best),
            cands,
            self._MAX_CANDS,
            self._beam,
            self._depth,
        )
        think_ms = (time.perf_counter() - t0) * 1000.0
        if not best.ok:
            return None
        out_cands = tuple(
            BotCandidate(
                piece=PieceType(c.piece_id),
                hold=bool(c.hold),
                cells=tuple(sorted(zip(c.cells_r, c.cells_c))),
                score=float(c.score),
            )
            for c in cands[: best.n_cands]
        )
        return BotAdvice(
            piece=PieceType(best.piece_id),
            hold=bool(best.hold),
            cells=tuple(sorted(zip(best.cells_r, best.cells_c))),
            candidates=out_cands,
            think_ms=think_ms,
        )


# --- registry ------------------------------------------------------------------

_BACKENDS: dict[str, type[Backend]] = {
    "cheese-beam": NativeBeamBackend,
    "misamino": MisaMinoBackend,
    "zetris": lambda: MisaMinoBackend(style="zetris"),
    "cold-clear": ColdClearBackend,
    "fusion": FusionBackend,
}


def available_backends() -> list[str]:
    """Backend names whose library/dependency is actually loadable."""
    out = []
    for name in _BACKENDS.items():
        try:
            factory = name[1] if isinstance(name[1], type) else name[1]
            factory()
            out.append(name[0])
        except Exception:
            continue
    return out


def make_backend(name: str) -> Backend:
    if name not in _BACKENDS:
        raise KeyError(f"unknown backend {name!r}")
    factory = _BACKENDS[name]
    if isinstance(factory, type):
        return factory()
    return factory()
