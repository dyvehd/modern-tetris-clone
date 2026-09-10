"""pygame-ce application: menu, settings (key remapping + handling), game loop.

Run with ``python -m tetris`` (or the ``tetris`` script after pip install).
"""

from __future__ import annotations

import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pygame

from .config import (
    DEFAULT_LOG_DIR,
    DEFAULT_SETTINGS_PATH,
    MODES,
    AppConfig,
    Keybinds,
    load_config,
    make_mode_config,
    save_config,
)
from .engine import board as B
from .engine.constants import PIECE_LETTERS, piece_from_letter
from .engine.game import Btn, Game
from .input import InputConfig, InputController, InputLogger
from .render.renderer import Renderer

FIXED_DT = 1 / 60

KEY_ALIASES = {"ctrl": ("LCTRL", "RCTRL"), "shift": ("LSHIFT", "RSHIFT"), "alt": ("LALT", "RALT")}

_KEY_CACHE: dict[str, list[int]] = {}


def parse_key_names(spec: str) -> list[int]:
    """Map a comma-separated key-name spec to pygame key codes.

    Accepts lowercase names from settings ("left", "escape", "space", "z",
    "f12") and resolves them case-insensitively against pygame's K_* constants
    (pygame defines single letters lowercase, everything else uppercase).
    Results are cached; unknown names warn exactly once.
    """
    cached = _KEY_CACHE.get(spec)
    if cached is not None:
        return cached
    codes: list[int] = []
    for name in [s.strip().lower() for s in spec.split(",") if s.strip()]:
        candidates = KEY_ALIASES.get(name, (name,))
        for cand in candidates:
            code = getattr(pygame, f"K_{cand}", None)
            if code is None:
                code = getattr(pygame, f"K_{cand.upper()}", None)
            if code is not None:
                codes.append(code)
                break
        else:
            print(f"warning: unknown key '{name}' in keybinds", file=sys.stderr)
    _KEY_CACHE[spec] = codes
    return codes


_KEY_CODE_NAMES: dict[int, str] | None = None


def key_code_name(code: int) -> str:
    """Reverse of parse_key_names: a pygame key code -> a spec name that
    round-trips through the parser (e.g. K_LSHIFT -> "lshift")."""
    global _KEY_CODE_NAMES
    if _KEY_CODE_NAMES is None:
        names: dict[int, str] = {}
        for attr, value in vars(pygame).items():
            if attr.startswith("K_") and isinstance(value, int):
                names.setdefault(value, attr[2:].lower())
        _KEY_CODE_NAMES = names
    return _KEY_CODE_NAMES.get(code, "")


@dataclass
class SettingRow:
    row_id: str  # "das" | "arr" | "sdf" | "key:<field>"
    kind: str  # "header" | "value" | "bind"
    label: str


BINDABLE_KEYS: list[tuple[str, str]] = [
    ("left", "Move left"),
    ("right", "Move right"),
    ("soft_drop", "Soft drop"),
    ("hard_drop", "Hard drop"),
    ("rotate_cw", "Rotate CW"),
    ("rotate_ccw", "Rotate CCW"),
    ("rotate_180", "Rotate 180"),
    ("hold", "Hold"),
    ("restart", "Restart"),
    ("pause", "Pause"),
    ("screenshot", "Screenshot"),
]

DAS_MAX = 333.0
ARR_MAX = 100.0
SDF_MAX = 40.0
MAX_UNDO = 500  # zen undo history depth (one snapshot per spawned piece)


@dataclass
class _EditStroke:
    """One mouse drag on the board (a click is a 1-cell stroke).

    ``cells`` are the distinct cells this stroke changed, in order —
    four-tris's StrokeCoord: 4 gray cells get recognized as a tetromino,
    the 5th reverts them to gray. ``pushed_undo`` snapshots the board once
    per stroke so Ctrl+Z reverts the whole edit.
    """

    erase: bool
    cells: list[tuple[int, int]] = field(default_factory=list)
    seen: set = field(default_factory=set)
    pushed_undo: bool = False

# gameplay keybinds: (Keybinds field, virtual button). Drives both key
# routing and the input log, so a rebound key logs its action name.
_GAME_BTNS: tuple[tuple[str, Btn], ...] = (
    ("left", Btn.LEFT),
    ("right", Btn.RIGHT),
    ("soft_drop", Btn.SOFT),
    ("rotate_cw", Btn.ROT_CW),
    ("rotate_ccw", Btn.ROT_CCW),
    ("rotate_180", Btn.ROT_180),
    ("hard_drop", Btn.HARD),
    ("hold", Btn.HOLD),
)


class App:
    STATE_MENU = "menu"
    STATE_SETTINGS = "settings"
    STATE_PLAY = "play"
    STATE_PAUSE = "pause"
    STATE_OVER = "over"

    def __init__(self, config: AppConfig | None = None,
                 settings_path: Path | None = None) -> None:
        pygame.init()
        # Auto-repeat is handy in the menu and settings screens (hold a key
        # to drag a value). Gameplay never sees synthetic repeats: events
        # for keys that are already down are dropped in handle_events.
        pygame.key.set_repeat(300, 50)
        self.cfg: AppConfig = config if config is not None else load_config()
        self.settings_path = settings_path or DEFAULT_SETTINGS_PATH
        self.screen = pygame.display.set_mode((960, 780))
        pygame.display.set_caption("Modern Tetris")
        self.renderer = Renderer(self.screen, cell=self.cfg.render.cell,
                                 buffer_rows=self.cfg.render.buffer_rows_shown)
        self.controller = InputController(self.cfg.input)
        self.state = self.STATE_MENU
        self.mode_idx = 0
        self.modes = list(MODES.keys())
        self.game: Game | None = None
        self.trainer = False
        self._trainer_ticks = 0
        # the AI cheese trainer (None outside the trainer mode)
        self.ai_trainer = None
        self._last_flashed = None  # last MoveQuality that flashed a popup
        # zen undo (TETR.IO-style Ctrl+Z): while you control a piece we keep
        # its spawn snapshot; when it locks, that snapshot joins the undo
        # stack — undoing puts the placed piece back in your hands
        self.undo_enabled = False
        self._undo_stack: list[Game] = []
        self._spawn_snapshot: Game | None = None
        self._last_undo_active: object | None = None
        self._undo_pieces = 0
        # zen sandbox mouse editor (four-tris style): board painting and the
        # queue editor. A stroke paints editor-gray cells as you drag and
        # auto-colors exactly-4-cell tetromino strokes with the piece color.
        self.edit_enabled = False
        self._edit_stroke: _EditStroke | None = None
        self._edit_last_pos: tuple[int, int] | None = None
        self._hover_cell: tuple[int, int] | None = None
        # queue editor dialog state (None = closed): dict with keys
        # seq (str), off (str), field (0=seq 1=offset), error (str)
        self.queue_edit: dict | None = None
        # physical key codes currently down; dedups OS auto-repeat events
        self._keys_down: set[int] = set()
        # settings screen state
        self.settings_idx = 0  # index among selectable rows
        self.capture_field: str | None = None
        self._settings_return = self.STATE_MENU
        # key/piece event log for debugging handling (one file per session,
        # created lazily on the first logged event)
        self.keylog: InputLogger | None = None
        if self.cfg.debug.log_input:
            log_dir = (Path(self.cfg.debug.input_log_dir) if self.cfg.debug.input_log_dir
                       else DEFAULT_LOG_DIR)
            self.keylog = InputLogger(
                log_dir / f"input_{time.strftime('%Y%m%d_%H%M%S')}.log")

    # ------------------------------------------------------------------ flow

    def run(self) -> None:
        clock = pygame.time.Clock()
        running = True
        acc = 0.0
        try:
            while running:
                dt = clock.tick(60) / 1000.0
                running = self.handle_events()
                acc = min(acc + dt, 0.25)
                while acc >= FIXED_DT:
                    self.logic_tick()
                    acc -= FIXED_DT
                self.render()
        finally:
            if self.keylog is not None:
                self.keylog.close()
            pygame.quit()

    def start_game(self) -> None:
        mode = self.modes[self.mode_idx]
        rules, trainer = make_mode_config(mode, self.cfg)
        # the AI trainer pins infinite SDF: the movegen/pathfinder model is
        # sonic drop, and finite soft drop would make bot hints unnavigable
        if MODES[mode].get("ai_trainer"):
            rules.soft_drop_factor = math.inf
        else:
            rules.soft_drop_factor = self.cfg.input.sdf
        seed = random.randrange(1 << 62)  # recorded in the input log: replayable
        self.game = Game(rules, seed=seed)
        self.trainer = trainer
        self._trainer_ticks = 0
        self.undo_enabled = bool(MODES[mode].get("undo", False))
        if not self.undo_enabled:
            self._undo_stack.clear()  # the history belongs to zen-style modes
        # the AI cheese trainer (100L trainer mode)
        if MODES[mode].get("ai_trainer"):
            if self.ai_trainer is None:
                from .trainer.trainer import Trainer, TrainerConfig

                self.ai_trainer = Trainer(TrainerConfig())
                self.ai_trainer.set_undo_hook(self._trainer_undo)
            self.ai_trainer.on_restart()
        elif self.ai_trainer is not None:
            self.ai_trainer.close()
            self.ai_trainer = None
        self.edit_enabled = bool(MODES[mode].get("edit", False))
        self._edit_stroke = None
        self._edit_last_pos = None
        self._hover_cell = None
        self.queue_edit = None
        # per-game undo bookkeeping restarts with the new game, but the
        # history itself survives restarts: R then Ctrl+Z reaches into the
        # game you just left, top-out included (TETR.IO zen behaviour)
        self._spawn_snapshot = None
        self._last_undo_active = None
        self._undo_pieces = 0
        self.renderer.popups = []
        self.controller = InputController(self.cfg.input)
        self.state = self.STATE_PLAY
        if self.keylog is not None:
            self.keylog.note(
                f"# {time.strftime('%Y-%m-%d %H:%M:%S')} game start: {mode}"
                f" | seed {seed} | {self._handling_str()}")

    def _note(self, text: str) -> None:
        if self.keylog is not None:
            self.keylog.note(text)

    def _handling_str(self) -> str:
        i = self.cfg.input
        sdf = "inf" if math.isinf(i.sdf) else f"{i.sdf:g}x"
        return f"das {i.das_ms:g}ms arr {i.arr_ms:g}ms sdf {sdf}"

    def open_settings(self, from_state: str) -> None:
        self._settings_return = from_state
        self.capture_field = None
        self.settings_idx = 0
        # nothing is held while browsing menus; a stale DAS direction would
        # fire the moment the game resumes
        self.controller.release_all()
        self.state = self.STATE_SETTINGS

    def close_settings(self) -> None:
        save_config(self.cfg, self.settings_path)
        self.controller.release_all()
        self._apply_handling()
        self._note(f"handling changed: {self._handling_str()}")
        self.state = self._settings_return

    def _apply_handling(self) -> None:
        """Push DAS/ARR/SDF into the live controller and running game."""
        self.controller.apply_config(self.cfg.input)
        if self.game is not None:
            # the AI trainer keeps infinite SDF: the movegen/pathfinder
            # model is sonic drop, and finite soft drop would desync bot
            # hints (the settings screen still adjusts DAS/ARR live)
            if not self.ai_trainer:
                self.game.cfg.soft_drop_factor = self.cfg.input.sdf

    def restart(self) -> None:
        self.start_game()

    def _trainer_undo(self) -> bool:
        """Undo hook for the AI trainer (live feedback's auto-undo). Same
        snapshot discipline as _undo, plus the trainer-side cleanup."""
        if not self._undo_stack:
            return False
        restored = self._undo()
        if restored and self.ai_trainer is not None:
            self.ai_trainer.on_undo()
        return restored

    # AI trainer keybinds (trainer mode only; fixed keys by design — they
    # are overlay switches, not gameplay inputs):
    #   F1  AI on/off          F2  shadows on/off       F3  live feedback
    #   F4  automove on/off    F5  step mode           F6  next AI model
    #   F7/F8  lookahead -/+   F9/F10 automove PPS -/+
    def _trainer_key(self, key: int) -> bool:
        """Handle an AI-trainer hotkey. Returns True when consumed."""
        t = self.ai_trainer
        if t is None:
            return False
        cfg = t.cfg
        from .trainer import backends as BB

        if key == pygame.K_F1:
            cfg.toggle("ai")
            note = f"AI {'on' if cfg.ai_on else 'off'}"
        elif key == pygame.K_F2:
            cfg.toggle("shadows")
            note = f"shadows {'on' if cfg.shadows_on else 'off'}"
        elif key == pygame.K_F3:
            cfg.toggle("feedback")
            note = f"live feedback {'on' if cfg.live_feedback else 'off'}"
        elif key == pygame.K_F4:
            cfg.toggle("automove")
            note = f"automove {'on' if cfg.automove else 'off'}"
        elif key == pygame.K_F5:
            cfg.toggle("step")
            note = f"step mode {'on' if cfg.step_mode else 'off'}"
        elif key == pygame.K_F6:
            names = BB.available_backends()
            if not names:
                return True
            i = names.index(cfg.backend) if cfg.backend in names else 0
            t.set_backend(names[(i + 1) % len(names)])
            note = f"model: {t.cfg.backend}"
        elif key in (pygame.K_F7, pygame.K_F8):
            cfg.lookahead = max(0, min(9, cfg.lookahead + (1 if key == pygame.K_F8 else -1)))
            note = f"lookahead {cfg.lookahead + 1}"
        elif key in (pygame.K_F9, pygame.K_F10):
            cfg.automove_pps = max(
                0.5, min(20.0, cfg.automove_pps + (0.5 if key == pygame.K_F10 else -0.5))
            )
            note = f"automove {cfg.automove_pps:g} pps"
        else:
            return False
        self._note(note)
        self.renderer.add_popup(note.upper())
        return True

    def _undo(self) -> bool:
        """Zen undo: restore the snapshot taken when the piece you just
        placed spawned — the board, queue, hold and score go back to just
        before that placement, with the piece back in your hands. Returns
        True if a snapshot was restored."""
        if not self._undo_stack:
            return False
        self.game = self._undo_stack.pop()
        self._last_undo_active = self.game.active
        self._spawn_snapshot = self.game.clone()
        self._undo_pieces = self.game.pieces_placed
        self._note("undo")
        self.renderer.add_popup("UNDO")
        return True

    # ---------------------------------------------------------------- events

    def handle_events(self) -> bool:
        """One pump of the event queue. OS key auto-repeat never reaches the
        game: a KEYDOWN for a key that is already down is a synthetic repeat
        (pygame-ce repeat events carry no reliable ``repeat`` attribute), so
        it is dropped by tracking held keys — not by trusting event metadata.
        The settings screen is exempt: it wants repeat to drag values."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.WINDOWFOCUSLOST:
                # No KEYUPs arrive for keys held while the window is unfocused;
                # stale entries would otherwise eat the next real press.
                self._keys_down.clear()
                if self.state == self.STATE_PLAY:
                    self.controller.release_all()
            if event.type == pygame.KEYDOWN:
                if event.key in self._keys_down:
                    if self.state not in (self.STATE_MENU, self.STATE_SETTINGS):
                        continue  # synthetic repeat: a held key is not a press
                else:
                    self._keys_down.add(event.key)
                if not self.handle_key(event.key, event):
                    return False
            elif event.type == pygame.KEYUP:
                self._keys_down.discard(event.key)
                self.handle_keyup(event.key)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                shift = bool(pygame.key.get_mods() & pygame.KMOD_SHIFT)
                self.handle_mouse_buttondown(event.pos, event.button, shift)
            elif event.type == pygame.MOUSEMOTION:
                self.handle_mouse_motion(event.pos, event.buttons)
            elif event.type == pygame.MOUSEBUTTONUP:
                self.handle_mouse_buttonup(event.button)
        return True

    def handle_key(self, key: int, event: pygame.event.Event) -> bool:
        if self.state == self.STATE_SETTINGS:
            return self._settings_key(key, event)
        keys = self.cfg.keys
        if key in parse_key_names(keys.screenshot):
            self.screenshot()
        if self.queue_edit is not None:  # modal: the queue dialog owns keys
            return self._queue_dialog_key(key)
        if self.state == self.STATE_MENU:
            if key in (pygame.K_UP, pygame.K_w):
                self.mode_idx = (self.mode_idx - 1) % len(self.modes)
            elif key == pygame.K_DOWN:
                self.mode_idx = (self.mode_idx + 1) % len(self.modes)
            elif key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                self.start_game()
            elif key == pygame.K_s:
                self.open_settings(self.STATE_MENU)
            elif key == pygame.K_ESCAPE:
                return False
        elif self.state == self.STATE_PLAY:
            if key in parse_key_names(keys.pause):
                self.state = self.STATE_PAUSE
                self._note("paused")
            elif key in parse_key_names(keys.restart):
                self.restart()
            elif key == pygame.K_q:
                self._note("quit to menu")
                self.state = self.STATE_MENU
            elif (self.undo_enabled and key == pygame.K_z
                    and event.mod & pygame.KMOD_CTRL):
                self._undo()
            elif self.ai_trainer is not None and self._trainer_key(key):
                pass  # a trainer hotkey was consumed
            elif (
                self.ai_trainer is not None
                and self.ai_trainer.cfg.step_mode
                and self.ai_trainer.cfg.automove
                and key in parse_key_names(keys.hard_drop)
            ):
                # step mode: the hard-drop key steps one bot move
                self.ai_trainer.request_step()
            else:
                self.route_game_key(key)
        elif self.state == self.STATE_PAUSE:
            if key in parse_key_names(keys.pause):
                self.state = self.STATE_PLAY
                self._note("resumed")
            elif key in parse_key_names(keys.restart):
                self.restart()
            elif key in (pygame.K_s, pygame.K_o):
                self.open_settings(self.STATE_PAUSE)
            elif key == pygame.K_q:
                self._note("quit to menu")
                self.state = self.STATE_MENU
        elif self.state == self.STATE_OVER:
            if key in parse_key_names(keys.restart):
                self.restart()
            elif (self.undo_enabled and key == pygame.K_z
                    and event.mod & pygame.KMOD_CTRL):
                # zen: step back out of a top-out into the finished game
                if self._undo():
                    self.state = self.STATE_PLAY
                    self._note("undo after top out")
            elif key == pygame.K_q or key == pygame.K_ESCAPE:
                self._note("quit to menu")
                self.state = self.STATE_MENU
        return True

    def handle_keyup(self, key: int) -> None:
        if self.state == self.STATE_SETTINGS:
            return
        # log releases only in a game context; menu key-ups have no down
        in_game = self.state in (self.STATE_PLAY, self.STATE_PAUSE, self.STATE_OVER)
        keys = self.cfg.keys
        for field_name, btn in _GAME_BTNS:
            if key in parse_key_names(getattr(keys, field_name)):
                self.controller.release(btn)
                if self.keylog is not None and in_game:
                    self.keylog.key(btn, False)

    def route_game_key(self, key: int) -> None:
        keys = self.cfg.keys
        c = self.controller
        for field_name, btn in _GAME_BTNS:
            if key in parse_key_names(getattr(keys, field_name)):
                c.press(btn)
                if self.keylog is not None:
                    self.keylog.key(btn, True)

    # ------------------------------------------------- mouse board editor

    def _editing(self) -> bool:
        return (
            self.edit_enabled
            and self.state == self.STATE_PLAY
            and self.game is not None
            and not self.game.over
            and self.queue_edit is None
        )

    def handle_mouse_buttondown(self, pos, button: int, shift: bool) -> None:
        if not self._editing():
            return
        if button == 1 and self.renderer.hit_next_box(pos):
            self._open_queue_edit()
            return
        if button not in (1, 3):
            return
        erase = button == 3 or shift  # right click or shift+left erases
        self._edit_stroke = _EditStroke(erase=erase)
        self._edit_last_pos = pos
        self._edit_apply_at(pos)

    def handle_mouse_motion(self, pos, buttons) -> None:
        if self._editing():
            cell = self.renderer.cell_at(pos)
            self._hover_cell = cell  # None when off the field
        stroke = self._edit_stroke
        if stroke is None or not buttons[0] and not buttons[2]:
            self._edit_last_pos = pos
            return
        # interpolate along the motion so fast drags leave no gaps
        # (four-tris samples 7 steps between mouse polls)
        x0, y0 = self._edit_last_pos if self._edit_last_pos is not None else pos
        x1, y1 = pos
        dist = max(abs(x1 - x0), abs(y1 - y0))
        steps = max(1, dist // max(1, self.renderer.cell // 2))
        for i in range(steps + 1):
            t = i / steps
            self._edit_apply_at((round(x0 + (x1 - x0) * t), round(y0 + (y1 - y0) * t)))
        self._edit_last_pos = pos

    def handle_mouse_buttonup(self, button: int) -> None:
        if button in (1, 3):
            self._edit_stroke = None
            self._edit_last_pos = None

    def _edit_apply_at(self, pos) -> None:
        """Apply the active stroke's paint/erase to the cell under ``pos``."""
        stroke = self._edit_stroke
        if stroke is None or self.game is None:
            return
        cell = self.renderer.cell_at(pos)
        if cell is None or cell in stroke.seen:
            # one paint per distinct cell per stroke: interpolation samples
            # revisit cells, and a revisit must not repaint an
            # auto-colored tetromino cell back to gray
            return
        ry, x = cell
        game = self.game
        occupied = game.rows[ry] >> x & 1
        if stroke.erase and not occupied:
            return
        if not stroke.pushed_undo:
            # the pre-edit board joins the undo stack BEFORE the first
            # change of the stroke — Ctrl+Z reverts the whole stroke at once
            stroke.pushed_undo = True
            self._undo_stack.append(game.clone())
            del self._undo_stack[:-MAX_UNDO]
            self._note("board edit stroke" + (" (erase)" if stroke.erase else ""))
        changed = game.edit_erase(ry, x) if stroke.erase else game.edit_paint(ry, x)
        if not changed:
            return
        stroke.seen.add(cell)
        stroke.cells.append(cell)
        if stroke.erase:
            return
        # four-tris AutoColor: while the stroke is 1-3 cells, any gray
        # component completed to exactly 4 by this paint becomes a piece;
        # the 4th stroke cell resolves the stroke itself; the 5th reverts
        # the first four to gray. PieceType.I is IntEnum value 0, so the
        # recognizer result must be tested against None, never truthiness.
        n = len(stroke.cells)
        if n < 4:
            comp = B.gray_component(game.styles, ry, x)
            if comp is not None and len(comp) == 4:
                piece = B.piece_from_cells(comp)
                if piece is not None:
                    for cy, cx in comp:
                        game.edit_paint(cy, cx, piece.value)
        elif n == 4:
            piece = B.piece_from_cells(stroke.cells)
            if piece is not None:
                for cy, cx in stroke.cells:
                    game.edit_paint(cy, cx, piece.value)
        elif n == 5:
            # the stroke went past 4 cells: it is not a tetromino gesture,
            # so the first four go back to plain gray
            for cy, cx in stroke.cells[:4]:
                game.edit_paint(cy, cx)

    # ---------------------------------------------------- queue edit dialog

    def _open_queue_edit(self) -> None:
        """Open the queue editor (four-tris BagSet): replace the upcoming
        pieces with a typed sequence + a 7-bag offset. Prefilled with the
        current queue; emptying the sequence returns to random bags."""
        assert self.game is not None
        self.queue_edit = {
            "seq": "".join(PIECE_LETTERS[p] for p in self.game.queue),
            "off": "0",
            "field": 0,  # 0 = sequence, 1 = offset
            "error": "",
        }
        self.controller.release_all()
        self._note("queue editor opened")

    def _queue_dialog_key(self, key: int) -> bool:
        qd = self.queue_edit
        if key in (pygame.K_ESCAPE, pygame.K_q):
            self.queue_edit = None
        elif key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self._apply_queue_edit()
        elif key in (pygame.K_TAB, pygame.K_UP, pygame.K_DOWN):
            qd["field"] = 1 - qd["field"]
            qd["error"] = ""
        elif key in (pygame.K_BACKSPACE, pygame.K_DELETE):
            if qd["field"] == 0:
                qd["seq"] = qd["seq"][:-1]
            else:
                qd["off"] = "0"
            qd["error"] = ""
        else:
            ch = pygame.key.name(key)
            if len(ch) == 1:
                if qd["field"] == 0:
                    if ch.upper() in "IJLOSTZ":
                        qd["seq"] += ch.upper()
                        qd["error"] = ""
                    else:
                        qd["error"] = "sequence: letters I J L O S T Z only"
                elif ch.isdigit():
                    qd["off"] = ch  # single-digit field
                    qd["error"] = ""
                else:
                    qd["error"] = "offset: one digit 0-6"
        return True  # the dialog swallows every key while open

    def _apply_queue_edit(self) -> None:
        qd = self.queue_edit
        game = self.game
        assert qd is not None and game is not None
        pieces = [piece_from_letter(c) for c in qd["seq"]]
        off = int(qd["off"]) if qd["off"].isdigit() else 0
        if not 0 <= off <= 6:
            qd["error"] = "offset must be 0-6"
            return
        if list(game.queue) != pieces or game.bag_pos != off:
            # like a board edit: the pre-edit state joins the undo stack
            self._undo_stack.append(game.clone())
            del self._undo_stack[:-MAX_UNDO]
            game.set_queue(pieces, bag_offset=off)
            self.renderer.add_popup("QUEUE SET")
            self._note(f"queue set: {qd['seq'] or '(random)'} offset {off}")
        self.queue_edit = None

    # -------------------------------------------------------------- settings

    def _settings_rows(self) -> list[SettingRow]:
        rows = [
            SettingRow("h1", "header", "HANDLING"),
            SettingRow("das", "value", "DAS"),
            SettingRow("arr", "value", "ARR"),
            SettingRow("sdf", "value", "Soft drop speed"),
            SettingRow("h2", "header", "KEY BINDINGS"),
        ]
        rows.extend(SettingRow(f"key:{name}", "bind", label) for name, label in BINDABLE_KEYS)
        return rows

    def _bind_value_text(self, field_name: str) -> str:
        if self.capture_field == field_name:
            return "[ press a key ]" if pygame.time.get_ticks() // 400 % 2 == 0 else ""
        spec = getattr(self.cfg.keys, field_name)
        codes = parse_key_names(spec)
        if not codes:
            return "-"
        return ", ".join(pygame.key.name(c) for c in codes)

    def _settings_items(self) -> list[tuple[str, str, str, bool]]:
        cfg = self.cfg.input
        values = {
            "das": f"{cfg.das_ms:g} ms",
            "arr": f"{cfg.arr_ms:g} ms" + (" (instant)" if cfg.arr_ms == 0 else ""),
            "sdf": "inf (instant)" if math.isinf(cfg.sdf) else f"{cfg.sdf:g}x",
        }
        items: list[tuple[str, str, str, bool]] = []
        selectable_i = 0
        capture_row = self.capture_field
        for row in self._settings_rows():
            if row.kind == "header":
                items.append((row.kind, row.label, "", False))
                continue
            if row.kind == "value":
                value = values[row.row_id]
            else:
                value = self._bind_value_text(row.row_id[4:])
            items.append((row.kind, row.label, value, selectable_i == self.settings_idx))
            selectable_i += 1
        # flag the row being captured so the renderer blinks it
        if capture_row is not None:
            for i, (kind, label, value, selected) in enumerate(items):
                if kind == "bind" and label == self._bind_label(capture_row):
                    items[i] = (kind, label, value, True)
        return items

    @staticmethod
    def _bind_label(field_name: str) -> str:
        return next(label for name, label in BINDABLE_KEYS if name == field_name)

    def _settings_key(self, key: int, event: pygame.event.Event) -> bool:
        # pygame-ce events only carry .repeat on actual auto-repeat events;
        # normal keydowns raise AttributeError on attribute access.
        repeat = getattr(event, "repeat", False)
        rows = self._settings_rows()
        selectable = [r for r in rows if r.kind != "header"]
        if self.capture_field is not None:
            if repeat:
                return True
            if key == pygame.K_ESCAPE:
                self.capture_field = None
            elif key in (pygame.K_BACKSPACE, pygame.K_DELETE):
                setattr(self.cfg.keys, self.capture_field, "")
                self.capture_field = None
            else:
                setattr(self.cfg.keys, self.capture_field, key_code_name(key))
                self.capture_field = None
            return True
        if repeat and key not in (pygame.K_LEFT, pygame.K_RIGHT):
            return True
        self.settings_idx %= len(selectable)
        row = selectable[self.settings_idx]
        if key in (pygame.K_UP, pygame.K_w):
            self.settings_idx = (self.settings_idx - 1) % len(selectable)
        elif key in (pygame.K_DOWN, pygame.K_s):
            self.settings_idx = (self.settings_idx + 1) % len(selectable)
        elif key == pygame.K_LEFT:
            self._adjust_setting(row.row_id, -1)
        elif key == pygame.K_RIGHT:
            self._adjust_setting(row.row_id, +1)
        elif key == pygame.K_RETURN:
            if row.kind == "bind":
                self.capture_field = row.row_id[4:]
        elif key == pygame.K_r:
            self.cfg.input = InputConfig()
            self.cfg.keys = Keybinds()
            self._apply_handling()
        elif key in (pygame.K_ESCAPE, pygame.K_q):
            self.close_settings()
        return True

    def _adjust_setting(self, row_id: str, direction: int) -> None:
        fine = bool(pygame.key.get_mods() & pygame.KMOD_SHIFT)
        cfg = self.cfg.input
        if row_id == "das":
            step = 1 if fine else 5
            cfg.das_ms = min(DAS_MAX, max(0.0, cfg.das_ms + direction * step))
        elif row_id == "arr":
            step = 1 if fine else 5
            cfg.arr_ms = min(ARR_MAX, max(0.0, cfg.arr_ms + direction * step))
        elif row_id == "sdf":
            if math.isinf(cfg.sdf):
                if direction < 0:
                    cfg.sdf = SDF_MAX
            else:
                cfg.sdf = min(SDF_MAX, max(1.0, cfg.sdf + direction))
                if cfg.sdf >= SDF_MAX and direction > 0:
                    cfg.sdf = math.inf
        self._apply_handling()

    # ----------------------------------------------------------------- logic

    def logic_tick(self) -> None:
        self.renderer.tick_popups()
        if self.state != self.STATE_PLAY or self.game is None:
            return
        if self.queue_edit is not None:
            return  # the queue dialog freezes the game while open
        game = self.game
        actions, held = self.controller.update()
        game.tick(actions, held)
        if self.keylog is not None:  # identity compare; writes on spawn only
            self.keylog.check_piece(game.active)

        # zen undo bookkeeping: a placement moves the placed piece's spawn
        # snapshot onto the undo stack; a spawn refreshes the held snapshot
        # (both can happen in the same tick — order matters)
        if self.undo_enabled and not game.over:
            if game.pieces_placed > self._undo_pieces and self._spawn_snapshot is not None:
                self._undo_pieces = game.pieces_placed
                self._undo_stack.append(self._spawn_snapshot)
                del self._undo_stack[:-MAX_UNDO]
            if game.active is not None and game.active is not self._last_undo_active:
                self._last_undo_active = game.active
                self._spawn_snapshot = game.clone()

        # AI trainer: annotation, live feedback, shadows data, automove.
        # Runs BEFORE the popup loop so a lock's annotation can flash in
        # the same tick it happened; automove's own ticks return their
        # events to this same handling.
        trainer_events: list[dict] = []
        if self.ai_trainer is not None and not game.over:
            trainer_events = self.ai_trainer.tick(game, list(game.events))

        for ev in list(game.events) + trainer_events:
            if ev.get("kind") == "clear":
                self.renderer.add_popup(ev["label"], ev.get("attack", 0))
            elif ev.get("kind") == "tspin":
                self.renderer.add_popup(ev["label"], 0)
            elif (
                ev.get("kind") == "lock"
                and self.ai_trainer is not None
                and self.ai_trainer.last_quality is not None
                and self.ai_trainer.cfg.shadows_on is False
            ):
                # the annotation flashes only when hints are hidden (with
                # shadows on, the board already shows the answer)
                q = self.ai_trainer.last_quality
                if self.ai_trainer.stats.n and q is not self._last_flashed:
                    self._last_flashed = q
                    self.renderer.add_popup(q.label.upper(), 0)

        if self.trainer:
            self._trainer_ticks += 1
            # send 4 rows every 15 s, cancelled/rises like a VS opponent
            if self._trainer_ticks % (15 * 60) == 0:
                game.add_garbage(4)

        if game.over:
            self.state = self.STATE_OVER
            self._note("finished" if game.won else "top out")

    # ---------------------------------------------------------------- render

    def render(self) -> None:
        # The renderer only composes onto self.screen; exactly one flip per
        # frame happens here. (Flipping per draw-call made the game-over
        # screen flash: draw() flipped the undimmed scene, then the overlay
        # flipped again on top of it, every frame.)
        if self.state == self.STATE_SETTINGS:
            self.renderer.draw_settings(self._settings_items(), self.capture_field is not None)
        elif self.state == self.STATE_MENU:
            self.renderer.draw_menu(self.modes, self.mode_idx, [MODES[m]["desc"] for m in self.modes])
        elif self.game is None:
            self.renderer.draw_menu(self.modes, self.mode_idx, [MODES[m]["desc"] for m in self.modes])
        elif self.state == self.STATE_OVER:
            self.renderer.draw(self.game, self.modes[self.mode_idx])
            self.renderer.draw_game_over(self.game, self.modes[self.mode_idx],
                                         undo_hint=self.undo_enabled)
        else:
            edit = self.edit_enabled and self.state == self.STATE_PLAY
            shadows = []
            if self.ai_trainer is not None and self.state == self.STATE_PLAY:
                shadows = self.ai_trainer.shadow_placements(self.game)
            self.renderer.draw(self.game, self.modes[self.mode_idx],
                               paused=(self.state == self.STATE_PAUSE),
                               undo_hint=self.undo_enabled, edit=edit,
                               hover=self._hover_cell if edit else None,
                               ai_shadows=shadows)
            if self.ai_trainer is not None and self.state == self.STATE_PLAY:
                self.renderer.draw_trainer_panel(self.ai_trainer.hud_lines())
            if self.queue_edit is not None:
                self.renderer.draw_queue_dialog(self.queue_edit)
        pygame.display.flip()

    def screenshot(self) -> None:
        name = f"tetris_{time.strftime('%Y%m%d_%H%M%S')}.png"
        pygame.image.save(self.screen, name)
        print(f"saved {name}")


def main() -> None:
    App().run()


if __name__ == "__main__":
    main()
