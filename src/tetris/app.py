"""pygame-ce application: menu, settings (key remapping + handling), game loop.

Run with ``python -m tetris`` (or the ``tetris`` script after pip install).
"""

from __future__ import annotations

import math
import random
import sys
import time
from dataclasses import dataclass
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
        # zen undo (TETR.IO-style Ctrl+Z): while you control a piece we keep
        # its spawn snapshot; when it locks, that snapshot joins the undo
        # stack — undoing puts the placed piece back in your hands
        self.undo_enabled = False
        self._undo_stack: list[Game] = []
        self._spawn_snapshot: Game | None = None
        self._last_undo_active: object | None = None
        self._undo_pieces = 0
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
        rules.soft_drop_factor = self.cfg.input.sdf  # handling lives in [input]
        seed = random.randrange(1 << 62)  # recorded in the input log: replayable
        self.game = Game(rules, seed=seed)
        self.trainer = trainer
        self._trainer_ticks = 0
        self.undo_enabled = bool(MODES[mode].get("undo", False))
        self._undo_stack.clear()
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
            self.game.cfg.soft_drop_factor = self.cfg.input.sdf

    def restart(self) -> None:
        self.start_game()

    def _undo(self) -> None:
        """Zen undo: restore the snapshot taken when the piece you just
        placed spawned — the board, queue, hold and score go back to just
        before that placement, with the piece back in your hands."""
        if not self._undo_stack:
            return
        self.game = self._undo_stack.pop()
        self._last_undo_active = self.game.active
        self._spawn_snapshot = self.game.clone()
        self._undo_pieces = self.game.pieces_placed
        self._note("undo")
        self.renderer.add_popup("UNDO")

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
        return True

    def handle_key(self, key: int, event: pygame.event.Event) -> bool:
        if self.state == self.STATE_SETTINGS:
            return self._settings_key(key, event)
        keys = self.cfg.keys
        if key in parse_key_names(keys.screenshot):
            self.screenshot()
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
        for field, btn in _GAME_BTNS:
            if key in parse_key_names(getattr(keys, field)):
                self.controller.release(btn)
                if self.keylog is not None and in_game:
                    self.keylog.key(btn, False)

    def route_game_key(self, key: int) -> None:
        keys = self.cfg.keys
        c = self.controller
        for field, btn in _GAME_BTNS:
            if key in parse_key_names(getattr(keys, field)):
                c.press(btn)
                if self.keylog is not None:
                    self.keylog.key(btn, True)

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

        for ev in game.events:
            if ev.get("kind") == "clear":
                self.renderer.add_popup(ev["label"], ev.get("attack", 0))
            elif ev.get("kind") == "tspin":
                self.renderer.add_popup(ev["label"], 0)

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
            self.renderer.draw_game_over(self.game, self.modes[self.mode_idx])
        else:
            self.renderer.draw(self.game, self.modes[self.mode_idx],
                               paused=(self.state == self.STATE_PAUSE),
                               undo_hint=self.undo_enabled)
        pygame.display.flip()

    def screenshot(self) -> None:
        name = f"tetris_{time.strftime('%Y%m%d_%H%M%S')}.png"
        pygame.image.save(self.screen, name)
        print(f"saved {name}")


def main() -> None:
    App().run()


if __name__ == "__main__":
    main()
