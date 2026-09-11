"""Game configuration and mode presets, plus optional settings.toml override.

Everything here is plain Python (no pygame) so the config can be loaded in
headless/AI contexts too. A ``settings.toml`` next to the repo root (or in
``~/.config/modern-tetris-clone/``) may override any field, e.g.::

    [rules]
    lock_delay_ms = 500
    soft_drop_factor = 20

    [input]
    das_ms = 133
    arr_ms = 0
"""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, replace, fields
from pathlib import Path

from .engine.game import GameConfig
from .input.controller import InputConfig


@dataclass
class RenderConfig:
    cell: int = 30  # playfield cell size in px
    buffer_rows_shown: int = 2  # buffer rows drawn above the visible field
    next_count: int = 5
    ghost: bool = True
    show_grid: bool = True


@dataclass
class Keybinds:
    # comma-separated pygame key names, lowercased (see app.py for mapping)
    left: str = "left"
    right: str = "right"
    soft_drop: str = "down"
    hard_drop: str = "space"
    rotate_cw: str = "up,x"
    rotate_ccw: str = "z,ctrl"
    rotate_180: str = "a"
    hold: str = "c,shift"
    restart: str = "r"
    pause: str = "escape,p"
    screenshot: str = "f12"


@dataclass
class DebugConfig:
    log_input: bool = True  # write a key/piece event log per session
    input_log_dir: str = ""  # empty -> <repo root>/logs


@dataclass
class AppConfig:
    rules: GameConfig = dataclasses.field(default_factory=GameConfig)
    input: InputConfig = dataclasses.field(default_factory=InputConfig)
    render: RenderConfig = dataclasses.field(default_factory=RenderConfig)
    keys: Keybinds = dataclasses.field(default_factory=Keybinds)
    debug: DebugConfig = dataclasses.field(default_factory=DebugConfig)


# Mode presets --------------------------------------------------------------

MODES: dict[str, dict] = {
    "Marathon": {
        "rules": {"gravity_curve": True, "goal_lines": 150},
        "desc": "Guideline curve gravity, ends at 150 lines",
    },
    "Sprint 40 Lines": {
        "rules": {"gravity_g": 0.02, "goal_lines": 40},
        "desc": "Race to 40 lines (0.02G, like TETR.IO 40L)",
    },
    "Cheese 10": {
        "rules": {"gravity_g": 0.02, "goal_lines": 10, "cheese_rows": 9},
        "desc": "Dig through 10 lines of cheese (Jstris 9-row stack)",
    },
    "Cheese 18": {
        "rules": {"gravity_g": 0.02, "goal_lines": 18, "cheese_rows": 9},
        "desc": "Dig through 18 lines of cheese",
    },
    "Cheese 100": {
        "rules": {"gravity_g": 0.02, "goal_lines": 100, "cheese_rows": 9},
        "desc": "Dig through 100 lines of cheese",
    },
    "100L Cheese Trainer": {
        "rules": {"gravity_g": 0.0, "goal_lines": 100, "cheese_rows": 9},
        "ai_trainer": True,
        "undo": True,
        "desc": "Dig 100 lines, 0G, with AI: shadows, automove, live feedback",
    },
    "Cheese ∞": {
        "rules": {"gravity_g": 0.02, "goal_lines": None, "cheese_rows": 9},
        "desc": "Endless digging — the cheese stack never runs out",
    },
    "Zen": {
        "rules": {"gravity_g": 0.02, "goal_lines": None},
        "undo": True,
        "edit": True,
        "desc": "Endless, slow and calm (Ctrl+Z undo)",
    },
    "Zen 0G": {
        "rules": {"gravity_g": 0.0, "goal_lines": None},
        "undo": True,
        "edit": True,
        "desc": "Endless, zero gravity — pieces stay where you move them (Ctrl+Z undo)",
    },
    "VS Sandbox": {
        "rules": {"gravity_g": 0.02, "goal_lines": None},
        "trainer": True,
        "undo": True,
        "edit": True,
        "desc": "Garbage trainer: cancel/rise like a VS match (Ctrl+Z undo)",
    },
}


def make_mode_config(mode: str, base: AppConfig) -> tuple[GameConfig, bool]:
    preset = MODES[mode]
    overrides = dict(preset.get("rules", {}))
    # a user-set [rules] cheese_rows (e.g. 10, four-tris/Techmino style)
    # wins over the 9-row Jstris preset in the cheese modes — but must not
    # leak into non-cheese modes
    if "cheese_rows" in overrides:
        if base.rules.cheese_rows > 0:
            overrides["cheese_rows"] = base.rules.cheese_rows
    else:
        overrides["cheese_rows"] = 0
    rules = replace(base.rules, **overrides)
    return rules, bool(preset.get("trainer", False))


_SETTINGS_PATHS = (
    # next to the package (repo root for editable installs), then user config
    Path(__file__).resolve().parents[2] / "settings.toml",
    Path.home() / ".config" / "modern-tetris-clone" / "settings.toml",
)
DEFAULT_SETTINGS_PATH = _SETTINGS_PATHS[0]
DEFAULT_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"

_MANAGED_SECTIONS = ("input", "keys")


def _fmt_toml_float(value: float) -> str:
    import math

    if math.isinf(value):
        return "inf"  # TOML 1.0 float infinity; tomllib parses it back
    return repr(float(value))


def save_config(cfg: AppConfig, path: Path | None = None) -> Path:
    """Persist the [input] and [keys] sections to settings.toml.

    Any other sections already present in the file (e.g. a hand-edited
    [rules]) are preserved verbatim.
    """
    path = path or DEFAULT_SETTINGS_PATH
    preserved: list[str] = []
    if path.exists():
        skipping = False
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                skipping = stripped[1:-1].strip() in _MANAGED_SECTIONS
                if skipping:
                    continue
            if not skipping:
                preserved.append(line)
    while preserved and not preserved[-1].strip():
        preserved.pop()

    out = preserved + [
        "",
        "[input]",
        f"das_ms = {_fmt_toml_float(cfg.input.das_ms)}",
        f"arr_ms = {_fmt_toml_float(cfg.input.arr_ms)}",
        f"sdf = {_fmt_toml_float(cfg.input.sdf)}",
        "",
        "[keys]",
    ]
    for field in dataclasses.fields(cfg.keys):
        value = getattr(cfg.keys, field.name)
        out.append(f'{field.name} = "{value}"')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n")
    return path


def load_config() -> AppConfig:
    """Built-in defaults, overridden by settings.toml if present."""
    cfg = AppConfig()
    for path in _SETTINGS_PATHS:
        if not path.exists():
            continue
        try:
            data = tomllib.loads(path.read_text())
        except tomllib.TOMLDecodeError as exc:  # pragma: no cover
            raise SystemExit(f"Invalid settings file {path}: {exc}") from exc
        for section in ("rules", "input", "render", "debug"):
            if section not in data:
                continue
            target = getattr(cfg, section)
            names = {f.name for f in fields(target)}
            for key, value in data[section].items():
                if key not in names:
                    raise SystemExit(f"Unknown setting [{section}] {key}")
                current = getattr(target, key)
                if isinstance(current, float):
                    value = float(value)
                elif isinstance(current, int) and not isinstance(current, bool):
                    value = int(value)
                setattr(target, key, value)
        if "keys" in data:
            for key, value in data["keys"].items():
                if hasattr(cfg.keys, key):
                    setattr(cfg.keys, key, str(value))
        break
    return cfg


def config_to_dict(cfg: AppConfig) -> dict:
    return {
        "rules": dataclasses.asdict(cfg.rules),
        "input": dataclasses.asdict(cfg.input),
        "render": dataclasses.asdict(cfg.render),
    }
