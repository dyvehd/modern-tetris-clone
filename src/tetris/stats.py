"""Advanced race stats (Jstris+ definitions, ported).

Formulas from Jstris+ `src/stats.js` (github.com/JstrisPlus/
jstris-plus-userscript, MIT), verified against the source; the mapping
to our engine:

    Jstris+                ours
    -------------------     ------------------------
    placedBlocks            Game.pieces_placed
    gamedata.attack         Game.attack_sent
    gamedata.garbageCleared Game.cheese_dug
    totalLines              Game.cfg.goal_lines
    linesCleared/linesLeft  cheese_dug / goal - dug
    clock (seconds)         Game.seconds

APP and PPD are all-mode stats there; the two paces are cheese/dig-only
(``enabledMode: 3``) — here that means a cheese mode with a line goal,
the trainer being the canonical case. Every formula divides by a
possibly-zero progress counter, so each returns ``"-"`` until meaningful
(their ``replaceBadValues`` shows 0; a dash reads better and never
implies a wrong 0.000).
"""

from __future__ import annotations

from .engine.game import Game


def _fmt_3(x: float) -> str:
    return f"{x:.3f}"


def blocks_used(game: Game) -> int:
    """Pieces placed so far (Jstris's "blocks" = tetrominoes)."""
    return game.pieces_placed


def app(game: Game) -> str:
    """Attack Per Piece: attack sent / pieces placed."""
    if game.pieces_placed <= 0:
        return "-"
    return _fmt_3(game.attack_sent / game.pieces_placed)


def ppd(game: Game) -> str:
    """Pieces Per Downstack-line: pieces placed / garbage lines dug."""
    if game.cheese_dug <= 0:
        return "-"
    return _fmt_3(game.pieces_placed / game.cheese_dug)


def _cheese_goal(game: Game) -> tuple[int, int] | None:
    """(total goal lines, lines dug) when the pace stats apply — a cheese
    mode with a line goal (Jstris+'s enabledMode 3 dig modes)."""
    if not game.cfg.cheese_rows or not game.cfg.goal_lines:
        return None
    return (game.cfg.goal_lines, game.cheese_dug)


def block_pace(game: Game) -> str:
    """Forecast of the TOTAL pieces needed to finish the race:
    (lines left / lines dug) * pieces placed + pieces placed."""
    ctx = _cheese_goal(game)
    if ctx is None:
        return "-"
    total, dug = ctx
    if dug <= 0:
        return "-"
    lines_left = max(0, total - dug)
    return str(int((lines_left / dug) * game.pieces_placed + game.pieces_placed))


def time_pace(game: Game) -> str:
    """Forecast of the total time to finish the race:
    (total lines / lines dug) * elapsed."""
    ctx = _cheese_goal(game)
    if ctx is None:
        return "-"
    total, dug = ctx
    if dug <= 0:
        return "-"
    seconds = (total / dug) * game.seconds
    m, s = divmod(int(seconds), 60)
    cs = int((seconds - int(seconds)) * 100)
    return f"{m:02d}:{s:02d}.{cs:02d}"


def advanced_rows(game: Game) -> list[tuple[str, str]]:
    """The advanced stats block, ready for the left-column HUD:

    APP always; PPD and the two paces only in cheese-with-goal modes
    (their Jstris+ mode gate). "Blocks used" is a basic stat (it sits in
    the main HUD rows, not here).
    """
    rows = [("APP", app(game))]
    if _cheese_goal(game) is not None:
        rows += [
            ("PPD", ppd(game)),
            ("BLOCK PACE", block_pace(game)),
            ("TIME PACE", time_pace(game)),
        ]
    return rows
