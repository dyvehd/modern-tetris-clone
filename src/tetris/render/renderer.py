"""pygame-ce renderer: playfield, ghost, hold/next, HUD, popups, screens.

Draw methods only compose onto ``self.screen``; the caller owns
``pygame.display.flip()`` (one flip per frame).
"""

from __future__ import annotations

import pygame

from ..engine import board as B
from ..engine.constants import (
    FIELD_W,
    GARBAGE_COLOR,
    PIECE_CELLS,
    PIECE_COLORS,
    PieceType,
    VISIBLE_H,
    VISIBLE_TOP,
)
from ..engine.game import Game
from ..stats import advanced_rows

BG = (13, 16, 21)
PANEL = (24, 29, 37)
PANEL_EDGE = (44, 52, 64)
GRID = (33, 40, 50)
BORDER = (86, 95, 108)
TEXT = (214, 221, 230)
TEXT_DIM = (120, 130, 144)
ACCENT = (86, 204, 242)
GARBAGE_METER = (196, 74, 74)
B2B_COLOR = (240, 178, 62)
COMBO_COLOR = (240, 106, 106)

POPUP_TTL = 70  # display frames (~1.2 s at 60 fps)


def _dim(color: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return tuple(min(255, int(c * factor)) for c in color)  # type: ignore[return-value]


class Renderer:
    def __init__(self, screen: pygame.Surface, cell: int = 30, buffer_rows: int = 2) -> None:
        self.screen = screen
        self.cell = cell
        self.buffer_rows = buffer_rows
        self.field_px_w = FIELD_W * cell
        self.field_px_h = VISIBLE_H * cell
        self.buffer_px_h = buffer_rows * cell
        # field top-left (leaving room above for the buffer rows)
        self.field_x = 340
        self.field_y = 40 + self.buffer_px_h
        self._font_cache: dict[int, pygame.font.Font] = {}
        self.popups: list[dict] = []
        # next-queue box rect, published for hit-testing (queue editor)
        self.next_rect: pygame.Rect | None = None

    # -------------------------------------------------------------- picking

    def cell_at(self, pos) -> tuple[int, int] | None:
        """The (row, col) a screen position falls on, for the mouse editor.

        Covers the visible field plus the drawn buffer strip; None anywhere
        else (including just outside the field).
        """
        px, py = pos
        fx, c = self.field_x, self.cell
        if not fx <= px < fx + self.field_px_w:
            return None
        top = self.field_y
        if top <= py < top + self.field_px_h:
            return VISIBLE_TOP + (py - top) // c, (px - fx) // c
        if top - self.buffer_px_h <= py < top:
            row = VISIBLE_TOP - self.buffer_rows + (py - (top - self.buffer_px_h)) // c
            return int(row), (px - fx) // c
        return None

    def hit_next_box(self, pos) -> bool:
        return self.next_rect is not None and self.next_rect.collidepoint(pos)

    # ------------------------------------------------------------------ util

    def font(self, size: int, bold: bool = False) -> pygame.font.Font:
        key = (size, bold)
        if key not in self._font_cache:
            f = pygame.font.SysFont("consolas,menlo,dejavusansmono,monospace", size, bold=bold)
            self._font_cache[key] = f
        return self._font_cache[key]

    def text(self, s: str, x: int, y: int, size: int = 18, color=TEXT, bold: bool = False, align: str = "left") -> None:
        surf = self.font(size, bold).render(s, True, color)
        rect = surf.get_rect()
        if align == "left":
            rect.topleft = (x, y)
        elif align == "right":
            rect.topright = (x, y)
        else:
            setattr(rect, align, (x, y))
        self.screen.blit(surf, rect)

    def piece_color(self, piece: PieceType | None):
        return GARBAGE_COLOR if piece is None else PIECE_COLORS[piece]

    def draw_cell(self, px: int, py: int, size: int, color, ghost: bool = False, dim_factor: float = 1.0) -> None:
        rect = pygame.Rect(px, py, size, size)
        if ghost:
            pygame.draw.rect(self.screen, _dim(color, dim_factor), rect, width=2, border_radius=3)
            return
        pygame.draw.rect(self.screen, _dim(color, dim_factor), rect, border_radius=2)
        top = pygame.Rect(px + 1, py + 1, size - 2, max(2, size // 6))
        pygame.draw.rect(self.screen, _dim(color, min(1.35, dim_factor + 0.35)), top, border_radius=2)

    def draw_matrix(self, piece: PieceType, px: int, py: int, size: int, ghost: bool = False, dim: float = 1.0) -> None:
        color = self.piece_color(piece)
        for cx, cy in PIECE_CELLS[piece][0]:
            self.draw_cell(px + cx * size, py + cy * size, size, color, ghost, dim)

    def draw_mini_piece(self, piece: PieceType, center_x: int, center_y: int, size: int, dim: float = 1.0) -> None:
        """Draw a piece in spawn orientation, centered on (center_x, center_y)."""
        cells = PIECE_CELLS[piece][0]
        min_cx = min(x for x, _ in cells)
        max_cx = max(x for x, _ in cells)
        min_cy = min(y for _, y in cells)
        max_cy = max(y for _, y in cells)
        ox = center_x - (max_cx - min_cx + 1) * size / 2 - min_cx * size
        oy = center_y - (max_cy - min_cy + 1) * size / 2 - min_cy * size
        color = self.piece_color(piece)
        for x, y in cells:
            self.draw_cell(int(ox + x * size), int(oy + y * size), size, color, dim_factor=dim)

    # ----------------------------------------------------------------- frame

    def draw(self, game: Game, mode: str, paused: bool = False, undo_hint: bool = False,
             edit: bool = False, hover=None, ai_shadows=None) -> None:
        self.screen.fill(BG)
        self.draw_field(game, hover=hover if edit else None)
        if ai_shadows:
            self.draw_ai_shadows(ai_shadows)
        self.draw_side_panels(game, bag_separators=edit)
        self.draw_stats(game, mode)
        self.draw_popups()
        if paused:
            self.draw_pause(undo_hint)

    AI_HINT = (150, 200, 255)  # fallback tint for trainer chrome (panel header)

    def draw_ai_shadows(self, shadows) -> None:
        """Draw the bot's plan as corner-tick outlines.

        Rank 0 is the placement the bot will actually play (drawn at
        full strength); deeper ranks are its planned placements for the
        coming pieces, faded by plan depth. Every shadow uses its own
        piece's color, so a glance tells you which piece goes where. The
        ticks stay visually distinct from the player's ghost, which is a
        solid full-cell border.
        """
        c = self.cell
        fx, fy = self.field_x, self.field_y
        for placement, rank, hold in shadows:
            fade = max(0.45, 1.0 - 0.14 * rank)
            color = _dim(self.piece_color(placement.piece), fade)
            for ry, cx in placement.cells:
                px = fx + cx * c
                py = fy + (ry - VISIBLE_TOP) * c
                t = max(3, c // 5)  # corner tick length
                m = 2
                pts = (
                    ((px + m, py + m + t), (px + m, py + m), (px + m + t, py + m)),
                    ((px + c - m - t, py + m), (px + c - m, py + m), (px + c - m, py + m + t)),
                    ((px + c - m, py + c - m - t), (px + c - m, py + c - m), (px + c - m - t, py + c - m)),
                    ((px + m + t, py + c - m), (px + m, py + c - m), (px + m, py + c - m - t)),
                )
                for tri in pts:
                    pygame.draw.polygon(self.screen, color, tri)
            if hold and rank == 0:
                # the hold flag on the top move: a small dot in each cell
                for ry, cx in placement.cells:
                    pygame.draw.circle(
                        self.screen, color,
                        (fx + cx * c + c // 2, fy + (ry - VISIBLE_TOP) * c + c // 2),
                        max(2, c // 8),
                    )

    def draw_trainer_panel(self, rows: list[tuple[str, str]]) -> None:
        """The AI trainer status panel (right side, under the next queue).

        Anchored to the bottom of the right column: if the row list grows
        (live stats add two rows), the panel moves up rather than spilling
        off the bottom of the window.
        """
        x = 700
        line_h = 18
        top = self.field_y + 5 * int(round(3.04 * self.cell)) + 90
        bottom = self.field_y + self.field_px_h + 74
        y = min(top, bottom - (22 + line_h * len(rows)))
        self.text("AI TRAINER", x, y, 15, self.AI_HINT, bold=True)
        y += 22
        for label, value in rows:
            self.text(label, x, y, 13, TEXT_DIM, bold=True)
            self.text(value, x + 110, y, 13, TEXT)
            y += line_h

    def cell_style(self, game: Game, ry: int, x: int):
        """Color of a locked cell from the engine's style grid."""
        v = game.styles[ry][x]
        return GARBAGE_COLOR if v < 0 else PIECE_COLORS[PieceType(v)]

    def draw_field(self, game: Game, hover=None) -> None:
        c = self.cell
        fx, fy = self.field_x, self.field_y

        # buffer rows (dimmed) above the visible field
        pygame.draw.rect(
            self.screen, PANEL,
            pygame.Rect(fx - 4, fy - self.buffer_px_h - 4, self.field_px_w + 8, self.buffer_px_h + 4),
            border_radius=4,
        )
        for ry in range(VISIBLE_TOP - self.buffer_rows, VISIBLE_TOP):
            if ry < 0:
                continue
            row = game.rows[ry]
            for x in range(FIELD_W):
                if row >> x & 1:
                    self.draw_cell(fx + x * c, fy - (VISIBLE_TOP - ry) * c, c,
                                   self.cell_style(game, ry, x), dim_factor=0.45)

        # visible field
        pygame.draw.rect(self.screen, PANEL, pygame.Rect(fx - 4, fy - 4, self.field_px_w + 8, self.field_px_h + 8), border_radius=4)
        pygame.draw.rect(self.screen, BORDER, pygame.Rect(fx - 4, fy - 4, self.field_px_w + 8, self.field_px_h + 8), width=2, border_radius=4)
        # skyline: top of the visible field
        pygame.draw.line(self.screen, PANEL_EDGE, (fx - 4, fy), (fx + self.field_px_w + 4, fy), 1)

        for i in range(VISIBLE_H):
            ry = VISIBLE_TOP + i
            row = game.rows[ry]
            py = fy + i * c
            for x in range(FIELD_W):
                if row >> x & 1:
                    self.draw_cell(fx + x * c, py, c, self.cell_style(game, ry, x))

        if game.active is not None:
            p = game.active
            gy = B.ghost_y(game.rows, p.type, p.rot, p.x, p.y)
            if gy != p.y:
                for cx, cy in PIECE_CELLS[p.type][p.rot]:
                    self.draw_cell(fx + (p.x + cx) * c, fy + (gy + cy - VISIBLE_TOP) * c, c,
                                   self.piece_color(p.type), ghost=True, dim_factor=0.8)
            for cx, cy in PIECE_CELLS[p.type][p.rot]:
                px, py = fx + (p.x + cx) * c, fy + (p.y + cy - VISIBLE_TOP) * c
                if p.y + cy >= VISIBLE_TOP:
                    self.draw_cell(px, py, c, self.piece_color(p.type))
                else:
                    self.draw_cell(px, fy - (VISIBLE_TOP - (p.y + cy)) * c, c, self.piece_color(p.type), dim_factor=0.45)

        if hover is not None:
            ry, x = hover
            py = fy + (ry - VISIBLE_TOP) * c
            pygame.draw.rect(self.screen, TEXT, pygame.Rect(fx + x * c, py, c, c),
                             width=2, border_radius=2)

        self.draw_garbage_meter(game)

    def draw_garbage_meter(self, game: Game) -> None:
        pending = sum(len(b.rows) for b in game.garbage_queue)
        if not pending:
            return
        px = self.field_x - 16
        h = min(pending * 6, self.field_px_h)
        pygame.draw.rect(
            self.screen, GARBAGE_METER,
            pygame.Rect(px, self.field_y + self.field_px_h - h, 6, h),
            border_radius=2,
        )

    def draw_side_panels(self, game: Game, bag_separators: bool = False) -> None:
        c = self.cell

        # hold ------------------------------------------------------------
        # Same convention as the next queue: preview pieces are drawn at
        # full board-cell size (Jstris does this too).
        hx, hy = 80, 70
        c = self.cell
        self.text("HOLD", hx, hy - 26, 16, TEXT_DIM, bold=True)
        pygame.draw.rect(self.screen, PANEL, pygame.Rect(hx - 10, hy - 10, 4 * c + 20, 2 * c + 20), border_radius=6)
        if game.hold_type is not None:
            color = self.piece_color(game.hold_type)
            dim = 1.0 if game.can_hold else 0.35
            offset = {"I": 0, "O": c}.get(game.hold_type.name, c // 2)
            for cx, cy in PIECE_CELLS[game.hold_type][0]:
                self.draw_cell(hx + offset + cx * c, hy + cy * c, c, color, dim_factor=dim)

        # next -------------------------------------------------------------
        # Sized like Jstris (measured against a real 2026-09-07 screenshot:
        # preview 408 px for a 537 px board, ~0.76 board heights tall, top
        # edge at 0 buffer lines relative to the visible board): pieces at
        # full board-cell size, five slots of ~3 rows, queue top aligned
        # with the top of the visible field.
        size = self.cell
        slot = int(round(3.04 * self.cell))
        nx, ny = 700, self.field_y
        box = pygame.Rect(nx - 12, ny - 6, 4 * size + 24, 5 * slot + 12)
        pygame.draw.rect(self.screen, PANEL, box, border_radius=6)
        pygame.draw.rect(self.screen, PANEL_EDGE, box, width=1, border_radius=6)
        self.text("NEXT", nx, ny - 30, 16, TEXT_DIM, bold=True)
        self.next_rect = box
        for i, piece in enumerate(game.queue[:5]):
            self.draw_mini_piece(piece, nx + 2 * size, ny + i * slot + slot // 2, size)
        if bag_separators:
            # four-tris-style bag separators: a line between preview slots
            # wherever a 7-bag boundary falls (queue[i] is the (bag_pos+i+1)-th
            # piece of its bag, so the boundary is after piece 6-bag_pos)
            for i in range(4):
                if (game.bag_pos + i + 1) % 7 == 0:
                    y = ny + (i + 1) * slot - 8
                    pygame.draw.line(self.screen, BORDER,
                                     (box.left + 4, y), (box.right - 4, y), 2)

        # cheese counter (four-tris/Jstris style, under the next queue) -----
        if game.cfg.cheese_rows:
            cx = nx + 2 * size
            cy = ny + 5 * slot + 34
            if game.cfg.goal_lines is None:
                self.text(str(game.cheese_dug), cx, cy, 38, ACCENT, bold=True, align="center")
                self.text("lines dug", cx, cy + 44, 14, TEXT_DIM, align="center")
            else:
                remaining = max(0, game.cfg.goal_lines - game.cheese_dug)
                self.text(str(remaining), cx, cy, 38, ACCENT, bold=True, align="center")
                self.text("lines remaining", cx, cy + 44, 14, TEXT_DIM, align="center")

    def draw_stats(self, game: Game, mode: str) -> None:
        x = 80
        y = 250
        self.text(mode.upper(), x, y - 40, 18, ACCENT, bold=True)
        # in cheese modes the goal counts dug garbage, not total line clears
        if game.cfg.cheese_rows:
            if game.cfg.goal_lines is None:
                dig = str(game.cheese_dug)
            else:
                dig = f"{game.cheese_dug} / {game.cfg.goal_lines}"
            lines_row = ("DIG", dig)
        else:
            lines_row = ("LINES", str(game.lines) if game.cfg.goal_lines is None
                         else f"{game.lines} / {game.cfg.goal_lines}")
        rows = [
            ("SCORE", f"{game.score:,}"),
            lines_row,
            ("LEVEL", str(game.level)),
            ("TIME", self._fmt_time(game.seconds)),
            ("PPS", f"{game.pieces_placed / game.seconds:.2f}" if game.seconds > 1 else "-"),
            ("BLOCKS", str(game.pieces_placed)),
            ("ATTACK", str(game.attack_sent)),
        ]
        for i, (label, value) in enumerate(rows):
            self.text(label, x, y + i * 40, 14, TEXT_DIM, bold=True)
            self.text(value, x, y + i * 40 + 18, 20, TEXT, bold=True)

        y2 = y + len(rows) * 40 + 10
        if game.b2b_chain >= 2:
            self.text(f"B2B x{game.b2b_chain - 1}", x, y2, 18, B2B_COLOR, bold=True)
            y2 += 26
        if game.combo >= 2:
            self.text(f"{game.combo - 1} COMBO", x, y2, 18, COMBO_COLOR, bold=True)
            y2 += 26

        # advanced race stats (Jstris+ definitions) — compact rows under
        # the banners, above the footer
        adv = advanced_rows(game)
        if adv:
            self.text("ADVANCED", x, 596, 12, ACCENT, bold=True)
            for i, (label, value) in enumerate(adv):
                yy = 618 + i * 22
                self.text(label, x, yy, 12, TEXT_DIM, bold=True)
                self.text(value, x + 118, yy, 14, TEXT, align="right")

        self.text("ESC pause  R restart", x, 720, 14, TEXT_DIM)
        self.text("Q menu  F12 screenshot", x, 740, 14, TEXT_DIM)

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        cs = int((seconds - int(seconds)) * 100)
        return f"{m:02d}:{s:02d}.{cs:02d}"

    # ---------------------------------------------------------------- popups

    def add_popup(self, label: str, attack: int = 0) -> None:
        self.popups.append({"label": label, "attack": attack, "ttl": POPUP_TTL})

    def tick_popups(self) -> None:
        for p in self.popups:
            p["ttl"] -= 1
        self.popups = [p for p in self.popups if p["ttl"] > 0]

    def draw_popups(self) -> None:
        cx = self.field_x + self.field_px_w // 2
        cy = self.field_y + self.field_px_h // 3
        for i, p in enumerate(self.popups):
            age = 1 - p["ttl"] / POPUP_TTL
            alpha = max(0.0, 1.0 - age * 1.4)
            color = B2B_COLOR if p["label"].startswith("B2B") else TEXT
            size = 24 - i * 2
            self._blit_alpha(self.font(size, True).render(p["label"], True, color), (cx, cy + i * 30 - int(age * 14)), alpha)
            if p["attack"] > 0:
                self._blit_alpha(
                    self.font(20, True).render(f"+{p['attack']}", True, COMBO_COLOR),
                    (cx, cy + (i + 1) * 30 - int(age * 14)),
                    alpha,
                )

    def _blit_alpha(self, surf: pygame.Surface, center: tuple[int, int], alpha: float) -> None:
        if alpha <= 0:
            return
        surf = surf.copy()
        surf.set_alpha(int(255 * alpha))
        rect = surf.get_rect(center=center)
        self.screen.blit(surf, rect)

    # ---------------------------------------------------------------- screens

    def draw_menu(self, modes: list[str], selected: int, descs: list[str]) -> None:
        self.screen.fill(BG)
        self.text("MODERN TETRIS", 480, 90, 44, ACCENT, bold=True, align="center")
        self.text("Guideline engine / Jstris-style rules / pygame-ce", 480, 132, 16, TEXT_DIM, align="center")
        step = 44 if len(modes) > 7 else 52
        top = 190
        for i, mode in enumerate(modes):
            color = TEXT if i == selected else TEXT_DIM
            prefix = "> " if i == selected else "  "
            self.text(prefix + mode, 480, top + i * step, 26, color, bold=(i == selected), align="center")
            self.text(descs[i], 480, top + i * step + 26, 14, TEXT_DIM, align="center")
        hint_y = top + len(modes) * step + 28
        self.text("UP/DOWN select   ENTER start   S settings", 480, hint_y, 16, TEXT_DIM, align="center")
        self.text("Arrows move  Z/X/A rotate  SPACE hard drop  C hold", 480, hint_y + 26, 14, TEXT_DIM, align="center")

    def draw_settings(self, items, capture_active: bool) -> None:
        """Draw the settings screen.

        ``items`` rows: (kind, label, value_text, is_selected) where kind is
        "header" | "value" | "bind".
        """
        self.screen.fill(BG)
        self.text("SETTINGS", 480, 70, 34, ACCENT, bold=True, align="center")
        blink = pygame.time.get_ticks() // 400 % 2 == 0
        y = 124
        for kind, label, value, selected in items:
            if kind == "header":
                self.text(label, 260, y, 15, TEXT_DIM, bold=True)
                y += 32
                continue
            marker = "> " if selected else "  "
            color = TEXT if selected else TEXT_DIM
            self.text(marker + label, 260, y, 20, color, bold=selected)
            if capture_active and selected:
                value = "[ press a key ]" if blink else ""
                color = B2B_COLOR
            self.text(value, 740, y, 20, color, bold=selected, align="right")
            y += 30
        self.text(
            "UP/DOWN select   LEFT/RIGHT adjust (Shift = fine)   ENTER rebind",
            480, 660, 14, TEXT_DIM, align="center",
        )
        self.text(
            "BACKSPACE clear bind   R reset defaults   ESC back (auto-saves)",
            480, 684, 14, TEXT_DIM, align="center",
        )

    def draw_pause(self, undo_hint: bool = False) -> None:
        overlay = pygame.Surface(self.screen.get_size())
        overlay.set_alpha(170)
        overlay.fill(BG)
        self.screen.blit(overlay, (0, 0))
        self.text("PAUSED", 480, 320, 40, TEXT, bold=True, align="center")
        line = "ESC resume   R restart   S settings   Q menu"
        if undo_hint:
            line = "Ctrl+Z undo (in game)   " + line
        self.text(line, 480, 380, 18, TEXT_DIM, align="center")

    def draw_queue_dialog(self, qd: dict) -> None:
        """The zen queue editor: a sequence of piece letters + a 7-bag
        offset. ``qd`` keys: seq, off, field (0/1), error."""
        overlay = pygame.Surface(self.screen.get_size())
        overlay.set_alpha(170)
        overlay.fill(BG)
        self.screen.blit(overlay, (0, 0))
        panel = pygame.Rect(480, 360, 520, 300)
        panel.center = (480, 360)
        pygame.draw.rect(self.screen, PANEL, panel, border_radius=8)
        pygame.draw.rect(self.screen, BORDER, panel, width=2, border_radius=8)
        self.text("EDIT QUEUE", 480, panel.top + 24, 26, ACCENT, bold=True, align="center")

        # sequence field: letters drawn in their piece colors
        seq_y = panel.top + 86
        active = qd["field"] == 0
        self.text("sequence", panel.left + 32, seq_y - 26, 14, TEXT_DIM, bold=True)
        cursor = "_" if active and pygame.time.get_ticks() // 400 % 2 == 0 else " "
        letters = (qd["seq"] + cursor)[-24:]
        for i, ch in enumerate(letters):
            color = PIECE_COLORS[PieceType("IJLOSTZ".index(ch))] if ch in "IJLOSTZ" else TEXT
            if ch == cursor.strip() and qd["seq"]:
                color = TEXT
            self.text(ch, panel.left + 32 + i * 20, seq_y, 26, color, bold=True)
        if not qd["seq"]:
            self.text(cursor if cursor.strip() else "_", panel.left + 32, seq_y, 26, TEXT, bold=True)

        # offset field
        off_y = seq_y + 70
        active_off = qd["field"] == 1
        self.text("7-bag offset", panel.left + 32, off_y - 26, 14, TEXT_DIM, bold=True)
        marker = "> " if active_off else "  "
        off_cursor = "_" if active_off and pygame.time.get_ticks() // 400 % 2 == 0 else ""
        self.text(marker + str(qd["off"]) + off_cursor, panel.left + 32, off_y, 26,
                   TEXT if active_off else TEXT_DIM, bold=True)
        seq_marker = "> " if qd["field"] == 0 else "  "
        self.text(seq_marker, panel.left + 8, seq_y, 26, TEXT_DIM, bold=True)
        self.text(f"= pieces already dealt from the current bag ({qd['off']})", panel.left + 100, off_y + 8, 13, TEXT_DIM)

        if qd["error"]:
            self.text(qd["error"], 480, panel.bottom - 74, 16, (239, 99, 99), bold=True, align="center")
        self.text("letters I J L O S T Z   TAB switch field", 480, panel.bottom - 46, 14, TEXT_DIM, align="center")
        self.text("ENTER apply   ESC cancel   empty sequence = random bags", 480, panel.bottom - 26, 14, TEXT_DIM, align="center")

    def draw_game_over(self, game: Game, mode: str, undo_hint: bool = False) -> None:
        overlay = pygame.Surface(self.screen.get_size())
        overlay.set_alpha(200)
        overlay.fill(BG)
        self.screen.blit(overlay, (0, 0))
        title = "PERFECT!" if game.won else "TOP OUT"
        color = B2B_COLOR if game.won else (220, 80, 80)
        self.text(title, 480, 200, 46, color, bold=True, align="center")
        lines = [
            f"TIME      {self._fmt_time(game.seconds)}",
            f"SCORE     {game.score:,}",
            f"LINES     {game.lines}",
        ]
        if game.cfg.cheese_rows:
            lines.append(f"DIGGED    {game.cheese_dug}")
        lines += [
            f"PIECES    {game.pieces_placed}",
            f"PPS       {game.pieces_placed / game.seconds:.2f}" if game.seconds > 1 else "PPS       -",
            f"T-SPINS   {game.tspins}   QUADS {game.quads}   PC {game.perfect_clears}",
            f"ATTACK    {game.attack_sent}",
        ]
        for i, line in enumerate(lines):
            self.text(line, 480, 280 + i * 34, 20, TEXT, align="center")
        # advanced race stats (Jstris+ definitions) on the finish screen
        y3 = 280 + len(lines) * 34 + 18
        for label, value in advanced_rows(game):
            if value == "-":
                continue  # nothing meaningful measured yet
            self.text(f"{label}  {value}", 480, y3, 16, TEXT_DIM, align="center")
            y3 += 24
        bottom = "Ctrl+Z undo    R restart    Q menu" if undo_hint else "R restart    Q menu"
        self.text(bottom, 480, max(560, y3 + 14), 18, TEXT_DIM, align="center")
