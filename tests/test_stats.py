"""Advanced race stats (Jstris+ ports): formula math, zero-guards and
the cheese-with-goal mode gate.
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from conftest import make_game  # noqa: E402

from tetris.stats import advanced_rows, app, block_pace, blocks_used, ppd, time_pace  # noqa: E402


def cheese_game(goal: int | None = 100, cheese_rows: int = 9):
    return make_game(cheese_rows=cheese_rows, goal_lines=goal)


def played_game(pieces: int, dug: int, seconds: float, attack: int = 0,
                goal: int | None = 100, cheese: bool = True):
    g = cheese_game(goal=goal) if cheese else make_game(goal_lines=goal)
    g.pieces_placed = pieces
    g.cheese_dug = dug
    g.attack_sent = attack
    g.tick_count = int(seconds * 60)
    return g


class TestFormulas:
    def test_app(self):
        # 6 attack over 20 pieces = 0.300
        assert app(played_game(20, 10, 30.0, attack=6)) == "0.300"

    def test_ppd(self):
        # 20 pieces for 10 dug lines = 2.000
        assert ppd(played_game(20, 10, 30.0)) == "2.000"

    def test_blocks_used(self):
        assert blocks_used(played_game(37, 12, 40.0)) == 37

    def test_block_pace(self):
        # (90 left / 10 dug) * 20 placed + 20 = 200
        assert block_pace(played_game(20, 10, 30.0)) == "200"

    def test_block_pace_at_finish_forecasts_total(self):
        # at the goal (100/100 dug, 150 pieces) the forecast is exactly the
        # pieces used — the stat collapses to the realized total
        assert block_pace(played_game(150, 100, 240.0)) == "150"

    def test_time_pace(self):
        # (100 / 10) * 61.5s = 615s = 10:15.00
        g = played_game(20, 10, 61.5)
        assert time_pace(g) == "10:15.00"

    def test_time_pace_at_finish_forecasts_total(self):
        # at the goal the forecast is exactly the elapsed time
        g = played_game(150, 100, 123.45)
        assert time_pace(g) == "02:03.45"


class TestGuards:
    def test_zero_pieces_app_is_dash(self):
        assert app(played_game(0, 0, 0.0)) == "-"

    def test_zero_dug_ppd_is_dash(self):
        assert ppd(played_game(20, 0, 30.0)) == "-"

    def test_zero_dug_paces_are_dash(self):
        g = played_game(20, 0, 30.0)
        assert block_pace(g) == "-"
        assert time_pace(g) == "-"


class TestModeGate:
    def test_cheese_with_goal_shows_all(self):
        rows = dict(advanced_rows(played_game(20, 10, 30.0)))
        assert set(rows) == {"APP", "PPD", "BLOCK PACE", "TIME PACE"}

    def test_marathon_shows_app_only(self):
        # no cheese: PPD/pace make no sense (Jstris+ gates them to dig)
        g = played_game(20, 10, 30.0, cheese=False)
        rows = dict(advanced_rows(g))
        assert set(rows) == {"APP"}

    def test_endless_cheese_shows_app_only(self):
        # cheese without a goal: no total to forecast against
        g = played_game(20, 10, 30.0, goal=None)
        rows = dict(advanced_rows(g))
        assert set(rows) == {"APP"}


class TestAdvancedRowsContract:
    def test_rows_are_label_value_pairs(self):
        for label, value in advanced_rows(played_game(20, 10, 30.0)):
            assert isinstance(label, str) and isinstance(value, str)


# ------------------------------------------------------------- rendering


class TestStatsRendering:
    """Layout regressions: the HUD's left column must fit its lane (no
    play-field overlap, footer clear of the advanced block) and the
    game-over screen must show the advanced stats without collisions."""

    def _game(self, pieces=20, dug=12, seconds=34.0, won=False):
        g = cheese_game()
        g.pieces_placed = pieces
        g.cheese_dug = dug
        g.tick_count = int(seconds * 60)
        g.combo = 3  # worst case: both banners + all advanced rows shown
        g.b2b_chain = 3
        if won:
            g.over = g.won = True
        return g

    def _collect(self, draw):
        """Draw with Renderer.text spied on, returning (label, rect) pairs."""
        import pygame

        from tetris.render.renderer import Renderer

        pygame.init()  # the spy renders fonts before any real draw call
        surf = pygame.Surface((960, 780))
        r = Renderer(surf)
        seen: list[tuple[str, pygame.Rect]] = []

        def spy(s, x, y, size=18, color=None, bold=False, align="left"):
            rr = r.font(size, bold).render(s, True, color or (255, 255, 255)).get_rect()
            if align == "left":
                rr.topleft = (x, y)
            elif align == "right":
                rr.topright = (x, y)
            else:
                setattr(rr, align, (x, y))
            seen.append((s, rr))

        r.text = spy  # type: ignore[method-assign]
        draw(r)
        return seen

    def test_hud_advanced_block_fits_left_lane(self):
        from tetris.render.renderer import Renderer

        g = self._game()
        seen = self._collect(
            lambda r: r.draw_stats(g, "100L Cheese Trainer")
        )
        W = 960
        for s, rect in seen:
            assert 0 <= rect.left and rect.right <= 330, f"{s!r} enters the field lane"
            assert rect.bottom <= 780, f"{s!r} leaves the window"
        # the advanced rows exist and stay clear of the footer lane
        adv = [s for s, rect in seen if rect.top >= 596]
        assert any(s.startswith(("APP", "PPD", "BLOCK PACE", "TIME PACE")) for s in adv)
        for s, rect in seen:
            if s.startswith(("APP", "PPD", "BLOCK PACE", "TIME PACE")):
                assert rect.bottom <= 720, f"{s!r} collides with the footer"
        footer = [s for s, rect in seen if rect.top >= 714]
        assert set(footer) <= {"ESC pause  R restart", "Q menu  F12 screenshot"}
        # basic row BLOCKS present (the new basic stat)
        assert any(s == "BLOCKS" for s, _ in seen)
        _ = Renderer  # noqa: F841  (import kept for parity with other tests)

    def test_hud_marathon_shows_only_app_row(self):
        g = self._game()
        g.cfg.cheese_rows = 0  # marathon: pace/PPD gated off
        g.pieces_placed = 20
        seen = self._collect(lambda r: r.draw_stats(g, "Marathon"))
        adv_labels = [
            s for s, rect in seen
            if 596 <= rect.top < 714 and rect.left == 80  # labels at x=80, footer excluded
        ]
        assert adv_labels == ["ADVANCED", "APP"], adv_labels

    def test_game_over_shows_advanced_and_no_collisions(self):
        g = self._game(pieces=150, dug=100, seconds=123.45, won=True)
        seen = self._collect(lambda r: r.draw_game_over(g, "100L Cheese Trainer"))
        labels = [s for s, _ in seen]
        # the four advanced stats render, after the base stats
        assert "APP  0.000" in labels
        assert any(s.startswith("PPD  ") for s in labels)
        assert any(s.startswith("BLOCK PACE  150") for s in labels)
        assert any(s.startswith("TIME PACE  ") for s in labels)
        # nothing runs off the window and no two texts collide
        for i, (s, rect) in enumerate(seen):
            assert rect.right <= 960 and rect.bottom <= 780, f"{s!r} out of window"
            for s2, rect2 in seen[i + 1:]:
                assert not rect.colliderect(rect2), f"{s!r} collides with {s2!r}"
        # the controls line sits below the advanced block
        adv_bottom = max(
            rect.bottom for s, rect in seen if s.startswith(("APP", "PPD", "BLOCK", "TIME"))
        )
        ctrl_top = min(rect.top for s, rect in seen if "restart" in s)
        assert ctrl_top >= adv_bottom
