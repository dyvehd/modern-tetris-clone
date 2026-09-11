"""Trainer tests: annotation math, advisor threading, backends, and the
trainer state machine end-to-end.

The native-backend tests (misamino / fusion / cold-clear) skip cleanly
when their shared libraries are not built (``bots/build_bots.sh``) — the
trainer is fully functional with the pure-Python cheese-beam backend
alone, and CI / a fresh clone must not fail on the missing artifacts.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pytest  # noqa: E402

pygame = pytest.importorskip("pygame")  # noqa: E402

from tetris.ai import enumerate_placements
from tetris.ai.pathfinder import find_path
from tetris.config import make_mode_config, load_config
from tetris.engine.constants import PieceType
from tetris.engine.game import Action, Game
from tetris.trainer import annotation as ann
from tetris.trainer.advisor import Advisor, piece_token
from tetris.trainer.annotation import AccuracyStats, annotate
from tetris.trainer.backends import (
    BotAdvice,
    BotCandidate,
    PlanStep,
    make_backend,
    available_backends,
    pin_best,
)
from tetris.trainer.trainer import Trainer, TrainerConfig


def trainer_game(seed: int = 99) -> Game:
    cfg = load_config()
    rules, _ = make_mode_config("100L Cheese Trainer", cfg)
    rules.soft_drop_factor = float("inf")
    g = Game(rules, seed=seed)
    g.tick()
    return g


# ------------------------------------------------------------------ annotation


def _advice(scores: list[float], top: int = 0, piece: PieceType = PieceType.I) -> BotAdvice:
    """Synthetic advice: candidates with piece cells derived from their rank."""
    cells_by_rank = [
        ((30, 0), (30, 1), (30, 2), (30, 3)),
        ((30, 4), (30, 5), (30, 6), (30, 7)),
        ((31, 0), (31, 1), (31, 2), (31, 3)),
        ((31, 4), (31, 5), (31, 6), (31, 7)),
    ]
    cands = [
        BotCandidate(piece=piece, hold=False, cells=cells_by_rank[i % 4], score=s)
        for i, s in enumerate(scores)
    ]
    return BotAdvice(piece=piece, hold=False, cells=cells_by_rank[top],
                     candidates=tuple(sorted(cands, key=lambda c: -c.score)))


class TestAnnotation:
    def test_top_move_is_best(self):
        adv = _advice([100, 50, 10])
        q = annotate(adv, adv.cells, False)
        assert q is not None
        assert q.rank == 0 and q.label == ann.LABEL_BEST and q.z_gap == 0.0

    def test_second_ranked_is_scored(self):
        adv = _advice([100, 50, 10])
        q = annotate(adv, adv.candidates[1].cells, False)
        assert q is not None
        assert q.rank == 1 and q.label != ann.LABEL_BEST and q.z_gap > 0

    def test_unknown_placement_returns_none(self):
        adv = _advice([100, 50, 10])
        assert annotate(adv, ((5, 5), (5, 6), (5, 7), (5, 8)), False) is None

    def test_no_candidates_returns_none(self):
        adv = BotAdvice(piece=PieceType.I, hold=False, cells=((30, 0), (30, 1), (30, 2), (30, 3)))
        assert annotate(adv, adv.cells, False) is None

    def test_degenerate_sigma(self):
        # all candidates identical: any gap is a blunder, not a div-by-zero
        adv = _advice([10, 10, 10])
        q = annotate(adv, ((31, 0), (31, 1), (31, 2), (31, 3)), False)
        assert q is not None
        assert q.label == ann.LABEL_BEST  # zero gap == the top move's score

    def test_accuracy_stats_accumulate(self):
        adv = _advice([100, 90, 50, 10])
        stats = AccuracyStats()
        q0 = annotate(adv, adv.candidates[0].cells, False)
        q2 = annotate(adv, adv.candidates[2].cells, False)
        assert q0 and q2
        stats.record(q0)
        stats.record(q2)
        assert stats.n == 2 and stats.best_moves == 1
        assert stats.best_rate == 0.5
        assert stats.accuracy > 0

    def test_labels_thresholds(self):
        assert ann.label_for(0.0) == ann.LABEL_BEST
        assert ann.label_for(0.5) == ann.LABEL_GOOD
        assert ann.label_for(1.5) == ann.LABEL_INACCURACY
        assert ann.label_for(3.0) == ann.LABEL_MISTAKE
        assert ann.label_for(5.0) == ann.LABEL_BLUNDER


# ------------------------------------------------------------------ advisor


class TestAdvisor:
    def test_advice_arrives_and_is_token_keyed(self):
        be = make_backend("cheese-beam")
        g = trainer_game()
        adv = Advisor(be)
        try:
            tok = piece_token(g)
            assert tok
            adv.request(g)
            deadline = time.time() + 10
            while time.time() < deadline and adv.advice_for(tok) is None:
                time.sleep(0.01)
            got = adv.advice_for(tok)
            assert got is not None
            assert got.piece is not None
            # a different token gets no advice (token-keyed, never stale)
            assert adv.advice_for(()) is None
        finally:
            adv.close()

    def test_worker_survives_backend_exception(self):
        class Exploding:
            name = "boom"

            def think(self, game):
                raise RuntimeError("boom")

            def close(self):
                pass

        adv = Advisor(Exploding())
        g = trainer_game()
        try:
            adv.request(g)
            deadline = time.time() + 5
            while time.time() < deadline and adv.advice_for(piece_token(g)) is None:
                time.sleep(0.01)
            # the exception became a None advice, published for the token
            assert adv.advice_for(piece_token(g)) is None
            # and the worker is still alive (next request would be served)
            assert adv._worker.is_alive()
        finally:
            adv.close()


class TestPlanLookahead:
    """Lookahead = the bot's PLAN depth: rank 0 is the placement it will
    play; deeper ranks are its intended placements for the coming pieces
    (simulated through the real engine between thinking steps)."""

    def _planned(self, depth: int, timeout: float = 60.0):
        be = make_backend("cheese-beam")
        adv = Advisor(be)
        g = trainer_game()
        tok = piece_token(g)
        adv.request(g, depth)
        deadline = time.time() + timeout
        while time.time() < deadline and (
            adv.advice_for(tok) is None or len(adv.plan_for(tok)) < depth
        ):
            time.sleep(0.02)
        return adv, g, tok

    def test_plan_covers_requested_depth(self):
        adv, g, tok = self._planned(3)
        try:
            advice = adv.advice_for(tok)
            plan = adv.plan_for(tok)
            assert advice is not None
            assert len(plan) >= 2, f"plan too short for depth 3: {plan}"
            # step 0 is what the bot plays (same placement the automove uses)
            assert plan[0].key == advice.key
            for step in plan:
                assert len(set(step.cells)) == 4
                assert all(0 <= r <= 39 and 0 <= c <= 9 for r, c in step.cells)
            # the plan is about FUTURE pieces: beyond step 0, each step's
            # piece must be one of the bot's remaining sources (the queue
            # or a hold-out hold)
            sources = set(g.queue[:7])
            if g.hold_type is not None:
                sources = sources | {g.hold_type}
            for step in plan[1:]:
                assert step.piece in sources, (
                    f"plan step piece {step.piece} is not a future piece"
                )
        finally:
            adv.close()

    def test_trainer_shadows_are_plan_steps(self):
        from tetris.trainer.trainer import PlannedPlacement

        tr = Trainer(TrainerConfig(lookahead=2))
        g = trainer_game()
        try:
            deadline = time.time() + 90
            token = piece_token(g)
            while time.time() < deadline:
                tr.tick(g, list(g.events))
                advice = tr.advisor.advice_for(token)
                if advice is not None and len(tr.advisor.plan_for(token)) >= 2:
                    break
                time.sleep(0.02)
            tr.last_advice = tr.advisor.advice_for(token)
            tr._token = token
            shadows = tr.shadow_placements(g)
            assert shadows, "no shadows"
            assert shadows[0][1] == 0
            assert set(shadows[0][0].cells) == set(tr.last_advice.cells)
            # deeper ranks are FUTURE placement steps (plan semantics), not
            # alternatives of the active piece
            for shape, rank, hold in shadows[1:]:
                assert rank >= 1
                assert isinstance(shape, PlannedPlacement)
                assert hold is False
        finally:
            tr.close()


# ------------------------------------------------------------------ backends


class TestPinBest:
    """Regression: a backend's ranked candidates can disagree with its own
    chooser (root rescore vs deep search) — the shadow engine and the
    annotation both assume candidates[0] IS the move that gets played."""

    def test_chosen_move_lifted_to_rank0(self):
        advice = _advice([30.0, 20.0, 10.0], top=2)
        pinned = pin_best(advice)
        top = pinned.candidates[0]
        assert (top.piece, top.hold, tuple(sorted(top.cells))) == advice.key
        assert top.score == 30.0  # lifted to the previous top's score
        assert len(pinned.candidates) == 3
        assert pinned.piece == advice.piece

    def test_noop_when_rank0_already_matches(self):
        advice = _advice([30.0, 20.0, 10.0], top=0)
        assert pin_best(advice) is advice

    def test_chosen_move_outside_candidates_stays_unpinned(self):
        advice = _advice([30.0, 20.0], top=0, piece=PieceType.I)
        ghost = BotAdvice(piece=PieceType.T, hold=False, cells=advice.cells,
                          candidates=advice.candidates)
        assert pin_best(ghost).candidates == advice.candidates

    def test_annotation_sees_played_move_as_best(self):
        advice = pin_best(_advice([30.0, 20.0, 10.0], top=1))
        q = annotate(advice, advice.cells, False)
        assert q is not None
        assert q.rank == 0
        assert q.label == ann.LABEL_BEST


def test_cold_clear_plan_steps_are_free_lookahead():
    if not _native_available("cold-clear"):
        pytest.skip("cold-clear library not built (bots/build_bots.sh)")
    g = trainer_game()
    be = make_backend("cold-clear")
    try:
        advice = be.think(g)
        steps = be.plan_steps(advice)
        assert advice is not None and steps is not None and steps
        # step 0 IS the chosen move; deeper steps (when its search has
        # them) are free plan lookahead for the advisor
        assert steps[0].key == advice.key
    finally:
        be.close()


NATIVE_BACKENDS = ["misamino", "fusion", "cold-clear", "zetris", "blockfish"]


def _native_available(name: str) -> bool:
    try:
        make_backend(name)
        return True
    except Exception:
        return False


@pytest.mark.parametrize("name", NATIVE_BACKENDS)
def test_native_backend_advises_on_cheese(name):
    if not _native_available(name):
        pytest.skip(f"{name} library not built (bots/build_bots.sh)")
    g = trainer_game()
    be = make_backend(name)
    advice = be.think(g)
    assert advice is not None, f"{name} returned no advice"
    assert advice.piece is not None
    assert len(set(advice.cells)) == 4
    for r, c in advice.cells:
        assert 0 <= r < 40 and 0 <= c < 10, f"{name} placed outside the field"


@pytest.mark.parametrize("name", NATIVE_BACKENDS)
def test_native_backend_plays_reachable_moves(name):
    """The full trainer contract: every backend's placements exist in OUR
    movegen (what the shadows promise is what the pathfinder can reach)."""
    if not _native_available(name):
        pytest.skip(f"{name} library not built (bots/build_bots.sh)")
    g = trainer_game()
    be = make_backend(name)
    try:
        for _ in range(6):
            advice = be.think(g)
            assert advice is not None
            if advice.hold:
                g.tick([Action.HOLD])
            placements = enumerate_placements(list(g.rows), advice.piece)
            p = next((p for p in placements if set(p.cells) == set(advice.cells)), None)
            assert p is not None, (
                f"{name} advised an unreachable placement: {advice.piece.name} {advice.cells}"
            )
            path = find_path(list(g.rows), advice.piece, p)
            assert path is not None, f"{name}: no path to {sorted(advice.cells)}"
            for a in path:
                g.tick([a])
            if g.over:
                break
    finally:
        be.close()


def test_registry_lists_built_backends():
    names = available_backends()
    assert "cheese-beam" in names  # pure Python: always available
    for n in names:
        assert n in ("cheese-beam", *NATIVE_BACKENDS)


def test_blockfish_advises_after_player_hold():
    """The HOLD state contract: our engine hides the active piece on a hold
    (hold slot occupied, hold locked for the new active) — blockfish must
    still advise, about the piece actually in play, without asking for a
    second hold (regression: shim fed blockfish the wrong queue and the
    backend hid the occupied hold slot)."""
    if not _native_available("blockfish"):
        pytest.skip("blockfish library not built (bots/build_bots.sh)")
    be = make_backend("blockfish")
    try:
        g = trainer_game()
        be.think(g)  # any pre-state; ensure hold slot known-good
        g2 = trainer_game()
        g2.tick([Action.HOLD])
        assert g2.can_hold is False
        assert g2.hold_type is not None
        advice = pin_best(be.think(g2))
        assert advice is not None, "no advice after player hold"
        assert advice.hold is False, "blockfish asked for a locked hold"
        assert advice.piece is g2.active.type, (
            f"advice about {advice.piece.name}, active is {g2.active.type.name}"
        )
        placements = enumerate_placements(list(g2.rows), advice.piece)
        assert any(set(p.cells) == set(advice.cells) for p in placements)
    finally:
        be.close()


def test_blockfish_advice_survives_midgame_hold():
    """Multi-decision contract: advice keeps arriving once the hold slot is
    occupied (regression: the shim's replay queue diverged from
    blockfish's state after the first hold, killing every later answer)."""
    if not _native_available("blockfish"):
        pytest.skip("blockfish library not built (bots/build_bots.sh)")
    be = make_backend("blockfish")
    try:
        g = trainer_game(seed=1234)
        held_once = False
        for _ in range(12):
            if g.active is None or g.over:
                break
            advice = pin_best(be.think(g))
            assert advice is not None, "advice died midgame"
            placements = enumerate_placements(list(g.rows), advice.piece)
            p = next(
                (p for p in placements if set(p.cells) == set(advice.cells)),
                None,
            )
            assert p is not None, f"unreachable placement {advice.cells}"
            # exercise both branches: force one hold along the way
            if not held_once and advice.hold:
                held_once = True
                g.tick([Action.HOLD])
            p = next(
                (p for p in enumerate_placements(list(g.rows), advice.piece)
                 if set(p.cells) == set(advice.cells)),
            )
            g.active.rot, g.active.x, g.active.y = p.rot, p.x, p.y
            g.tick([Action.HARD_DROP])
            g.tick()
        assert held_once, "seed never exercised the hold branch"
    finally:
        be.close()


# ------------------------------------------------------------------ trainer


class TestTrainer:
    def _trained(self, **cfg_over) -> tuple[Trainer, Game]:
        tr = Trainer(TrainerConfig(**cfg_over))
        g = trainer_game()
        return tr, g

    def _wait_advice(self, tr: Trainer, g: Game, timeout: float = 10.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline and tr.last_advice is None:
            tr.tick(g, list(g.events))
            time.sleep(1 / 60)
        assert tr.last_advice is not None, "no advice arrived"

    def test_shadows_are_own_placements(self):
        tr, g = self._trained()
        try:
            self._wait_advice(tr, g)
            shadows = tr.shadow_placements(g)
            assert shadows, "advice ready but no shadows"
            for placement, rank, hold in shadows:
                assert placement.piece is not None
                assert set(placement.cells)
                assert rank < 1 + tr.cfg.lookahead
            # rank 0 is the advice's own placement
            top = shadows[0][0]
            assert set(top.cells) == set(tr.last_advice.cells)
        finally:
            tr.close()

    def test_shadows_off_when_hidden(self):
        tr, g = self._trained(shadows_on=False)
        try:
            self._wait_advice(tr, g)
            assert tr.shadow_placements(g) == []
        finally:
            tr.close()

    def test_ai_off_disables_everything(self):
        tr, g = self._trained(ai_on=False)
        try:
            self._wait_advice(tr, g)
            assert tr.shadow_placements(g) == []
            tr.cfg.automove = True
            for _ in range(120):
                tr.tick(g, [])
            assert g.pieces_placed == 0, "automove played with AI off"
        finally:
            tr.close()

    def test_automove_plays_pieces(self):
        tr, g = self._trained(automove=True, automove_pps=4.0)
        try:
            self._wait_advice(tr, g)
            deadline = time.time() + 15
            while time.time() < deadline and g.pieces_placed < 3 and not g.over:
                tr.tick(g, list(g.events))
                time.sleep(1 / 60)
            assert g.pieces_placed >= 3, "automove did not play 3 pieces"
        finally:
            tr.close()

    def test_step_mode_moves_on_request(self):
        tr, g = self._trained(automove=True, step_mode=True)
        try:
            self._wait_advice(tr, g)
            placed0 = g.pieces_placed
            for _ in range(60):  # no step requested: nothing happens
                tr.tick(g, list(g.events))
            assert g.pieces_placed == placed0
            tr.request_step()
            steps = 0
            while steps < 300 and g.pieces_placed == placed0:
                tr.tick(g, list(g.events))
                steps += 1
            assert g.pieces_placed > placed0, "step request did not play a move"
        finally:
            tr.close()

    def test_annotation_records_player_placements(self):
        tr, g = self._trained(shadows_on=False)
        try:
            self._wait_advice(tr, g)
            # simulate the player placing the bot's own move
            advice = tr.last_advice
            tr.tick(g, [{"kind": "lock", "piece": advice.piece,
                         "cells": advice.cells, "tspin": "none"}])
            assert tr.stats.n == 1
            assert tr.last_quality is not None
            assert tr.last_quality.rank == 0
            assert tr.last_quality.label == ann.LABEL_BEST
        finally:
            tr.close()

    def test_live_feedback_undoes_wrong_moves(self):
        tr, g = self._trained(shadows_on=False, live_feedback=True)
        undone = []

        def hook():
            undone.append(1)
            return True

        tr.set_undo_hook(hook)
        try:
            self._wait_advice(tr, g)
            advice = tr.last_advice
            # the bot's own move: no undo
            tr.tick(g, [{"kind": "lock", "piece": advice.piece,
                         "cells": advice.cells, "tspin": "none"}])
            assert not undone
            # a wrong placement: undo fires
            wrong = next(
                p for p in enumerate_placements(list(g.rows), advice.piece)
                if set(p.cells) != set(advice.cells)
            )
            tr._token = piece_token(g)  # keep the same decision point
            tr.tick(g, [{"kind": "lock", "piece": advice.piece,
                         "cells": wrong.cells, "tspin": "none"}])
            assert undone, "live feedback did not undo a wrong placement"
            assert tr.last_quality is not None
            assert tr.last_quality.label != ann.LABEL_BEST
        finally:
            tr.close()

    def test_live_feedback_tolerates_bot_move_when_disabled(self):
        tr, g = self._trained(shadows_on=False, live_feedback=False)
        undone = []
        tr.set_undo_hook(lambda: (undone.append(1), True)[1])
        try:
            self._wait_advice(tr, g)
            advice = tr.last_advice
            wrong = next(
                p for p in enumerate_placements(list(g.rows), advice.piece)
                if set(p.cells) != set(advice.cells)
            )
            tr.tick(g, [{"kind": "lock", "piece": advice.piece,
                         "cells": wrong.cells, "tspin": "none"}])
            assert not undone, "feedback fired while disabled"
        finally:
            tr.close()

    def test_backend_switch_recovers_advice(self):
        """Regression: switching backends mid-game must deliver fresh
        advice (the token must be re-requested)."""
        tr, g = self._trained()
        try:
            self._wait_advice(tr, g)
            for name in ["misamino", "fusion", "cold-clear", "cheese-beam"]:
                if not _native_available(name):
                    continue
                tr.set_backend(name)
                self._wait_advice(tr, g, timeout=10)
                assert tr.last_advice is not None, f"no advice after switching to {name}"
        finally:
            tr.close()

    def test_restart_clears_stats(self):
        tr, g = self._trained()
        try:
            self._wait_advice(tr, g)
            advice = tr.last_advice
            tr.tick(g, [{"kind": "lock", "piece": advice.piece,
                         "cells": advice.cells, "tspin": "none"}])
            assert tr.stats.n == 1
            tr.on_restart()
            assert tr.stats.n == 0
            assert tr.last_quality is None
            assert tr.last_advice is None
        finally:
            tr.close()


# ---------------------------------------------- config / HUD / renderer


class TestTrainerConfig:
    def test_toggle_flips_each_named_switch(self):
        from tetris.trainer.trainer import _TOGGLE_ATTR

        cfg = TrainerConfig()
        for name, attr in _TOGGLE_ATTR.items():
            before = getattr(cfg, attr)
            assert cfg.toggle(name) is (not before)
            assert getattr(cfg, attr) is (not before)
            # and back
            assert cfg.toggle(name) is before
            assert getattr(cfg, attr) is before

    def test_toggle_unknown_name_raises(self):
        with pytest.raises(KeyError):
            TrainerConfig().toggle("nope")


class TestTrainerHud:
    def _bare(self, **cfg_over) -> Trainer:
        """A Trainer with no advisor thread — hud_lines only reads cfg/stats."""
        tr = Trainer.__new__(Trainer)
        tr.cfg = TrainerConfig(**cfg_over)
        tr.stats = AccuracyStats()
        tr.last_quality = None
        return tr

    def test_automove_pace_folds_into_mode_row(self):
        rows = dict(self._bare(automove=True, step_mode=True, automove_pps=3.5).hud_lines())
        assert "PPS" not in rows  # no seventh row (it would overflow the panel)
        assert rows["AI"] == "automove·step 3.5pps"

    def test_row_count_stays_within_panel_budget(self):
        from tetris.trainer.annotation import MoveQuality

        tr = self._bare(automove=True, automove_pps=2.0)
        q = MoveQuality(placed_cells=(), matched=True, hold=False, rank=0,
                        n_candidates=3, z_gap=0.0, label="best",
                        best_cells=(), best_hold=False)
        tr.last_quality = q
        tr.stats.record(q)
        rows = tr.hud_lines()
        assert len(rows) <= 6, rows
        assert dict(rows)["LAST MOVE"] == "best (1/3)"


class TestTrainerRendering:
    def test_shadow_outline_uses_piece_color(self):
        from tetris.ai.movegen import Placement
        from tetris.engine.constants import PIECE_COLORS, VISIBLE_TOP
        from tetris.render.renderer import Renderer

        surf = pygame.Surface((960, 780))
        r = Renderer(surf)
        cells = tuple((VISIBLE_TOP + 2 + i, 0) for i in range(4))
        pl = Placement(piece=PieceType.T, rot=0, x=0, y=VISIBLE_TOP + 2,
                       spin="none", cells=cells)
        r.draw_ai_shadows([(pl, 0, False)])
        py = r.field_y + (cells[0][0] - VISIBLE_TOP) * r.cell
        got = surf.get_at((r.field_x + 4, py + 4))[:3]
        assert got == PIECE_COLORS[PieceType.T], (got, PIECE_COLORS[PieceType.T])

    def test_panel_rows_fit_on_screen(self):
        from tetris.render.renderer import Renderer

        surf = pygame.Surface((960, 780))
        r = Renderer(surf)
        seen: list[tuple[str, int]] = []
        r.text = lambda s, x, y, *a, **k: seen.append((s, y))  # noqa: ARG005
        rows = [
            ("AI", "automove·step 2pps"), ("MODEL", "cheese-beam"),
            ("SHADOWS", "4"), ("FEEDBACK", "off"),
            ("LAST MOVE", "best (1/1)"), ("ACCURACY", "0.00z / 100% top"),
        ]
        r.draw_trainer_panel(rows)
        assert seen[0][0] == "AI TRAINER"
        assert max(y for _, y in seen) + 18 <= surf.get_height()


class TestTrainerKeybinds:
    def test_function_keys_toggle_without_crashing(self):
        """Regression: F1-F5 crashed on TrainerConfig having no toggle()."""
        from tetris.app import App
        from tetris.config import AppConfig, DebugConfig

        app = App(AppConfig(debug=DebugConfig(log_input=False)))
        app.mode_idx = app.modes.index("100L Cheese Trainer")
        app.start_game()
        tr = app.ai_trainer
        assert tr is not None
        try:
            cfg = tr.cfg
            before = (cfg.automove, cfg.step_mode, cfg.live_feedback)
            for k in (pygame.K_F1, pygame.K_F2, pygame.K_F3, pygame.K_F4, pygame.K_F5,
                      pygame.K_F6, pygame.K_F7, pygame.K_F8, pygame.K_F9, pygame.K_F10):
                ev = pygame.event.Event(pygame.KEYDOWN, key=k)
                assert app.handle_key(k, ev) is True, f"F-key {k} not consumed"
            assert cfg.live_feedback is not before[2]  # F3
            assert cfg.automove is not before[0]       # F4
            assert cfg.step_mode is not before[1]      # F5
        finally:
            tr.close()


class TestStepModePlaysTheTopShadow:
    """Regression (user report): the drawn shadow must be exactly the
    placement step mode plays — played opposite placements were possible
    whenever a backend's ranked list disagreed with its own chooser."""

    def test_step_lock_matches_rank0_shadow(self):
        from tetris.app import App
        from tetris.config import AppConfig, DebugConfig

        app = App(AppConfig(debug=DebugConfig(log_input=False)))
        app.mode_idx = app.modes.index("100L Cheese Trainer")
        app.start_game()
        tr = app.ai_trainer
        assert tr is not None
        try:
            tr.cfg.automove = True
            tr.cfg.step_mode = True
            tr.cfg.lookahead = 2
            done = tried = 0
            session = time.time() + 240
            while time.time() < session and tried < 6 and done < 2:
                # wait for advice + a plan for the CURRENT decision point
                d = time.time() + 60
                while time.time() < d:
                    app.logic_tick()
                    tok = piece_token(app.game)
                    if (tok and tr.last_advice is not None
                            and len(tr.advisor.plan_for(tok)) >= 2):
                        break
                    time.sleep(1 / 240)
                tok = piece_token(app.game)
                if not (tr.last_advice is not None and tr.advisor.plan_for(tok)):
                    break
                shadows = tr.shadow_placements(app.game)
                if not shadows:
                    break
                aim = shadows[0][0]
                advice = tr.last_advice
                if advice.hold:
                    brings = (app.game.hold_type
                              if app.game.hold_type is not None
                              else app.game.queue[0])
                    if brings is not advice.piece:
                        tried += 1
                        continue  # hold advice our movement model can't play
                start = app.game.pieces_placed
                tr.request_step()
                d = time.time() + 20
                lock: list[dict] = []
                while time.time() < d:
                    app.logic_tick()
                    if app.game.pieces_placed > start:
                        lock = [e for e in app.game.events if e.get("kind") == "lock"]
                        break
                    time.sleep(1 / 240)
                assert lock, "step request did not place a piece"
                assert set(lock[-1]["cells"]) == set(aim.cells), (
                    "the bot played a placement other than the top shadow"
                )
                done += 1
            assert done >= 2, f"only {done} verified drops"
        finally:
            tr.close()


# ------------------------------------------------------------------ engine


class TestLockEvent:
    def test_lock_event_carries_cells(self):
        g = trainer_game()
        g.tick([Action.HARD_DROP])
        lock = [e for e in g.events if e.get("kind") == "lock"]
        assert len(lock) == 1
        assert lock[0]["piece"] is not None
        cells = lock[0]["cells"]
        assert len(set(cells)) == 4
        for r, c in cells:
            assert 0 <= r < 40 and 0 <= c < 10

    def test_lock_event_before_clear(self):
        """The trainer ranks placements, so the event must fire even when
        the placement clears lines (cells pre-clear). A hand-built board:
        one row short of full with a 1-wide gap, an I piece above it."""
        from tetris.engine.constants import FULL_ROW
        cfg = load_config()
        rules, _ = make_mode_config("Zen 0G", cfg)
        g = Game(rules, seed=1)
        g.tick()
        # rows 36-39 full except a 1-wide shaft at column 5 through them
        for y in range(36, 40):
            g.rows[y] = FULL_ROW ^ (1 << 5)
        g.spawn_forced(PieceType.I)
        # navigate the I into the shaft with our own pathfinder contract:
        # vertical at column 5
        placements = enumerate_placements(list(g.rows), PieceType.I)
        p = next(
            pl for pl in placements
            if set(pl.cells) == {(36, 5), (37, 5), (38, 5), (39, 5)}
        )
        for a in find_path(list(g.rows), PieceType.I, p):
            g.tick([a])
        lock = [e for e in g.events if e.get("kind") == "lock"]
        assert lock, "no lock event"
        assert any(r == 39 for r, _ in lock[0]["cells"])
        assert [e for e in g.events if e.get("kind") == "clear"], "expected a line clear"
