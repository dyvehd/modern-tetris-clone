"""Cheese-race harness tests.

The heart is the level-1 ground truth: extracted from the movegen (not
hand-derived — my initial "any pointy piece digs any hole" claim was wrong
about S/Z edge columns), pinned as seeds, and asserted against both
application modes. Direct and navigated runs must agree on every
cheese-relevant outcome; the navigated one replays real inputs through the
engine, end-to-end validating harness + movegen + pathfinder + engine.
"""

import math
from statistics import fmean

import pytest

from tetris.ai import (
    CheeseEnv,
    Decision,
    GreedyDigAgent,
    InvalidDecision,
    RandomAgent,
    run_batch,
    run_episode,
    apply_placement,
    count_holes,
    enumerate_placements,
    stack_height,
)
from tetris.ai.cheese import BatchResult
from tetris.ai.movegen import Placement
from tetris.engine import board as B
from tetris.engine.constants import FIELD_H, FIELD_W, PieceType
from tetris.engine.game import Action, Game


# ground truth: which hole columns each piece can dig in one placement -------
# (level 1 = one cheese row with one hole)


def _flat_cells(piece: PieceType, rot: int) -> list[tuple[int, int]]:
    """(row, col) cell offsets of a piece rotation, for synthetic placements."""
    from tetris.engine.constants import PIECE_CELLS

    return [(cy, cx) for cx, cy in PIECE_CELLS[piece][rot]]


def _one_line_rows(hole: int) -> list[int]:
    rows = [0] * FIELD_H
    rows[FIELD_H - 1] = B.garbage_row(hole)
    return rows


def _diggable_holes(piece: PieceType) -> set[int]:
    out = set()
    for hole in range(10):
        board = _one_line_rows(hole)
        for p in enumerate_placements(board, piece):
            _, lines = apply_placement(board, p)
            if lines >= 1:
                out.add(hole)
                break
    return out


def test_diggability_table():
    # pinned: I/J/L/T dig any hole, S misses col 0, Z misses col 9, O alone
    # digs nothing. A vertical S/Z pointy cell sits on its left/right column
    # respectively, so one board edge is always out of reach.
    assert _diggable_holes(PieceType.I) == set(range(10))
    assert _diggable_holes(PieceType.J) == set(range(10))
    assert _diggable_holes(PieceType.L) == set(range(10))
    assert _diggable_holes(PieceType.T) == set(range(10))
    assert _diggable_holes(PieceType.S) == set(range(1, 10))
    assert _diggable_holes(PieceType.Z) == set(range(0, 9))
    assert _diggable_holes(PieceType.O) == set()


# level-1 optimality ----------------------------------------------------------


def test_level1_greedy_uses_one_piece_when_possible():
    # over a 500-seed sweep, greedy wins with exactly 1 piece on every
    # episode where a single piece can win (either the active or the piece
    # a hold would bring out digs the hole), and wins every episode.
    env = CheeseEnv(level=1)
    agent = GreedyDigAgent()
    dig = {
        PieceType.I: set(range(10)),
        PieceType.J: set(range(10)),
        PieceType.L: set(range(10)),
        PieceType.T: set(range(10)),
        PieceType.S: set(range(1, 10)),
        PieceType.Z: set(range(0, 9)),
        PieceType.O: set(),
    }

    def one_possible(game: Game) -> bool:
        hole = [x for x in range(10) if not (game.rows[FIELD_H - 1] >> x & 1)][0]
        # hold is empty at spawn: holding brings queue[1]
        return hole in dig[game.queue[0]] or hole in dig[game.queue[1]]

    wins = one_piece = possible = 0
    for seed in range(500):
        game = Game(env.game_config(), seed=seed)
        if one_possible(game):
            possible += 1
        result = run_episode(agent, env, seed, navigate=False)
        if result.won:
            wins += 1
            if result.pieces == 1:
                one_piece += 1
    assert possible > 400  # the sweep genuinely covers the 1-piece region
    assert one_piece == possible  # optimal wherever 1 piece suffices
    assert wins == 500


def test_level1_greedy_can_one_piece_non_O():
    # pinned diggable starts: 1 piece in both modes
    env = CheeseEnv(level=1)
    for piece_name, seed in [("I", 100000), ("J", 100013), ("T", 100005)]:
        result = run_episode(GreedyDigAgent(), env, seed, navigate=False)
        assert result.first_piece.name == piece_name
        assert result.won and result.pieces == 1
    navigated = run_episode(GreedyDigAgent(), env, 100000, navigate=True)
    assert navigated.won and navigated.pieces == 1
    assert navigated.inputs is not None and len(navigated.inputs) == 1


def test_level1_greedy_holds_to_dig():
    # seed 500002: first piece O, hold brings a dig-anywhere T -> swap, dig
    env = CheeseEnv(level=1)
    game = Game(env.game_config(), seed=500002)
    assert game.queue[0] is PieceType.O and game.queue[1] is PieceType.T
    result = run_episode(GreedyDigAgent(), env, 500002, navigate=False)
    assert result.won and result.pieces == 1
    d = result.decisions[0]
    assert d.hold and d.placement.piece is PieceType.T


def test_level1_pinned_impossible1_edges():
    # seeds where no single placement clears the line (verified by
    # exhaustive enumeration): greedy finds the true 2-piece minimum
    env = CheeseEnv(level=1)
    # hole 0, active S (can't reach col 0), hold brings O (can't dig)
    r = run_episode(GreedyDigAgent(), env, 400012, navigate=False)
    assert (r.first_piece, r.pieces, r.reason) == (PieceType.S, 2, "cleared")
    # hole 9, active Z (can't reach col 9), hold brings O
    r = run_episode(GreedyDigAgent(), env, 401439, navigate=False)
    assert (r.first_piece, r.pieces, r.reason) == (PieceType.Z, 2, "cleared")
    # hole 0, active O, hold brings S (can't reach col 0), queue then L:
    # 2 pieces suffice (S as a step, L digs) but greedy's 1-ply myopia
    # costs one piece — 3, not 2. The gap search must close.
    r = run_episode(GreedyDigAgent(), env, 401938, navigate=False)
    assert (r.first_piece, r.pieces, r.reason) == (PieceType.O, 3, "cleared")


def test_level1_optimum_stats_pinned():
    # 500-episode sweep pinned: 99.2% win in 1 piece, the rest are the S/Z/O
    # edge-column 2-piece minimums (larger sweeps in the 20k run: 99.0%)
    env = CheeseEnv(level=1)
    batch = run_batch(GreedyDigAgent(), env, 500, seed0=200000)
    assert batch.win_rate == 1.0
    assert batch.pieces.count(1) == 496
    assert batch.pieces.count(2) == 4
    assert max(batch.pieces) <= 7


# harness mechanics -----------------------------------------------------------


def test_direct_and_navigated_equivalence():
    # every cheese-relevant outcome must agree across application modes
    env = CheeseEnv(level=10)
    for seed in range(8):
        direct = run_episode(GreedyDigAgent(), env, seed, navigate=False)
        nav = run_episode(GreedyDigAgent(), env, seed, navigate=True)
        assert (direct.pieces, direct.dug, direct.won, direct.reason) == (
            nav.pieces, nav.dug, nav.won, nav.reason,
        )
    # navigated results carry the per-piece input log, ending in hard drops
    nav = run_episode(GreedyDigAgent(), env, 2, navigate=True)
    assert nav.inputs is not None
    assert len(nav.inputs) == nav.pieces
    assert all(path[-1] is Action.HARD_DROP for path in nav.inputs)
    # direct results carry no input log (the fast path)
    direct = run_episode(GreedyDigAgent(), env, 2, navigate=False)
    assert direct.inputs is None


def test_determinism():
    # same agent + seed = identical result, either mode
    env = CheeseEnv(level=10)
    for navigate in (False, True):
        a = run_episode(GreedyDigAgent(), env, 7, navigate=navigate)
        b = run_episode(GreedyDigAgent(), env, 7, navigate=navigate)
        assert a == b
    # a seeded random agent is also reproducible
    r1 = run_episode(RandomAgent(42), env, 7, navigate=False)
    r2 = run_episode(RandomAgent(42), env, 7, navigate=False)
    assert r1 == r2


def test_topout_and_cap_accounting():
    env = CheeseEnv(level=10)
    # greedy tops out on some seeds (pinned: seed 0) — reason "topout", dug < level
    r = run_episode(GreedyDigAgent(), env, 0, navigate=False)
    assert r.reason == "topout" and not r.won and r.dug < env.level
    # a tiny piece cap turns an unfinished episode into "capped" without
    # raising — failures are outcomes, not errors
    capped_env = CheeseEnv(level=100, piece_cap=5)
    r = run_episode(GreedyDigAgent(), capped_env, 2, navigate=False)
    assert r.reason == "capped" and not r.won and r.pieces == 5


def test_invalid_decisions_raise():
    env = CheeseEnv(level=10)

    class WrongPiece:
        name = "wrong"

        def decide(self, obs):
            # a placement for a piece that is neither active nor holdable:
            # I when neither is I (level-10 boards start non-I often; find a
            # seed where the guarantee holds)
            piece = PieceType.I if obs.active is not PieceType.I else PieceType.J
            return Decision(Placement(
                piece=piece, rot=0, x=0, y=36, spin="none",
                cells=tuple(sorted((36 + cy, cx) for cx, cy in _flat_cells(piece, 0))),
            ))

    with pytest.raises(InvalidDecision):
        run_episode(WrongPiece(), env, 0, navigate=False)

    class FloatingAgent:
        name = "floating"

        def decide(self, obs):
            # a resting-looking position floating mid-air: direct mode must
            # reject it (collides=False above and below = not resting)
            return Decision(Placement(
                piece=obs.active, rot=0, x=0, y=30, spin="none",
                cells=tuple(sorted((30 + cy, cx) for cx, cy in _flat_cells(obs.active, 0))),
            ))

    with pytest.raises(InvalidDecision):
        run_episode(FloatingAgent(), env, 0, navigate=False)


def test_hold_disabled():
    env = CheeseEnv(level=1, hold_enabled=False)
    game = Game(env.game_config(), seed=500002)
    assert game.queue[0] is PieceType.O  # O with no hold: 1 piece impossible
    r = run_episode(GreedyDigAgent(), env, 500002, navigate=False)
    assert r.won and r.pieces == 2  # O placed as junk, then the T digs


def test_hold_disabled_obs_reports_no_hold():
    # with hold off, obs.can_hold is False so agents never offer a hold
    # (and a hold offer would raise InvalidDecision anyway)
    env = CheeseEnv(level=1, hold_enabled=False)

    class HoldSpy:
        name = "hold-spy"

        def decide(self, obs):
            assert obs.can_hold is False
            return Decision(obs.placements()[0])

    run_episode(HoldSpy(), env, 500002, navigate=False)


def test_hold_disabled_invalid_hold_raises():
    env = CheeseEnv(level=10, hold_enabled=False)

    class HoldAnyway:
        name = "hold-anyway"

        def decide(self, obs):
            return Decision(obs.placements()[0], hold=True)

    with pytest.raises(InvalidDecision):
        run_episode(HoldAnyway(), env, 0, navigate=False)


def test_invalid_hold_mismatch_raises():
    # decision.hold=True but the placement is for the active piece, not the
    # piece the hold would bring out
    env = CheeseEnv(level=10)

    class Mismatch:
        name = "mismatch"

        def decide(self, obs):
            return Decision(obs.placements()[0], hold=True)

    with pytest.raises(InvalidDecision):
        run_episode(Mismatch(), env, 0, navigate=False)


def test_obs_is_immutable_snapshot():
    # the observation must not let an agent mutate engine state through it
    from tetris.ai.cheese import run_episode as _run  # noqa: F401

    captured = []

    class Spy:
        name = "spy"

        def decide(self, obs):
            captured.append(obs)
            try:
                obs.rows[0] |= 1  # type: ignore[index]
                assert False, "rows should be immutable"
            except TypeError:
                pass
            try:
                obs.queue.append(obs.queue[0])  # type: ignore[union-attr]
                assert False, "queue should be immutable"
            except AttributeError:
                pass
            return Decision(obs.placements()[0])

    env = CheeseEnv(level=10)
    run_episode(Spy(), env, 3, navigate=False)
    assert captured


def test_env_validation():
    with pytest.raises(ValueError):
        CheeseEnv(level=0)
    with pytest.raises(ValueError):
        CheeseEnv(stack=0)
    with pytest.raises(ValueError):
        CheeseEnv(messiness=101)
    with pytest.raises(ValueError):
        CheeseEnv(previews=0)
    with pytest.raises(ValueError):
        CheeseEnv(piece_cap=0)


# board helpers ---------------------------------------------------------------


def test_apply_placement_pure():
    # build the dig placement from the real movegen to stay honest about
    # geometry (a vertical I at origin x=3 occupies column 5)
    rows = _one_line_rows(5)
    dig = [
        p for p in enumerate_placements(rows, PieceType.I)
        if apply_placement(rows, p)[1] >= 1
    ][0]
    assert dig.piece is PieceType.I
    merged, lines = apply_placement(rows, dig)
    assert lines == 1
    # the cheese row is gone; the three I cells that stuck out above it
    # remain, shifted down onto the floor
    assert merged == [0] * (FIELD_H - 3) + [1 << 5] * 3
    assert rows[FIELD_H - 1] != 0  # original untouched


def test_count_holes_and_height():
    # a hole is an empty cell with any filled cell above it in its column.
    rows = [0] * FIELD_H
    rows[FIELD_H - 1] = B.garbage_row(5)
    assert count_holes(rows) == 0  # bottom row: nothing covers its hole
    rows[FIELD_H - 2] = 1 << 5  # a pillar directly over the cheese hole
    assert count_holes(rows) == 1
    rows[FIELD_H - 4] = 1 << 2  # a floating cell at col 2, two empty rows under
    assert count_holes(rows) == 3
    assert stack_height(rows) == 4  # occupied rows counted from the floor


# batch math -----------------------------------------------------------------


def test_batch_summary_math():
    # exact arithmetic on a synthetic batch: 3 wins (10/12/14 pieces), 2
    # topouts, 1 cap — mean 12, stdev 2, CI 1.96*2/sqrt(3)
    batch = BatchResult(
        agent_name="synthetic",
        env="L=10 stack=9 mess=100% refill=jstris hold=on",
        level=10,
        seeds=(0, 1, 2, 3, 4, 5),
        pieces=(10, 50, 12, 14, 60, 400),
        reasons=("cleared", "topout", "cleared", "cleared", "topout", "capped"),
    )
    assert batch.episodes == 6
    assert batch.wins == 3
    assert batch.win_rate == 0.5
    assert batch.mean_pieces == pytest.approx(12.0)
    assert batch.ci95_pieces == pytest.approx(1.96 * 2 / math.sqrt(3))
    assert batch.mean_pieces_per_line == pytest.approx(1.2)
    text = batch.summary()
    assert "episodes 6" in text and "wins 3 (50.0%)" in text
    assert "12.00" in text and "pieces/line 1.200" in text
    # a single win: no stdev, no CI, but a mean exists
    one = BatchResult(
        agent_name="x", env="e", level=2, seeds=(0,), pieces=(7,),
        reasons=("cleared",),
    )
    assert one.mean_pieces == 7.0
    assert one.ci95_pieces is None
    assert one.mean_pieces_per_line == pytest.approx(3.5)


def test_batch_real_run_matches_math():
    # a small real batch: the properties stay consistent with the raw arrays
    env = CheeseEnv(level=10)
    batch = run_batch(GreedyDigAgent(), env, 10, seed0=0, navigate=False)
    assert batch.episodes == 10
    assert len(batch.pieces) == len(batch.reasons) == len(batch.seeds) == 10
    assert batch.wins == sum(1 for r in batch.reasons if r == "cleared")
    wins = [p for p, r in zip(batch.pieces, batch.reasons) if r == "cleared"]
    if len(wins) >= 2:
        assert batch.mean_pieces == pytest.approx(fmean(wins))
    assert "greedy-dig" in batch.summary()


def test_batch_no_wins_summary():
    # an env nothing can clear: level 1000 with a tiny cap — no wins, no stats
    env = CheeseEnv(level=1000, piece_cap=3)
    batch = run_batch(GreedyDigAgent(), env, 10, seed0=0)
    assert batch.wins == 0
    assert batch.mean_pieces is None
    assert "no wins" in batch.summary()
