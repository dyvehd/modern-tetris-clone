"""Search-baseline tests: 1-ply and beam agents on the cheese harness.

The money tests are the pinned level-1 edge seeds (401938 etc.): 2-piece
true minimums that greedy's 1-ply myopia misses — the search must find
them. The quiescence rule is what makes that reliable: a no-clear placement
is a leaf at every ply, because under Jstris refill semantics the cheese
tops back up after it with unknowable hole positions, so no future win can
be credited through a combo break (the beam originally lost these seeds by
committing junk placements on the promise of a next-piece dig that the
refill destroyed).
"""

import pytest

from tetris.ai import (
    BeamAgent,
    CheeseEnv,
    GreedyDigAgent,
    OnePlyAgent,
    run_batch,
    run_episode,
)
from tetris.ai.eval import (
    EvalWeights,
    column_heights,
    count_col_transitions,
    count_holes_covered,
    count_row_transitions,
    count_well_cells,
    eval_board,
)
from tetris.ai.search import lock_and_count
from tetris.engine import board as B
from tetris.engine.constants import FIELD_H, PieceType


def _board(*mods) -> list[int]:
    rows = [0] * FIELD_H
    for r, v in mods:
        rows[r] = v
    return rows


# eval features ---------------------------------------------------------------


def test_column_heights():
    # heights measured from the floor: the cheese row raises every column
    # but the hole's; the floating cell at col 2 sits 3 high
    rows = _board((FIELD_H - 1, B.garbage_row(5)), (FIELD_H - 3, 1 << 2))
    assert column_heights(rows) == [1, 1, 3, 1, 1, 0, 1, 1, 1, 1]
    assert column_heights([0] * FIELD_H) == [0] * 10


def test_holes_and_covered():
    # a pillar over the cheese hole: one hole, two covered cells
    rows = _board((FIELD_H - 1, B.garbage_row(5)), (FIELD_H - 2, 1 << 5), (FIELD_H - 3, 1 << 5))
    assert count_holes_covered(rows) == (1, 2)
    # a clean cheese row: no hole above the floor
    assert count_holes_covered(_board((FIELD_H - 1, B.garbage_row(0)))) == (0, 0)


def test_row_col_transitions():
    # one cheese row, hole col 5: along the row, the left wall merges with
    # the fill run (no change at x=0) and the right wall likewise — the only
    # transitions are the two around the hole
    rows = _board((FIELD_H - 1, B.garbage_row(5)))
    assert count_row_transitions(rows) == 2
    # column transitions (floor counts as filled): the hole column reads
    # fill(floor)->empty = 1; every filled column stays 0 until the row's
    # fill ends at the field top = 10 total (1 hole column + 9 filled ones
    # ending at the top edge)
    assert count_col_transitions(rows) == 10


def test_well_cells():
    # a 1-wide shaft between two cheese pillars: 2 deep = 2 well cells
    rows = _board(
        (FIELD_H - 1, B.garbage_row(5)),
        (FIELD_H - 2, (1 << 4) | (1 << 6)),  # walls flanking col 5
    )
    assert count_well_cells(rows) == 2
    # an open surface: no wells
    assert count_well_cells(_board((FIELD_H - 1, 1))) == 0


def test_eval_prefers_dig_over_junk():
    # ordering: a dug (cleared) board > intact cheese > junk placed on it.
    # A cleared board is exempt from the bottom-row-hole penalty (progress,
    # not a hole to dig); junk covers cells and leaves the cheese hole.
    cheese = _board((FIELD_H - 1, B.garbage_row(5)))
    dug = _board()  # the row cleared: empty board
    junk = _board(
        (FIELD_H - 1, B.garbage_row(5)),
        (FIELD_H - 2, 0b1100000000),  # an O sitting over cols 8-9
    )
    assert eval_board(dug) > eval_board(cheese) > eval_board(junk)


def test_lock_and_count_dug_region():
    # dug counting: cleared rows within the bottom cheese region count.
    # Two cheese rows, a vertical I digs the bottom one: lines=1, dug=1,
    # and the upper cheese row shifts down with the I's protruding cells
    from tetris.ai import enumerate_placements

    rows = _board(
        (FIELD_H - 1, B.garbage_row(5)),
        (FIELD_H - 2, B.garbage_row(3)),
    )
    dig = next(
        p for p in enumerate_placements(rows, PieceType.I)
        if lock_and_count(rows, p, 2)[1] == 1
    )
    rows_after, lines, dug = lock_and_count(rows, dig, 2)
    assert (lines, dug) == (1, 1)
    # the I threads the hole-3 shaft of the upper cheese row and completes
    # it (it cannot reach through the bottom row's hole-5 shaft from col 3)
    assert rows_after[FIELD_H - 1] == B.garbage_row(5)  # the bottom cheese remains
    assert rows_after[FIELD_H - 2] != 0  # the I cells above the clear, shifted
    # the placement list itself is untouched (pure)
    assert rows[FIELD_H - 1] == B.garbage_row(5) and rows[FIELD_H - 2] == B.garbage_row(3)


# agents ----------------------------------------------------------------------


def test_edge_seed_two_piece_minimums():
    # the money seeds: hole-0/9 + S/Z/O starts where 1 piece cannot win.
    # Search must find the 2-piece minimum greedy misses (on 401938 greedy
    # burns a third piece; the others it finds by luck).
    env = CheeseEnv(level=1)
    agents = [OnePlyAgent(), BeamAgent(width=40, depth=5), BeamAgent(width=8, depth=3)]
    for seed, first in [(401938, PieceType.O), (400012, PieceType.S), (401439, PieceType.Z)]:
        for agent in agents:
            r = run_episode(agent, env, seed, navigate=False)
            assert (r.first_piece, r.pieces, r.reason) == (first, 2, "cleared"), (
                seed, agent.name, r.pieces,
            )


def test_greedy_still_misses_401938():
    # the documented baseline gap: greedy's 1-ply lexicographic takes 3
    # on 401938 (O junk, L junk, I dig) — the myopia cost search recovers
    env = CheeseEnv(level=1)
    r = run_episode(GreedyDigAgent(), env, 401938, navigate=False)
    assert (r.pieces, r.reason) == (3, "cleared")


def test_level1_search_stays_optimal():
    # the search agents must not regress the level-1 optimum: 1 piece on
    # every seed where 1 piece wins (the greedy sweep pin: 496/500)
    env = CheeseEnv(level=1)
    for agent in [OnePlyAgent(), BeamAgent(width=20, depth=4)]:
        batch = run_batch(agent, env, 500, seed0=200000)
        assert batch.win_rate == 1.0
        assert batch.pieces.count(1) == 496
        assert max(batch.pieces) <= 3  # worst case: the 2-piece edges + 1


def test_level10_win_rate_100():
    # the headline baseline result: eval-driven agents never top out at
    # level 10 (greedy: ~42%), at ~4.9 pieces/line (greedy: ~9.2)
    env = CheeseEnv(level=10)
    for agent in [OnePlyAgent(), BeamAgent(width=20, depth=4)]:
        batch = run_batch(agent, env, 10, seed0=0)
        assert batch.wins == 10, (agent.name, batch.reasons)
        mean = batch.mean_pieces
        assert mean is not None and mean < 60, (agent.name, mean)


def test_beam_beats_1ply_mean_pieces():
    # paired comparison over the same seeds: beam's mean pieces-to-clear is
    # at or below 1-ply's (the beam's edge is small at level 10 — the
    # quiescence leaf keeps it honest — but never worse)
    env = CheeseEnv(level=10)
    one = run_batch(OnePlyAgent(), env, 30, seed0=0)
    beam = run_batch(BeamAgent(width=20, depth=4), env, 30, seed0=0)
    assert beam.mean_pieces is not None and one.mean_pieces is not None
    assert beam.mean_pieces <= one.mean_pieces + 0.5


def test_search_determinism():
    env = CheeseEnv(level=10)
    for agent in [OnePlyAgent(), BeamAgent(width=20, depth=4)]:
        a = run_episode(agent, env, 3, navigate=False)
        b = run_episode(agent, env, 3, navigate=False)
        assert a == b


def test_beam_validation_and_naming():
    with pytest.raises(ValueError):
        BeamAgent(width=0)
    with pytest.raises(ValueError):
        BeamAgent(depth=0)
    assert BeamAgent(width=20, depth=4).name == "beam20x4"
    assert OnePlyAgent().name == "1ply"


def test_beam_direct_and_navigated_agree():
    # search agents through both application modes: identical outcomes
    env = CheeseEnv(level=2)
    for agent in [OnePlyAgent(), BeamAgent(width=10, depth=3)]:
        d = run_episode(agent, env, 1, navigate=False)
        n = run_episode(agent, env, 1, navigate=True)
        assert (d.pieces, d.dug, d.won, d.reason) == (n.pieces, n.dug, n.won, n.reason)
