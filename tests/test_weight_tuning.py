"""Cross-entropy weight-tuning tests.

Small budgets only (the real tuning runs minutes-hours in background):
the cost model's arithmetic, the CEM loop's invariants (incumbent always
evaluated, no-regression best tracking, bound clamping), determinism, and
the agent factory. The cost model is the load-bearing piece — a topout
costing the full cap is what keeps survival a hard constraint during
tuning.

The fork-pool warning is filtered: the tuner forks worker processes by
design (copy-on-write; the threads CPython's detection sees are pytest's
own runner, not ours).
"""

import pytest

from tetris.ai import BeamAgent, CheeseEnv, OnePlyAgent, run_batch
from tetris.ai.eval import EvalWeights, eval_board
from tetris.ai.tuning import (
    TUNABLE,
    TuningConfig,
    _cost_of_batch,
    _evaluate_candidate,
    make_agent,
    tune,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:This process.*multi-threaded.*fork.*:DeprecationWarning"
)


def test_cost_model():
    # failures cost the cap: a topout-heavy batch must score worse than
    # any winning batch, and the arithmetic is the plain capped mean
    class FakeBatch:
        pieces = (10, 12, 400, 400)
        reasons = ("cleared", "cleared", "topout", "capped")

    assert _cost_of_batch(FakeBatch(), cap=400) == pytest.approx((10 + 12 + 400 + 400) / 4)
    # a real batch: all wins, cost == mean pieces
    batch = run_batch(OnePlyAgent(), CheeseEnv(level=1), 10, seed0=0)
    cost = _cost_of_batch(batch, cap=400)
    assert cost == pytest.approx(sum(batch.pieces) / 10)
    assert batch.win_rate == 1.0


def test_tunable_bounds_are_sane():
    # every tunable weight has a valid (lo, hi) with lo < hi, and the
    # objective-critical weights (lines, win) are NOT tunable
    for name, (lo, hi) in TUNABLE.items():
        assert lo < hi, name
        assert hasattr(EvalWeights(), name)
    assert "lines" not in TUNABLE and "win" not in TUNABLE


def test_make_agent_factory():
    w = EvalWeights()
    assert isinstance(make_agent("1ply", w), OnePlyAgent)
    beam = make_agent("beam20x4", w)
    assert isinstance(beam, BeamAgent)
    assert beam.width == 20 and beam.depth == 4 and beam.weights is w
    with pytest.raises(ValueError):
        make_agent("bogus", w)


def test_cem_determinism_and_no_regression():
    # tiny budget: the incumbent (initial weights) is always evaluated, so
    # the all-time best cost can never be worse than the incumbent's own
    # cost; identical cfg → identical trajectory
    cfg = TuningConfig(
        agent="1ply", levels=(1,), episodes=6, generations=2, candidates=4, seed0=0,
    )
    incumbent = _evaluate_candidate(
        ([getattr(EvalWeights(), n) for n in TUNABLE], cfg)
    )
    best1, hist1 = tune(cfg, verbose=False)
    best2, hist2 = tune(cfg, verbose=False)
    assert hist1 == hist2  # determinism
    assert best1 == best2
    best_cost = min(h["best_cost"] for h in hist1)
    assert best_cost <= incumbent  # never worse than the starting weights


def test_cem_respects_bounds():
    # candidates are clamped to the TUNABLE bounds: every generated weight
    # vector in the history lies within them
    cfg = TuningConfig(
        agent="1ply", levels=(1,), episodes=4, generations=2, candidates=4, seed0=3,
    )
    _, hist = tune(cfg, verbose=False)
    for rec in hist:
        for name, value in rec["best_weights"].items():
            lo, hi = TUNABLE[name]
            assert lo <= value <= hi, (name, value)
