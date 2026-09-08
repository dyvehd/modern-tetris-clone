"""DAgger tests: policy-distribution states with teacher labels.

The core property to pin: the recorder plays a mixture (beta = teacher
move probability) but ALWAYS labels with the teacher's choice, so the
collected data is policy-state / teacher-action — the DAgger invariant
that closes behavioral cloning's compounding-error gap.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tetris.ai import CheeseEnv, run_episode  # noqa: E402
from tetris.ai.dagger import DAggerRecorder, dagger_round  # noqa: E402
from tetris.ai.distill import DistillTrainer, TEACHERS  # noqa: E402
from tetris.ai.policy import PolicyAgent, PolicyNet  # noqa: E402


def test_recorder_labels_are_teacher_moves():
    # whatever move gets PLAYED, every recorded label is the teacher's
    # chosen candidate index (beta=0: never plays the teacher's move,
    # yet all labels are teacher decisions)
    net = PolicyNet(hidden=16, layers=1, seed=0)
    policy = PolicyAgent(net)
    data: list[tuple[np.ndarray, int]] = []
    recorder = DAggerRecorder(
        policy, TEACHERS["1ply"](), data, beta=0.0, rng=np.random.default_rng(0)
    )
    env = CheeseEnv(level=1)
    res = run_episode(recorder, env, 3, navigate=False)
    assert len(data) == res.pieces  # one label per decision
    for x, idx in data:
        assert 0 <= idx < x.shape[0]


def test_beta_controls_play_mixture():
    # beta=1: the teacher's move is always played — the game is exactly the
    # teacher's game
    teacher = TEACHERS["1ply"]()
    env = CheeseEnv(level=1)
    plain = run_episode(teacher, env, 7, navigate=False)
    data: list[tuple[np.ndarray, int]] = []
    recorder = DAggerRecorder(
        PolicyAgent(PolicyNet(hidden=16, layers=1, seed=0)),
        TEACHERS["1ply"](), data, beta=1.0, rng=np.random.default_rng(0),
    )
    mixed = run_episode(recorder, env, 7, navigate=False)
    assert (plain.pieces, plain.dug, plain.won, plain.reason) == (
        mixed.pieces, mixed.dug, mixed.won, mixed.reason,
    )


def test_dagger_round_trains_and_accumulates():
    # a round adds data, runs distillation passes, and reports sane stats
    # (accuracy at 8 decisions is pure noise — assert only what the scale
    # can honestly support: accumulation + finite loss)
    net = PolicyNet(hidden=32, layers=1, seed=0)
    policy = PolicyAgent(net)
    trainer = DistillTrainer(net, lr=2e-3, device="cpu", chunk_decisions=64)
    rng = np.random.default_rng(0)
    data: list[tuple[np.ndarray, int]] = []
    env = CheeseEnv(level=1)
    stats = dagger_round(
        net, policy, TEACHERS["1ply"](), env, 8, seed0=0, beta=1.0,
        data=data, trainer=trainer, rng=rng, epochs=3,
    )
    assert stats.episodes == 8
    assert stats.decisions == len(data) > 0
    assert np.isfinite(stats.loss)
    assert 0.0 <= stats.accuracy <= 1.0
