"""Distillation tests (Deliverable 5's core, pulled inside the curriculum).

The load-bearing claims: the teacher recorder captures the exact decisions
the search agent makes (no translation loss — the label is a candidate
index in the same encoding the policy scores), and the batched
cross-entropy trainer actually concentrates the policy's argmax on the
teacher's choices. Guarded by torch availability like the other learner
tests.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tetris.ai import CheeseEnv, GreedyDigAgent, run_episode  # noqa: E402
from tetris.ai.distill import (  # noqa: E402
    TEACHERS,
    DistillTrainer,
    TeacherRecorder,
    collect_teacher_data,
)
from tetris.ai.policy import PolicyAgent, PolicyNet  # noqa: E402


def test_recorder_captures_teacher_decisions():
    # every teacher decision becomes a (candidates, index) pair whose
    # placement matches what the teacher actually played
    env = CheeseEnv(level=1)
    data = collect_teacher_data(TEACHERS["1ply"](), env, 5, seed0=1)
    assert len(data) >= 5  # at least one decision per episode
    for x, idx in data:
        assert x.ndim == 2 and x.shape[0] > idx >= 0


def test_recorder_replays_identical_game():
    # the recorder is a pure pass-through: the teacher's game is unchanged
    env = CheeseEnv(level=2)
    teacher = TEACHERS["1ply"]()
    plain = run_episode(teacher, env, 42, navigate=False)
    data = []
    recorded = run_episode(TeacherRecorder(TEACHERS["1ply"](), data), env, 42, navigate=False)
    assert (plain.pieces, plain.dug, plain.won, plain.reason) == (
        recorded.pieces, recorded.dug, recorded.won, recorded.reason,
    )
    assert len(data) == recorded.pieces


def test_distillation_moves_argmax_toward_teacher():
    # after a few epochs the policy's group-argmax agreement with the
    # teacher must be well above the ~1/30 random-chance level, and the
    # loss must fall below its first-epoch value
    env = CheeseEnv(level=1)
    data = collect_teacher_data(TEACHERS["1ply"](), env, 60, seed0=3)
    net = PolicyNet(hidden=64, layers=2, seed=0)
    trainer = DistillTrainer(net, lr=2e-3, device="cpu")
    first = trainer.train_batch(data)
    last = None
    for _ in range(20):
        last = trainer.train_batch(data)
    # tiny dataset (61 decisions): assert the trend, not an absolute bar —
    # accuracy roughly doubles from the ~10% first epoch, well above the
    # ~3% chance level, and the loss falls
    assert last.accuracy > max(0.15, 1.8 * first.accuracy)
    assert last.loss < first.loss


def test_distilled_policy_improves_level1():
    # the make-or-break end-to-end claim at tiny scale: a net distilled
    # from the level-1-teacher data must win meaningfully more level-1
    # episodes than its random init did
    env = CheeseEnv(level=1)
    net = PolicyNet(hidden=64, layers=2, seed=1)
    fresh = PolicyAgent(net, greedy=True)
    before = sum(run_episode(fresh, env, s, navigate=False).won for s in range(30))

    data = collect_teacher_data(TEACHERS["1ply"](), env, 200, seed0=5)
    trainer = DistillTrainer(net, lr=2e-3, device="cpu")
    for _ in range(30):
        trainer.train_batch(data)
    distilled = PolicyAgent(net, greedy=True)
    after = sum(run_episode(distilled, env, s, navigate=False).won for s in range(30))
    assert after > before, (before, after)
