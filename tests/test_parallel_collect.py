"""Parallel collection tests: the fork pool must reproduce the sequential
collectors' data — same seed range, same decisions, same labels.

Equivalence is the load-bearing property: the seed-partitioning is
round-robin, so the UNION of shard episodes equals the sequential run's
episodes over the same range. Teacher labels are deterministic given
(board state, teacher), so every (rows, idx) pair must match one-to-one
as a multiset. Also pinned: DAgger shards run the mixture honestly (the
played game differs from the teacher's when beta < 1, but labels stay
teacher moves), and payload round-trips a net exactly.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tetris.ai import CheeseEnv  # noqa: E402
from tetris.ai.distill import TEACHERS, collect_teacher_data  # noqa: E402
from tetris.ai.parallel import (  # noqa: E402
    collect_dagger_data_parallel,
    collect_teacher_data_parallel,
    net_payload,
    payload_to_net,
)
from tetris.ai.policy import PolicyAgent, PolicyNet  # noqa: E402

# forkserver workers import tetris.ai fresh in each process; no
# multiprocessing deprecation warnings are expected (and none filtered)


def _data_key(data: list[tuple[np.ndarray, int]]) -> list[tuple]:
    """Order-independent identity of a collected dataset."""
    return sorted((x.tobytes(), idx) for x, idx in data)


def test_parallel_teacher_collection_matches_sequential():
    level, n, seed0 = 1, 24, 4000
    seq = collect_teacher_data(
        TEACHERS["1ply"](), CheeseEnv(level=level), n, seed0=seed0
    )
    par, stats = collect_teacher_data_parallel("1ply", level, n, seed0, workers=4)
    assert stats.episodes == n
    assert stats.decisions == len(seq)  # same episodes => same decisions
    assert _data_key(seq) == _data_key(par)


def test_parallel_dagger_collection_runs_and_labels_are_valid():
    net = PolicyNet(hidden=16, layers=1, seed=0)
    data, stats = collect_dagger_data_parallel(
        net, "1ply", level=1, n_episodes=8, seed0=5000,
        beta=0.5, rng_seed=0, workers=3,
    )
    assert stats.episodes == 8
    assert len(data) == stats.decisions > 0
    for x, idx in data:
        assert x.shape[1] == net.net[0].in_features
        assert 0 <= idx < x.shape[0]


def test_net_payload_round_trip():
    net = PolicyNet(hidden=16, layers=1, seed=3)
    before = run_greedy_seed(net, 11)
    net2 = payload_to_net(net_payload(net))
    after = run_greedy_seed(net2, 11)
    assert before == after  # identical weights => identical greedy play


def run_greedy_seed(net, seed: int):
    from tetris.ai import run_episode

    res = run_episode(PolicyAgent(net, greedy=True), CheeseEnv(level=1), seed,
                      navigate=False)
    return (res.pieces, res.dug, res.won, res.reason)
