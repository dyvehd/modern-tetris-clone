"""Parallel rollout/data collection for the server phase.

Measured on the RTX 3050: at any net size, the wall time of distillation
and DAgger is ~90% CPU rollouts (pure-Python engine stepping + movegen)
and ~10% GPU training. A bigger GPU cannot buy back that 90% — only more
CPU cores can. The server has 20 cores; the sequential collection loops
in ``distill.py`` / ``dagger.py`` used none of them.

Design: a forkserver-based process pool. A plain ``fork`` of this
process is a deadlock lottery: torch loads OpenMP threads, and a forked
child can inherit a held lock and hang before running any code
(measured: intermittent 0-CPU worker hangs under ``fork``; Python
deprecates multi-threaded fork outright). The forkserver spawns workers
from a clean single-threaded server process instead — deterministic
and safe. The engine itself is pure Python state (no threads, no torch
in the workers — the policy is shipped as a picklable state-dict
payload and rebuilt CPU-side in each worker). Each worker runs whole
episodes and returns the collected ``(candidate rows, label index)``
pairs.

Two entry points, matching the sequential versions' output shapes:

- :func:`collect_teacher_data_parallel` — parallels
  ``distill.collect_teacher_data``.
- :func:`collect_dagger_data_parallel` — parallels one ``dagger_round``
  collection pass (mixture play, teacher labels).

Seeds are partitioned round-robin across workers, so the union of
episodes is identical to the sequential run given the same seed range —
only the collection order differs, and ``DistillTrainer.train_batch``
shuffles before chunking anyway.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from multiprocessing import get_context
from typing import Callable

import numpy as np

from .cheese import CheeseEnv, run_episode
from .distill import TeacherRecorder
from .dagger import DAggerRecorder


def net_payload(net) -> dict:
    """Picklable snapshot of a ``PolicyNet``: architecture + weights as
    plain lists (the same shape ``policy.save_policy`` writes to JSON,
    minus the file). Fork workers rebuild the net from this."""
    from .policy import INPUT_DIM, PLACEMENT_DIM, STATE_DIM

    hidden = net.net[0].out_features
    layers = sum(1 for m in net.net if _is_linear(m)) - 1
    return {
        "hidden": hidden,
        "layers": layers,
        "state_dict": {k: v.detach().cpu().tolist() for k, v in net.state_dict().items()},
        "meta": {
            "state_dim": STATE_DIM,
            "placement_dim": PLACEMENT_DIM,
            "input_dim": INPUT_DIM,
        },
    }


def _is_linear(module) -> bool:
    from torch import nn

    return isinstance(module, nn.Linear)


def payload_to_net(payload: dict):
    """Rebuild a CPU ``PolicyNet`` from :func:`net_payload` output."""
    import torch

    from .policy import PolicyNet

    net = PolicyNet(hidden=payload["hidden"], layers=payload["layers"])
    net.load_state_dict(
        {k: torch.as_tensor(v, dtype=torch.float32) for k, v in payload["state_dict"].items()}
    )
    return net


def _teacher_shard(args: tuple) -> list[tuple[np.ndarray, int]]:
    """Worker: run one seed shard with a fresh teacher + recorder. The
    teacher arrives by NAME (a ``distill.TEACHERS`` key) because pool
    arguments are pickled and lambdas don't pickle."""
    teacher_name, level, seeds = args
    from .distill import TEACHERS

    teacher = TEACHERS[teacher_name]()
    env = CheeseEnv(level=level)
    data: list[tuple[np.ndarray, int]] = []
    recorder = TeacherRecorder(teacher, data)
    for s in seeds:
        run_episode(recorder, env, s, navigate=False)
    return data


# per-worker shared state, populated by the pool initializer (the net
# payload is pickled once per worker, not once per episode task)
_DAGGER_W: dict = {}


def _init_dagger_worker(payload: dict, teacher_name: str, level: int, beta: float) -> None:
    """Pool initializer: build this worker's policy + teacher + env from
    the shared payload. Runs once per worker; episode tasks then reuse
    the stash."""
    from .distill import TEACHERS
    from .policy import PolicyAgent

    net = payload_to_net(payload)
    _DAGGER_W["policy"] = PolicyAgent(net, rng=np.random.default_rng(0))
    _DAGGER_W["teacher"] = TEACHERS[teacher_name]()
    _DAGGER_W["env"] = CheeseEnv(level=level)
    _DAGGER_W["beta"] = beta


def _dagger_episode(args: tuple) -> list[tuple[np.ndarray, int]]:
    """Worker task: one mixture-play episode; records the TEACHER's
    index at every visited state. RNG seeds differ per task so sampling
    streams are independent."""
    seed, rng_seed = args
    policy = _DAGGER_W["policy"]
    policy.rng = np.random.default_rng(rng_seed)
    recorder = DAggerRecorder(
        policy, _DAGGER_W["teacher"], [], _DAGGER_W["beta"],
        np.random.default_rng(rng_seed + 1),
    )
    run_episode(recorder, _DAGGER_W["env"], seed, navigate=False)
    return recorder.data


@dataclass(frozen=True)
class CollectStats:
    """One parallel collection pass: totals and throughput."""

    episodes: int
    decisions: int
    workers: int
    seconds: float

    @property
    def episodes_per_sec(self) -> float:
        return self.episodes / self.seconds if self.seconds else 0.0

    def summary(self) -> str:
        return (
            f"collected {self.episodes} episodes ({self.decisions} decisions)"
            f" on {self.workers} workers in {self.seconds:.1f}s"
            f" ({self.episodes_per_sec:.1f} eps/s)"
        )


def _fan_out(
    shards: list[tuple],
    worker_fn: Callable,
    n_episodes: int,
    n_workers: int,
) -> tuple[list[tuple[np.ndarray, int]], CollectStats]:
    t0 = time.time()
    # forkserver: children never fork from this torch-loaded, multi-threaded
    # parent (plain fork is a deadlock lottery there — measured)
    ctx = get_context("forkserver")
    # chunksize=1: episode lengths vary wildly (a 1-piece win vs a 60-piece
    # dig), so fixed per-worker seed lists suffer stragglers; one episode
    # per task lets the pool load-balance (measured 3.6x -> 5.7x on 15 cores)
    with ctx.Pool(n_workers) as pool:
        shards_out = pool.map(worker_fn, shards, chunksize=1)
    data = [row for shard in shards_out for row in shard]
    stats = CollectStats(
        episodes=n_episodes,
        decisions=len(data),
        workers=n_workers,
        seconds=time.time() - t0,
    )
    return data, stats


def _n_workers(workers: int) -> int:
    import os

    return workers or max(1, (os.cpu_count() or 2) - 1)


def collect_teacher_data_parallel(
    teacher_name: str,
    level: int,
    n_episodes: int,
    seed0: int,
    workers: int = 0,
) -> tuple[list[tuple[np.ndarray, int]], CollectStats]:
    """Teacher episodes collected by a fork pool — the parallel version
    of ``distill.collect_teacher_data``. ``teacher_name`` is a
    ``distill.TEACHERS`` key (passed by name: pool arguments are
    pickled, lambdas are not). ``workers=0`` uses every core except
    one."""
    n_workers = _n_workers(workers)
    shards = [(teacher_name, level, [s]) for s in range(seed0, seed0 + n_episodes)]
    return _fan_out(shards, _teacher_shard, n_episodes, n_workers)


def collect_dagger_data_parallel(
    net,
    teacher_name: str,
    level: int,
    n_episodes: int,
    seed0: int,
    beta: float,
    rng_seed: int = 0,
    workers: int = 0,
) -> tuple[list[tuple[np.ndarray, int]], CollectStats]:
    """DAgger mixture episodes collected by a fork pool — the parallel
    version of one ``dagger_round`` collection pass. The policy payload
    is shipped to each worker ONCE (pool initializer), and tasks are
    bare episode seeds; the net stays CPU-side in workers (sampling
    rollouts don't need the GPU)."""
    n_workers = _n_workers(workers)
    t0 = time.time()
    ctx = get_context("forkserver")
    payload = net_payload(net)
    tasks = [
        (seed0 + i, int(rng_seed + 1_000_003 * (i + 1)))
        for i in range(n_episodes)
    ]
    with ctx.Pool(
        n_workers,
        initializer=_init_dagger_worker,
        initargs=(payload, teacher_name, level, beta),
    ) as pool:
        shards_out = pool.map(_dagger_episode, tasks, chunksize=1)
    data = [row for shard in shards_out for row in shard]
    stats = CollectStats(
        episodes=n_episodes,
        decisions=len(data),
        workers=n_workers,
        seconds=time.time() - t0,
    )
    return data, stats
