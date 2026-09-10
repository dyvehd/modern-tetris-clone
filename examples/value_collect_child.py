"""One value-data collection source — the torch-free child process.

`value_experiment.stage_collect` runs this in a detached sub-process per
source: the parent (which imports torch for the training stage) must
not plain-fork workers (OpenMP-thread deadlocks — parallel.py's measured
lesson), so each child imports only numpy + the harness, does the
forkserver collection, and writes one compressed .npz of (x, q, failed,
explored).

Imports numpy + tetris only — torch stays out of the fork path.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from tetris.ai.cheese import CheeseEnv, run_episode  # noqa: E402
from tetris.ai.distill import TEACHERS  # noqa: E402
from tetris.ai.value import MCValueRecorder  # noqa: E402


_STATE: dict = {}


def _init_pool(level: int, teacher_name: str, piece_cap: int | None) -> None:
    """Pool initializer: build this worker's env + teacher once (pool
    args are pickled; teachers ship by name)."""
    _STATE["env"] = CheeseEnv(level=level, piece_cap=piece_cap or 400)
    _STATE["teacher"] = TEACHERS[teacher_name]()


def _episode_shard(seed: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """One episode -> (rows, q, failed, explored) packed arrays."""
    data = []
    recorder = MCValueRecorder(_STATE["teacher"], data)
    result = run_episode(recorder, _STATE["env"], seed, navigate=False)
    recorder.flush(result.pieces, failed=not result.won)
    return [(
        np.stack([d.x for d in data]),
        np.asarray([d.q_pieces for d in data], dtype=np.float32),
        np.asarray([d.failed for d in data], dtype=np.float32),
        np.asarray([d.explored for d in data], dtype=np.float32),
    )]


def collect_one(
    level: int,
    episodes: int,
    seed0: int,
    workers: int,
    teacher_name: str,
    piece_cap: int | None,
    out_path: str,
) -> None:
    from multiprocessing import get_context
    import os

    n_workers = workers or max(1, (os.cpu_count() or 2) - 1)
    t0 = time.time()
    ctx = get_context("forkserver")
    with ctx.Pool(
        n_workers, initializer=_init_pool,
        initargs=(level, teacher_name, piece_cap),
    ) as pool:
        out = pool.map(_episode_shard, [seed0 + i for i in range(episodes)], chunksize=1)
    eps = [row for shard in out for row in shard]
    x = np.concatenate([e[0] for e in eps])
    q = np.concatenate([e[1] for e in eps])
    failed = np.concatenate([e[2] for e in eps])
    explored = np.concatenate([e[3] for e in eps])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, x=x, q=q, failed=failed, explored=explored)
    secs = time.time() - t0
    print(
        f"{teacher_name} L{level} cap={piece_cap}: {episodes} eps, {len(q)} rows "
        f"in {secs:.0f}s ({episodes / max(secs, 1):.2f} eps/s) | "
        f"mean q {q.mean():.2f} | failed {failed.mean():.1%}",
        flush=True,
    )


if __name__ == "__main__":
    collect_one(
        int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
        sys.argv[5], None if sys.argv[6] == "none" else int(sys.argv[6]), sys.argv[7],
    )
