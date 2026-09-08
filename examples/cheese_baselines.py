"""Cheese-race baseline table: run every baseline agent across the
curriculum levels and report pieces-per-line with confidence intervals —
the reference numbers every learner must beat to advance a level.

Run:  .venv/py.sh examples/cheese_baselines.py [--episodes N] [--levels 1,2,3,10,18,100]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tetris.ai import (  # noqa: E402
    BeamAgent,
    CheeseEnv,
    GreedyDigAgent,
    OnePlyAgent,
    RandomAgent,
    run_batch,
)

DEFAULT_LEVELS = (1, 2, 3, 5, 10, 18)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=200, help="episodes per cell")
    ap.add_argument("--levels", type=str, default=",".join(map(str, DEFAULT_LEVELS)))
    ap.add_argument("--seed0", type=int, default=0)
    args = ap.parse_args()

    levels = [int(x) for x in args.levels.split(",")]
    agents = [
        RandomAgent(seed=1),
        GreedyDigAgent(),
        OnePlyAgent(),
        BeamAgent(width=20, depth=4),
        BeamAgent(width=60, depth=5),
    ]

    header = f"| {'level':>5} | {'agent':<12} | {'win%':>6} | {'pieces/line':>11} | {'95% CI':>7} | {'mean pieces':>11} | {'time':>5} |"
    print(header)
    print(f"|{'-'*7}|{'-'*14}|{'-'*8}|{'-'*13}|{'-'*9}|{'-'*13}|{'-'*7}|")
    for level in levels:
        env = CheeseEnv(level=level)
        for agent in agents:
            if isinstance(agent, RandomAgent) and level > 2:
                print(
                    f"| {level:>5} | {agent.name:<12} | {100 * 0.0:5.1f} |           — |"
                    f"       — | skipped (random cannot clear) |"
                )
                continue
            t0 = time.time()
            batch = run_batch(agent, env, args.episodes, seed0=args.seed0)
            dt = time.time() - t0
            win_pct = f"{100 * batch.win_rate:5.1f}"
            if batch.mean_pieces is None:
                ppl = "—"
                ci = "—"
                mean_p = "—"
            else:
                ppl = f"{batch.mean_pieces_per_line:11.3f}"
                ci = f"{batch.ci95_pieces / level:7.3f}" if batch.ci95_pieces else "—"
                mean_p = f"{batch.mean_pieces:11.2f}"
            print(f"| {level:>5} | {agent.name:<12} | {win_pct} | {ppl} | {ci} | {mean_p} | {dt:4.0f}s |")


if __name__ == "__main__":
    main()
