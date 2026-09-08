"""Validate tuned cheese weights against the hand-set baseline.

Compares a tuned-weight search agent to the hand-set baseline across the
curriculum levels on a fixed seed batch — the acceptance check for a tuning
run before it becomes the new reference.

Run:  .venv/py.sh examples/validate_weights.py models/cheese_l10_beam20x4.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tetris.ai import BeamAgent, CheeseEnv, EvalWeights, OnePlyAgent, run_batch  # noqa: E402
from tetris.ai.tuning import make_agent  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    payload = json.loads(Path(sys.argv[1]).read_text())
    cfg = payload["config"]
    tuned = EvalWeights(**payload["best_weights"])
    levels_trained = cfg.get("levels") or (cfg.get("level"),)
    print(f"tuned cost during training: {payload['best_cost']:.2f}")
    print(f"agent {cfg['agent']} | trained at levels {tuple(levels_trained)}")
    print()

    header = (
        f"| {'level':>5} | {'agent':<16} | {'win%':>6} | {'pieces/line':>11} |"
        f" {'mean pieces':>11} |"
    )
    print(header)
    print(f"|{'-'*7}|{'-'*18}|{'-'*8}|{'-'*13}|{'-'*13}|")
    for level in (1, 2, 3, 5, 10):
        env = CheeseEnv(level=level)
        n = 100 if level <= 5 else 60
        for label, agent in [
            ("hand-set", make_agent(cfg["agent"], EvalWeights())),
            ("tuned", make_agent(cfg["agent"], tuned)),
        ]:
            batch = run_batch(agent, env, n, seed0=777)
            name = f"{cfg['agent']}/{label}"
            ppl = (
                f"{batch.mean_pieces_per_line:11.3f}"
                if batch.mean_pieces is not None
                else "           —"
            )
            mean_p = (
                f"{batch.mean_pieces:11.2f}" if batch.mean_pieces is not None else "           —"
            )
            print(
                f"| {level:>5} | {name:<16} | {100 * batch.win_rate:5.1f}% |"
                f" {ppl} | {mean_p} |"
            )


if __name__ == "__main__":
    main()
