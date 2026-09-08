"""Curriculum learner runner: train the cheese policy level by level.

GPU-first: rollouts run on the CPU through the harness (engine-bound),
every iteration's decisions update the net in one batched forward/backward
on the device (cuda when available). Same code scales to the RTX Pro 6000
server by changing --device.

Run:  .venv/py.sh examples/cheese_learner.py [--levels 1..10] [--iterations 40]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tetris.ai.curriculum import (  # noqa: E402
    Curriculum,
    CurriculumConfig,
    TrainConfig,
)
from tetris.ai.policy import PolicyAgent, PolicyNet, load_policy, save_policy  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start-level", type=int, default=1)
    ap.add_argument("--max-level", type=int, default=10)
    ap.add_argument("--level-step", type=int, default=1)
    ap.add_argument("--iterations", type=int, default=40, help="iterations per level")
    ap.add_argument("--episodes", type=int, default=64, help="rollouts per iteration")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--reference", type=str, default="beam20x4")
    ap.add_argument("--gate-episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None, help="cuda | cpu (default: auto)")
    ap.add_argument("--resume", type=str, default=None, help="policy checkpoint to resume from")
    ap.add_argument("--no-retention", action="store_true")
    ap.add_argument(
        "--distill-episodes", type=int, default=0,
        help="teacher episodes per level for the distillation warm-start (0 = pure RL)",
    )
    ap.add_argument("--distill-epochs", type=int, default=60)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    if args.resume:
        net, _ = load_policy(Path(args.resume), device=device)
        print(f"resumed from {args.resume}")
    else:
        torch.manual_seed(args.seed)
        net = PolicyNet(hidden=args.hidden, layers=args.layers, seed=args.seed)
    agent = PolicyAgent(net)

    train_cfg = TrainConfig(
        iterations=args.iterations,
        episodes=args.episodes,
        lr=args.lr,
        seed0=args.seed,
        device=device,
    )
    cur_cfg = CurriculumConfig(
        start_level=args.start_level,
        max_level=args.max_level,
        level_step=args.level_step,
        reference=args.reference,
        gate_episodes=args.gate_episodes,
        retention=not args.no_retention,
        checkpoint_dir="models/curriculum",
        distill_episodes=args.distill_episodes,
        distill_epochs=args.distill_epochs,
    )
    cur = Curriculum(cur_cfg, net, train_cfg, rng=np.random.default_rng(args.seed))

    import time

    t0 = time.time()
    for report in cur.run(agent, iterations_per_level=args.iterations):
        gate = report.gate
        ret = " | ".join(
            f"L{l}: {'ok' if g.passed else 'REGRESSED'}" for l, g in report.retention.items()
        ) or "n/a"
        print(
            f"level {report.level}: gate {'PASSED' if report.passed else 'blocked'}"
            f" — {gate.reason} | retention: {ret}"
        )
    save_policy(net, Path("models/curriculum/cheese_policy_final.json"))
    print(f"done in {(time.time() - t0) / 60:.1f} min; final policy saved")


if __name__ == "__main__":
    main()
