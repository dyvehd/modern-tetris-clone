#!/usr/bin/env python
"""Value-network experiment runner — expert iteration, round 1.

The reviews' convergence point (advisor review step 4): train
``V(afterstate)`` on Monte-Carlo cost-to-go labels from corrected-beam
rollouts, then plug it into the beam as the leaf evaluator. Acceptance:
**beam 10×3 + V ≥ beam 40×5 + linear eval** on a locked seed manifest.

Stages (each resumable; state in models/value/):

  collect  — teacher episodes across the curriculum levels (mixed, the
             distillation recipe: harder levels teach the same dig skill
             with more decisions per episode), every candidate of every
             decision recorded with the episode's realized return.
  train    — twin-headed regression on the GPU (q: softplus pieces-to-go
             MSE, fail: BCE), held-out split logged per epoch.
  eval     — the acceptance manifest: paired per-seed comparison of
             beam10x3+V vs beam40x5 (linear) at the curriculum levels,
             plus a strength ladder (V at several widths/depths vs the
             linear beams they replace) and the runtime measurement.

Seed bands (disjoint from every existing band — see CurriculumConfig):
  value-train 700M+, value-manifest 800M+ (locked, never trained on).

Usage (server):
  nohup python examples/value_experiment.py --stage collect --episodes 3000 \
      > value_collect.log 2>&1 &
  python examples/value_experiment.py --stage train
  python examples/value_experiment.py --stage eval
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tetris.ai import BeamAgent, CheeseEnv, run_episode  # noqa: E402
from tetris.ai.cheese import run_batch  # noqa: E402
from tetris.ai.value import (  # noqa: E402
    ValueNet,
    ValueTrainer,
    collect_value_data_parallel,
    evaluate_q_mae,
    load_value,
    save_value,
)
from tetris.ai.valuebeam import ValueBeamAgent  # noqa: E402

MODELS = Path("models/value")
LEVELS = (1, 3, 5, 10)  # the curriculum spread (mixed-level recipe)
TRAIN_SEED0 = 700_000_000
MANIFEST_SEED0 = 800_000_000
MANIFEST_EPISODES = 100

# one data source per collect pass: (member, teacher, level, piece_cap,
# episode multiplier). The beam teacher never fails at any level, so the
# failure head would see zero positives — the last source is a weak
# teacher under a tight cap: real failure labels (capped episodes), and
# its messy boards are exactly the junk-state distribution DAgger taught
# us to include.
SOURCES = [
    ("L1", "beam20x4", 1, None, 2.0),
    ("L3", "beam20x4", 3, None, 1.0),
    ("L5", "beam20x4", 5, None, 1.0),
    ("L10", "beam20x4", 10, None, 1.0),
    ("failL10", "1ply", 10, 40, 0.1),
]


def _rss_mb() -> float:
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return -1.0


def stage_collect(args) -> None:
    import subprocess

    MODELS.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    manifest: dict[str, dict] = {}
    for name, teacher, level, cap, mult in SOURCES:
        # one detached sub-process per source: the parent imports torch
        # (train stage) and must not fork workers from a torch-loaded
        # process (OpenMP deadlock — parallel.py's measured lesson), so
        # each child (value_collect_child: numpy+tetris only) does the
        # forkserver collection and writes one .npz.
        eps = max(1, int(round(args.episodes * mult)))
        cap_arg = "none" if cap is None else str(cap)
        cmd = [
            sys.executable, str(root / "examples" / "value_collect_child.py"),
            str(level), str(eps),
            str(TRAIN_SEED0 + 1_000_000 * level + (abs(hash(name)) % 10_000)),
            str(args.workers), teacher, cap_arg,
            str(MODELS / f"round1_{name}.npz"),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=root)
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-2000:], flush=True)
            raise SystemExit(f"collect child failed for {name}")
        print(r.stdout.strip(), flush=True)
        z = np.load(MODELS / f"round1_{name}.npz")
        manifest[name] = {
            "rows": int(len(z["q"])),
            "mean_q": float(z["q"].mean()),
            "failed": float(z["failed"].mean()),
            "size_mb": (MODELS / f"round1_{name}.npz").stat().st_size / 1e6,
        }
        z.close()
        print(f"collect {name}: {manifest[name]} | rss {_rss_mb():.0f} MB", flush=True)
    (MODELS / "round1_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"manifest -> {MODELS}/round1_manifest.json", flush=True)


def _load_packed():
    """The packed training block: (x, q, failed, explored) with every
    source materialized once, rows never copied individually. Rows stay
    views into the single x block."""
    xs, qs, fs, es = [], [], [], []
    for name, *_ in SOURCES:
        z = np.load(MODELS / f"round1_{name}.npz")
        xs.append(z["x"])  # decompress each member exactly once
        qs.append(z["q"])
        fs.append(z["failed"])
        es.append(z["explored"])
        z.close()
    return (
        np.concatenate(xs), np.concatenate(qs), np.concatenate(fs), np.concatenate(es)
    )


def stage_train(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, q, failed, explored = _load_packed()
    n = len(q)
    print(f"loaded {n} rows from {len(SOURCES)} sources "
          f"({x.nbytes / 1e9:.2f} GB) | rss {_rss_mb():.0f} MB", flush=True)
    # held-out split: the LAST 10% of rows never enters training (the
    # v7 distillation rule — never trust training-set fit). Sources are
    # concatenated in order, so the split is a contiguous tail of each
    # block's stream — fresh seeds never seen in training.
    n_eval = max(1, n // 10)
    eval_packed = (x[-n_eval:], q[-n_eval:], failed[-n_eval:], explored[-n_eval:])
    train_packed = (x[:-n_eval], q[:-n_eval], failed[:-n_eval], explored[:-n_eval])
    net = ValueNet(hidden=args.hidden, layers=args.layers, seed=args.seed)
    trainer = ValueTrainer(
        net, lr=args.lr, device=device, chunk_rows=args.chunk,
        rng=np.random.default_rng(args.seed),
    )
    best_mae = float("inf")
    for epoch in range(1, args.epochs + 1):
        stats = trainer.train_epoch(train_packed)
        if epoch == 1 or epoch % 10 == 0:
            mae = evaluate_q_mae(net, eval_packed, device)
            marker = ""
            if mae < best_mae:
                best_mae = mae
                save_value(net, MODELS / "value_round1.json")
                marker = " *saved*"
            print(
                f"epoch {epoch}/{args.epochs} | loss {stats.loss:.4f} "
                f"(q {stats.q_loss:.4f}, fail {stats.fail_loss:.4f}) "
                f"| train q-MAE {stats.q_mae:.3f} | held-out q-MAE {mae:.3f}"
                f" | fail_hat {stats.fail_rate_hat:.3f} "
                f"| {stats.seconds:.1f}s{marker}",
                flush=True,
            )
    print(f"best held-out q-MAE: {best_mae:.3f}; saved {MODELS}/value_round1.json", flush=True)


def stage_eval(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = load_value(MODELS / "value_round1.json", device=device)
    results: dict = {"levels": {}, "ladder": {}, "runtime": {}}

    # the acceptance manifest: paired per-seed comparison at each level
    for lvl in LEVELS:
        env = CheeseEnv(level=lvl)
        vb = ValueBeamAgent(net, width=10, depth=3, device=device)
        big = BeamAgent(width=40, depth=5)
        a = run_batch(vb, env, MANIFEST_EPISODES, seed0=MANIFEST_SEED0 + lvl * 1_000)
        b = run_batch(big, env, MANIFEST_EPISODES, seed0=MANIFEST_SEED0 + lvl * 1_000)
        # paired per-seed: wins counted by reason, mean over wins, plus
        # paired differences (failures charged the cap: 400)
        cap = env.piece_cap
        pa = [p if r == "cleared" else cap for p, r in zip(a.pieces, a.reasons)]
        pb = [p if r == "cleared" else cap for p, r in zip(b.pieces, b.reasons)]
        diffs = [x - y for x, y in zip(pa, pb)]
        n_better = sum(1 for d in diffs if d < 0)
        n_worse = sum(1 for d in diffs if d > 0)
        results["levels"][lvl] = {
            "v_mean_all_seeds": float(np.mean(pa)),
            "lin_mean_all_seeds": float(np.mean(pb)),
            "v_win_rate": a.win_rate,
            "lin_win_rate": b.win_rate,
            "v_mean_wins": a.mean_pieces,
            "lin_mean_wins": b.mean_pieces,
            "paired_better": n_better,
            "paired_worse": n_worse,
            "paired_equal": len(diffs) - n_better - n_worse,
        }
        print(
            f"L{lvl}: V10x3 mean(all) {np.mean(pa):.2f} win {a.win_rate:.0%} "
            f"vs linear40x5 {np.mean(pb):.2f} win {b.win_rate:.0%} "
            f"| paired better/worse/equal {n_better}/{n_worse}/{len(diffs) - n_better - n_worse}",
            flush=True,
        )

    # strength ladder: V at several sizes vs the linear beam of equal size
    for width, depth in [(5, 2), (10, 3), (20, 4), (40, 5)]:
        env = CheeseEnv(level=10)
        vb = ValueBeamAgent(net, width=width, depth=depth, device=device)
        lb = BeamAgent(width=width, depth=depth)
        a = run_batch(vb, env, 30, seed0=MANIFEST_SEED0 + 50_000)
        b = run_batch(lb, env, 30, seed0=MANIFEST_SEED0 + 50_000)
        results["ladder"][f"{width}x{depth}"] = {
            "v_mean_wins": a.mean_pieces, "v_win": a.win_rate,
            "lin_mean_wins": b.mean_pieces, "lin_win": b.win_rate,
        }
        print(
            f"L10 beam{width}x{depth}: +V {a.mean_pieces} (win {a.win_rate:.0%}) "
            f"vs linear {b.mean_pieces} (win {b.win_rate:.0%})",
            flush=True,
        )

    # runtime: seconds/episode of each agent at L10 (the compute story)
    env = CheeseEnv(level=10)
    for name, agent in [
        ("linear20x4", BeamAgent(width=20, depth=4)),
        ("linear40x5", BeamAgent(width=40, depth=5)),
        ("V10x3", ValueBeamAgent(net, width=10, depth=3, device=device)),
        ("V20x4", ValueBeamAgent(net, width=20, depth=4, device=device)),
    ]:
        t0 = time.time()
        run_batch(agent, env, 10, seed0=MANIFEST_SEED0 + 90_000)
        secs = time.time() - t0
        results["runtime"][name] = secs / 10
        print(f"runtime {name}: {secs / 10:.2f} s/episode", flush=True)

    MODELS.mkdir(parents=True, exist_ok=True)
    out = MODELS / "value_round1_eval.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"eval results -> {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=["collect", "train", "eval"])
    ap.add_argument("--episodes", type=int, default=3000, help="per level (collect)")
    ap.add_argument("--workers", type=int, default=19)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--chunk", type=int, default=65536)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    {"collect": stage_collect, "train": stage_train, "eval": stage_eval}[args.stage](args)


if __name__ == "__main__":
    main()
