"""Round-2 acceptance eval — V-beam vs BOTH bars on the locked manifest.

Same locked manifest as round 1 (band 800M + level*1000, 100 eps/level,
capped-mean with failures charged 400): the old bar (linear 40×5, the
advisor-review step-4 acceptance reference) and the new SOTA bar
(blockfish at search_limit 50k, measured 17.13 at L10 on this exact
manifest). Plus the equal-budget ladder (V vs same-size linear) and
runtimes.

Run (server):  python3 examples/bf_eval.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tetris.ai.blockfish_agent import BlockfishAgent  # noqa: E402
from tetris.ai.cheese import CheeseEnv, run_batch  # noqa: E402
from tetris.ai.search import BeamAgent  # noqa: E402
from tetris.ai.value import load_value  # noqa: E402
from tetris.ai.valuebeam import ValueBeamAgent  # noqa: E402

LEVELS = [1, 3, 5, 10]
MANIFEST_EPISODES = 100
MANIFEST_SEED0 = 800_000_000
LADDER_EPISODES = 30
MODELS = Path("models/value")


def capped(res, cap: int) -> list[int]:
    return [p if r == "cleared" else cap for p, r in zip(res.pieces, res.reasons)]


def paired(a, b, cap: int) -> tuple[int, int, int]:
    pa, pb = capped(a, cap), capped(b, cap)
    diffs = [x - y for x, y in zip(pa, pb)]
    n_better = sum(1 for d in diffs if d < 0)
    n_worse = sum(1 for d in diffs if d > 0)
    return n_better, n_worse, len(diffs) - n_better - n_worse


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = load_value(MODELS / "value_round2.json", device=device)
    results: dict = {"levels": {}, "ladder": {}, "runtime": {}}

    # per level: blockfish (the new bar), V-beam (pure and blended —
    # blend .5 was round 1's acceptance config), and the old linear bar
    bf = BlockfishAgent()
    bf_by_level: dict = {}
    for lvl in LEVELS:
        env = CheeseEnv(level=lvl)
        results["levels"][lvl] = {}
        t0 = time.time()
        res_bf = run_batch(bf, env, MANIFEST_EPISODES, seed0=MANIFEST_SEED0 + lvl * 1_000)
        bf_by_level[lvl] = res_bf
        results["levels"][lvl]["blockfish"] = {
            "mean_capped": float(np.mean(capped(res_bf, env.piece_cap))),
            "win_rate": res_bf.win_rate,
            "mean_wins": res_bf.mean_pieces,
            "sec_per_ep": (time.time() - t0) / MANIFEST_EPISODES,
        }
        big = BeamAgent(width=40, depth=5)
        b = run_batch(big, env, MANIFEST_EPISODES, seed0=MANIFEST_SEED0 + lvl * 1_000)
        results["levels"][lvl]["lin40x5"] = {
            "mean_capped": float(np.mean(capped(b, env.piece_cap))),
            "win_rate": b.win_rate,
            "mean_wins": b.mean_pieces,
        }
        for tag, blend in [("v10x3", 0.0), ("v10x3_b50", 0.5)]:
            vb = ValueBeamAgent(net, width=10, depth=3, eval_blend=blend, device=device)
            a = run_batch(vb, env, MANIFEST_EPISODES, seed0=MANIFEST_SEED0 + lvl * 1_000)
            results["levels"][lvl][tag] = {
                "mean_capped": float(np.mean(capped(a, env.piece_cap))),
                "win_rate": a.win_rate,
                "mean_wins": a.mean_pieces,
                "paired_vs_lin40x5": paired(a, b, env.piece_cap),
                "paired_vs_blockfish": paired(a, res_bf, env.piece_cap),
            }
        v0 = results["levels"][lvl]["v10x3"]
        v5 = results["levels"][lvl]["v10x3_b50"]
        print(
            f"L{lvl}: V {v0['mean_capped']:.2f} (win {v0['win_rate']:.0%}) | "
            f"V+b.5 {v5['mean_capped']:.2f} (win {v5['win_rate']:.0%}) "
            f"vs lin40x5 {results['levels'][lvl]['lin40x5']['mean_capped']:.2f} "
            f"vs bf {results['levels'][lvl]['blockfish']['mean_capped']:.2f} | "
            f"V-paired vs lin {v0['paired_vs_lin40x5'][0]}/{v0['paired_vs_lin40x5'][1]}/{v0['paired_vs_lin40x5'][2]}, "
            f"vs bf {v0['paired_vs_blockfish'][0]}/{v0['paired_vs_blockfish'][1]}/{v0['paired_vs_blockfish'][2]}",
            flush=True,
        )

    # equal-budget ladder at L10
    env = CheeseEnv(level=10)
    for width, depth in [(5, 2), (10, 3), (20, 4), (40, 5)]:
        vb = ValueBeamAgent(net, width=width, depth=depth, device=device)
        lb = BeamAgent(width=width, depth=depth)
        t0 = time.time()
        a = run_batch(vb, env, LADDER_EPISODES, seed0=MANIFEST_SEED0 + 50_000)
        t_v = time.time() - t0
        t0 = time.time()
        b = run_batch(lb, env, LADDER_EPISODES, seed0=MANIFEST_SEED0 + 50_000)
        t_l = time.time() - t0
        results["ladder"][f"{width}x{depth}"] = {
            "v_capped": float(np.mean(capped(a, env.piece_cap))),
            "v_win": a.win_rate,
            "lin_capped": float(np.mean(capped(b, env.piece_cap))),
            "lin_win": b.win_rate,
        }
        results["runtime"][f"{width}x{depth}"] = {
            "v_sec_per_ep": t_v / LADDER_EPISODES, "lin_sec_per_ep": t_l / LADDER_EPISODES,
        }
        print(f"L10 ladder {width}x{depth}: V {np.mean(capped(a, env.piece_cap)):.2f} "
              f"vs lin {np.mean(capped(b, env.piece_cap)):.2f} "
              f"({t_v / LADDER_EPISODES:.2f} vs {t_l / LADDER_EPISODES:.2f} s/ep)", flush=True)

    # V-beam runtime at the acceptance size
    vb = ValueBeamAgent(net, width=10, depth=3, device=device)
    env = CheeseEnv(level=10)
    t0 = time.time()
    run_batch(vb, env, 30, seed0=MANIFEST_SEED0 + 50_000)
    results["runtime"]["v10x3_manifest"] = (time.time() - t0) / 30
    big = BeamAgent(width=40, depth=5)
    t0 = time.time()
    run_batch(big, env, 30, seed0=MANIFEST_SEED0 + 50_000)
    results["runtime"]["lin40x5_manifest"] = (time.time() - t0) / 30

    dest = MODELS / "value_round2_eval.json"
    dest.write_text(json.dumps(results, indent=2))
    print("EVAL DONE ->", dest, flush=True)


if __name__ == "__main__":
    main()
