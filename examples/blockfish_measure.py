"""Measure blockfish on the locked value-experiment manifest.

The round-2 bar: blockfish plays exactly the seeds the V1 acceptance
manifest used (band 800M + level*1000, 100 episodes per level), so
every existing number (lin40x5 = 21.47 at L10, V10x3+blend = 25.89)
becomes directly comparable. Also records think-time per decision so
budget comparisons can be made at matched wall-clock, and saves the
per-episode results as JSON.

Run (server):  python3 blockfish_measure.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tetris.ai.blockfish_agent import BlockfishAgent
from tetris.ai.cheese import CheeseEnv, run_episode

LEVELS = [1, 3, 5, 10]
EPISODES = 100
BAND0 = 800_000_000


def main() -> None:
    agent = BlockfishAgent()
    out = {"agent": "blockfish", "search_limit": 50_000, "levels": {}}
    for level in LEVELS:
        env = CheeseEnv(level=level)
        seed0 = BAND0 + level * 1000
        pieces_list = []
        fails = 0
        t0 = time.time()
        for i in range(EPISODES):
            res = run_episode(agent, env, seed0 + i, navigate=False)
            if res.won:
                pieces_list.append(res.pieces)
            else:
                fails += 1
            if res.reason == "capped":
                print(f"L{level} seed {seed0+i}: CAPPED at episode level — investigate")
        dt = time.time() - t0
        wins = EPISODES - fails
        capped_mean = (
            (sum(pieces_list) + fails * 400) / EPISODES
            if pieces_list or fails
            else float("nan")
        )
        mean_won = sum(pieces_list) / len(pieces_list) if pieces_list else None
        out["levels"][str(level)] = {
            "episodes": EPISODES,
            "wins": wins,
            "win_rate": wins / EPISODES,
            "pieces_mean_won": mean_won,
            "capped_mean_fail400": capped_mean,
            "min": min(pieces_list) if pieces_list else None,
            "max": max(pieces_list) if pieces_list else None,
            "sec_per_ep": dt / EPISODES,
        }
        print(
            f"L{level}: win {wins}/{EPISODES} mean(won) {mean_won} "
            f"capped {capped_mean:.2f} ({dt/EPISODES:.2f} s/ep)"
        )
    dest = Path("models/value/blockfish_manifest.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"saved {dest}")


if __name__ == "__main__":
    main()
