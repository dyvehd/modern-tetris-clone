"""A/B: original beam vs corrected beam variants, same seeds, fast path."""
import sys, time, statistics as st
sys.path.insert(0, "src"); sys.path.insert(0, "docs/advisor-review-repro")
from multiprocessing import Pool
from tetris.ai.cheese import CheeseEnv, run_episode
from tetris.ai.search import BeamAgent
from beam2 import Beam2

VARIANTS = {
    "orig20x4": lambda: BeamAgent(20, 4),
    "holdfix": lambda: Beam2(20, 4, fix_hold=True, expand_all=False, horizon=False),
    "expand": lambda: Beam2(20, 4, fix_hold=False, expand_all=True, horizon=True),
    "holdfix+expand": lambda: Beam2(20, 4, fix_hold=True, expand_all=True, horizon=True),
}

def one(args):
    name, level, seed = args
    r = run_episode(VARIANTS[name](), CheeseEnv(level=level), seed, navigate=False)
    return name, level, seed, r.won, r.pieces

if __name__ == "__main__":
    levels = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [2, 3, 5, 10]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    seed0 = 5_000_000
    tasks = [(v, L, seed0 + i) for L in levels for v in VARIANTS for i in range(n)]
    t0 = time.time()
    with Pool(15) as pool:
        res = pool.map(one, tasks, chunksize=1)
    print(f"{len(tasks)} episodes in {time.time()-t0:.0f}s")
    by = {}
    for name, L, seed, won, pieces in res:
        by.setdefault((L, name), {})[seed] = (won, pieces)
    for L in levels:
        print(f"\n== level {L} ({n} seeds) ==")
        base = by[(L, "orig20x4")]
        for v in VARIANTS:
            d = by[(L, v)]
            wins = [p for w, p in d.values() if w]
            wr = len(wins) / n
            mean = st.fmean(wins) if wins else float("nan")
            ci = 1.96 * st.stdev(wins) / len(wins) ** 0.5 if len(wins) > 1 else 0
            # paired vs original on seeds both won
            both = [s for s in d if d[s][0] and base[s][0]]
            diff = st.fmean(d[s][1] - base[s][1] for s in both) if both else float("nan")
            better = sum(1 for s in both if d[s][1] < base[s][1]); worse = sum(1 for s in both if d[s][1] > base[s][1])
            print(f"{v:16s} win {wr:6.1%} | pieces {mean:6.2f} ± {ci:4.2f} | /line {mean/L:5.2f}"
                  f" | paired Δ vs orig {diff:+6.2f} (better {better}, worse {worse}, n={len(both)})")
