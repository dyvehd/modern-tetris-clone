import sys, time, statistics as st
sys.path.insert(0, "src"); sys.path.insert(0, "docs/advisor-review-repro")
from multiprocessing import Pool
from tetris.ai.cheese import CheeseEnv, run_episode
from beam2 import Beam2
CFG = {"w5d2": (5,2), "w10d3": (10,3), "w20x4": (20,4), "w40x5": (40,5), "w80x6": (80,6)}
def one(a):
    name, seed = a; w, d = CFG[name]; t0=time.time()
    r = run_episode(Beam2(w, d), CheeseEnv(level=10), seed, navigate=False)
    return name, r.won, r.pieces, time.time()-t0
if __name__ == "__main__":
    n = 40; tasks = [(k, 5_000_000+i) for k in CFG for i in range(n)]
    with Pool(15) as p: res = p.map(one, tasks, chunksize=1)
    for k in CFG:
        rs = [r for r in res if r[0]==k]; wins=[r[2] for r in rs if r[1]]
        print(f"L10 fixed beam {k}: win {len(wins)/n:.0%} pieces {st.fmean(wins):.2f} ± {1.96*st.stdev(wins)/len(wins)**.5:.2f} | /line {st.fmean(wins)/10:.2f} | {st.fmean(r[3] for r in rs):.1f}s/episode")
