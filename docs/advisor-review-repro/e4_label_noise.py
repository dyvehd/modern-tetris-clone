"""How arbitrary are teacher labels? (1) near-tie margins under the 1-ply score,
(2) preview dependence of the CORRECTED teacher, (3) beam vs 1-ply agreement."""
import sys, random, statistics as st
sys.path.insert(0, "src"); sys.path.insert(0, "docs/advisor-review-repro")
from dataclasses import replace
from multiprocessing import Pool
from tetris.ai.cheese import CheeseEnv, run_episode, candidate_moves
from tetris.ai.search import BeamAgent, lock_and_count
from tetris.ai.eval import EvalWeights, eval_board
from tetris.ai.agents import BaseAgent
from beam2 import Beam2

W = EvalWeights()

def oneply_scores(obs):
    out = []
    for p, h in candidate_moves(obs):
        rows, lines, dug = lock_and_count(list(obs.rows), p, obs.cheese_on_board)
        s = eval_board(rows, W) + W.lines * lines + (W.win if obs.cheese_dug + dug >= obs.goal else 0)
        out.append(s)
    return out

class Probe(BaseAgent):
    def __init__(self, seed, teacher):
        self.t = teacher; self.rng = random.Random(seed)
        self.n = 0; self.flip = 0; self.flip_eligible = 0; self.agree_1ply = 0
        self.within1 = 0; self.within5 = 0; self.gap = []
    def decide(self, obs):
        moves = candidate_moves(obs)
        d = self.t.decide(obs)
        idx = next(i for i, (p, h) in enumerate(moves) if p == d.placement and h == d.hold)
        s = oneply_scores(obs)
        order = sorted(range(len(s)), key=lambda i: -s[i])
        self.n += 1
        if order[0] == idx: self.agree_1ply += 1
        if len(s) > 1:
            gap = s[order[0]] - s[order[1]]; self.gap.append(gap)
            self.within1 += sum(1 for v in s if s[order[0]] - v <= 1.0) >= 2
            self.within5 += sum(1 for v in s if s[order[0]] - v <= 5.0) >= 2
        if len(obs.queue) > 2:
            tail = list(obs.queue[1:]); self.rng.shuffle(tail)
            if tuple(tail) != tuple(obs.queue[1:]):
                self.flip_eligible += 1
                if self.t.decide(replace(obs, queue=(obs.queue[0], *tail))) != d: self.flip += 1
        return d

def run(args):
    name, level, seed = args
    t = BeamAgent(20, 4) if name == "orig" else Beam2(20, 4)
    p = Probe(seed, t)
    run_episode(p, CheeseEnv(level=level), seed, navigate=False)
    return name, level, p.n, p.flip, p.flip_eligible, p.agree_1ply, p.within1, p.within5, p.gap

if __name__ == "__main__":
    levels = [3, 5, 10]; n = 30
    tasks = [(nm, L, 8_000_000 + i) for nm in ("orig", "fixed") for L in levels for i in range(n)]
    with Pool(15) as pool:
        res = pool.map(run, tasks, chunksize=1)
    for nm in ("orig", "fixed"):
        for L in levels:
            rs = [r for r in res if r[0] == nm and r[1] == L]
            N = sum(r[2] for r in rs); gaps = [g for r in rs for g in r[8]]
            print(f"{nm:5s} L{L}: {N} dec | agrees with 1-ply {sum(r[5] for r in rs)/N:.1%} | "
                  f"root 1-ply near-tie (2+ cands within 1.0 pt) {sum(r[6] for r in rs)/N:.1%}, within 5 pts {sum(r[7] for r in rs)/N:.1%} | "
                  f"median top-2 gap {st.median(gaps):.2f} | flips under hidden-preview permutation "
                  f"{sum(r[3] for r in rs)/max(1,sum(r[4] for r in rs)):.1%}")
