"""Representation ceiling of the policy encoding, measured on teacher-visited states.

(a) collision: the teacher's chosen candidate shares an identical input row with
    another candidate that locks different cells (the net cannot prefer one).
(b) preview dependence: permuting queue[1:] (invisible to the policy, which sees
    queue[0] only) changes the teacher's decision -> irreducible label noise.
(c) hidden-y disambiguation: among collided pairs, how many differ only in x.
"""
import sys, random, statistics as st
sys.path.insert(0, "src")
from dataclasses import replace
from multiprocessing import Pool
import numpy as np
from tetris.ai.cheese import CheeseEnv, run_episode, candidate_moves
from tetris.ai.search import BeamAgent
from tetris.ai.policy import encode_candidates
from tetris.ai.agents import BaseAgent


class Probe(BaseAgent):
    def __init__(self, seed):
        self.t = BeamAgent(20, 4); self.rng = random.Random(seed)
        self.n = 0; self.collide = 0; self.collide_teacher_not_first = 0; self.preview_flip = 0
        self.dup_any = 0; self.cands = []
    def decide(self, obs):
        moves = candidate_moves(obs)
        x = encode_candidates(obs, moves)
        d = self.t.decide(obs)
        idx = next(i for i, (p, h) in enumerate(moves) if p == d.placement and h == d.hold)
        self.n += 1; self.cands.append(len(moves))
        rows = [r.tobytes() for r in x]
        same = [i for i, r in enumerate(rows) if r == rows[idx] and moves[i][0].cells != moves[idx][0].cells]
        if same:
            self.collide += 1
            if min(same) < idx: self.collide_teacher_not_first += 1
        if len(set(rows)) < len(rows): self.dup_any += 1
        if len(obs.queue) > 2:
            tail = list(obs.queue[1:]); self.rng.shuffle(tail)
            if tuple(tail) != tuple(obs.queue[1:]):
                d2 = self.t.decide(replace(obs, queue=(obs.queue[0], *tail)))
                if d2 != d: self.preview_flip += 1
        return d

def run(args):
    level, seed = args
    p = Probe(seed)
    run_episode(p, CheeseEnv(level=level), seed, navigate=False)
    return level, p.n, p.collide, p.collide_teacher_not_first, p.preview_flip, p.dup_any, p.cands

if __name__ == "__main__":
    levels = [1, 2, 3, 5, 10]; n = 40
    tasks = [(L, 7_000_000 + i) for L in levels for i in range(n)]
    with Pool(15) as pool:
        res = pool.map(run, tasks, chunksize=1)
    for L in levels:
        rs = [r for r in res if r[0] == L]
        N = sum(r[1] for r in rs); col = sum(r[2] for r in rs); nf = sum(r[3] for r in rs)
        flip = sum(r[4] for r in rs); dup = sum(r[5] for r in rs)
        cands = [c for r in rs for c in r[6]]
        print(f"L{L}: {N} decisions | candidates/decision {st.fmean(cands):.1f} | "
              f"teacher label in an encoding collision {col/N:.1%} (argmax would pick another {nf/N:.1%}) | "
              f"any duplicate rows in the decision {dup/N:.1%} | teacher flips when hidden previews permuted {flip/N:.1%}")
