"""Representation A/B under matched data and optimizer steps:
  A) current encoding (state | 4x4 pattern, x one-hot, y, rot, hold, piece)  = 274 dims
  B) afterstate encoding (board AFTER lock+clear, lines, dug, hold flag, queue/hold one-hots, counters)
Same net (128x2 tanh), same Adam, same chunking, same epochs. Teacher = original beam20x4
(labels ~1-ply). Held-out strict-argmax accuracy + fraction of chosen moves that are 'hole-class' errors.
"""
import sys, time, statistics as st
sys.path.insert(0, "src")
from multiprocessing import Pool
import numpy as np, torch
from tetris.ai.cheese import CheeseEnv, run_episode, candidate_moves
from tetris.ai.search import BeamAgent, lock_and_count
from tetris.ai.eval import EvalWeights, eval_board
from tetris.ai.policy import encode_candidates, PIECE_INDEX, PolicyNet
from tetris.ai.distill import DistillTrainer
from tetris.ai.agents import BaseAgent
from tetris.engine.constants import FIELD_H, FIELD_W

W = EvalWeights()
AFTER_DIM = 200 + 3 + 1 + 5 * 7 + 7 + 7 + 3

def encode_after(obs, moves):
    ctx = np.zeros(5 * 7 + 7 + 7 + 3, dtype=np.float32)
    for k, q in enumerate(obs.queue[:5]): ctx[k * 7 + PIECE_INDEX[q]] = 1
    if obs.hold is not None: ctx[35 + PIECE_INDEX[obs.hold]] = 1
    ctx[42 + PIECE_INDEX[obs.active]] = 1
    ctx[49:52] = obs.cheese_on_board / 20, obs.cheese_dug / 100, obs.goal / 100
    out = np.zeros((len(moves), AFTER_DIM), dtype=np.float32)
    for i, (p, h) in enumerate(moves):
        rows, lines, dug = lock_and_count(list(obs.rows), p, obs.cheese_on_board)
        for br in range(20):
            r = rows[FIELD_H - 20 + br]
            for c in range(FIELD_W):
                if r >> c & 1: out[i, br * 10 + c] = 1
        out[i, 200:203] = lines / 4, dug / 4, float(obs.cheese_dug + dug >= obs.goal)
        out[i, 203] = float(h)
        out[i, 204:] = ctx
    return out

class Rec(BaseAgent):
    def __init__(self): self.t = BeamAgent(20, 4); self.A = []; self.B = []; self.S = []
    def decide(self, obs):
        d = self.t.decide(obs); moves = candidate_moves(obs)
        idx = next(i for i, (p, h) in enumerate(moves) if p == d.placement and h == d.hold)
        sc = []
        for p, h in moves:
            r, lines, dug = lock_and_count(list(obs.rows), p, obs.cheese_on_board)
            sc.append(eval_board(r, W) + W.lines * lines + (W.win if obs.cheese_dug + dug >= obs.goal else 0))
        self.A.append((encode_candidates(obs, moves), idx)); self.B.append((encode_after(obs, moves), idx)); self.S.append(np.array(sc))
        return d

def collect(seed):
    r = Rec(); run_episode(r, CheeseEnv(level=5), seed, navigate=False); return r.A, r.B, r.S

class Net(PolicyNet):
    def __init__(self, dim, hidden=128, layers=2):
        torch.nn.Module.__init__(self)
        dims = [dim] + [hidden] * layers + [1]; mods = []
        for a, b in zip(dims[:-1], dims[1:]):
            mods.append(torch.nn.Linear(a, b)); mods.append(torch.nn.Tanh())
        mods.pop(); self.net = torch.nn.Sequential(*mods)

def evaluate(net, data, S):
    strict = 0; hole = 0
    with torch.no_grad():
        for (x, idx), sc in zip(data, S):
            a = int(torch.argmax(net(torch.as_tensor(x))).item())
            strict += a == idx; hole += (sc[idx] - sc[a]) > 20
    return strict / len(data), hole / len(data)

if __name__ == "__main__":
    torch.manual_seed(0)
    with Pool(15) as pool: parts = pool.map(collect, range(10_000_000, 10_000_600), chunksize=1)
    A = [d for p in parts for d in p[0]]; B = [d for p in parts for d in p[1]]; S = [s for p in parts for s in p[2]]
    n = len(A); cut = int(n * 0.8)
    print(f"{n} L5 teacher decisions; train {cut}, held-out {n-cut}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for name, data, dim in (("A current", A, A[0][0].shape[1]), ("B afterstate", B, AFTER_DIM)):
        for epochs in (30, 100):
            torch.manual_seed(0); net = Net(dim)
            tr = DistillTrainer(net, lr=1e-3, device=dev, chunk_decisions=500, rng=np.random.default_rng(0))
            t0 = time.time()
            for e in range(epochs): s = tr.train_batch(data[:cut])
            net = net.cpu(); acc, hole = evaluate(net, data[cut:], S[cut:]); tacc, _ = evaluate(net, data[:cut][:1000], S[:1000])
            print(f"{name:13s} {epochs:3d} epochs ({epochs*((cut+499)//500)} updates): train-acc {tacc:.1%} | held-out strict acc {acc:.1%} | hole-class errors {hole:.1%} | {time.time()-t0:.0f}s")
