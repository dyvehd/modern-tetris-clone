"""Shipped checkpoint audit + material audit.
- greedy play of models/curriculum/cheese_policy_L1.json at L1..L3 on fresh seeds
- strict-argmax vs tie-inclusive teacher agreement on held-out teacher states
- regret of the policy's choice under the teacher's 1-ply score
- conservation audit for the beam: 4P = G + 10J + R on won games
"""
import sys, json, statistics as st
sys.path.insert(0, "src"); sys.path.insert(0, "docs/advisor-review-repro")
import numpy as np, torch
from pathlib import Path
from tetris.ai.cheese import CheeseEnv, run_episode, run_batch, candidate_moves
from tetris.ai.policy import load_policy, PolicyAgent, encode_candidates
from tetris.ai.search import BeamAgent, lock_and_count
from tetris.ai.eval import EvalWeights, eval_board
from tetris.ai.agents import BaseAgent
from tetris.engine.game import Game
from beam2 import Beam2

torch.set_num_threads(4)
W = EvalWeights()
net, _ = load_policy(Path("models/curriculum/cheese_policy_L1.json"))
print("checkpoint params:", sum(p.numel() for p in net.parameters()))

for L in (1, 2, 3):
    b = run_batch(PolicyAgent(net, greedy=True), CheeseEnv(level=L), 200, seed0=9_500_000)
    print(f"policy_L1 greedy @L{L}: win {b.win_rate:.1%} mean {b.mean_pieces} ci {b.ci95_pieces}")

# held-out teacher states -> strict vs tie agreement, regret
class Rec(BaseAgent):
    def __init__(self): self.t = BeamAgent(20, 4); self.rows = []
    def decide(self, obs):
        d = self.t.decide(obs); moves = candidate_moves(obs)
        idx = next(i for i, (p, h) in enumerate(moves) if p == d.placement and h == d.hold)
        sc = []
        for p, h in moves:
            r, lines, dug = lock_and_count(list(obs.rows), p, obs.cheese_on_board)
            sc.append(eval_board(r, W) + W.lines * lines + (W.win if obs.cheese_dug + dug >= obs.goal else 0))
        self.rows.append((encode_candidates(obs, moves), idx, np.array(sc)))
        return d

for L in (1, 2, 3):
    rec = Rec()
    for i in range(150): run_episode(rec, CheeseEnv(level=L), 9_600_000 + i, navigate=False)
    strict = tie = 0; regrets = []; catastrophic = 0
    for x, idx, sc in rec.rows:
        with torch.no_grad(): s = net(torch.as_tensor(x)).numpy()
        a = int(np.argmax(s)); strict += a == idx; tie += s[idx] == s.max()
        reg = sc[idx] - sc[a]; regrets.append(reg); catastrophic += reg > 20
    n = len(rec.rows)
    print(f"L{L} held-out teacher states n={n}: strict-argmax agreement {strict/n:.1%} | tie-inclusive {tie/n:.1%} | "
          f"mean 1-ply regret of policy choice {st.fmean(regrets):.2f} | regret>20 (hole-class error) {catastrophic/n:.1%}")

# conservation audit
def audit(agent_fn, L, n):
    Js, Rs, Ps = [], [], []
    for i in range(n):
        env = CheeseEnv(level=L); g = Game(env.game_config(), seed=9_700_000 + i)
        r = run_episode(agent_fn(), env, 9_700_000 + i, navigate=False)
        if not r.won: continue
        # replay to get final board: re-run capturing game -- cheap: run again with same decisions
        g.tick(); cells_final = None
        class Replay(BaseAgent):
            def __init__(s): s.k = 0
            def decide(s, obs): d = r.decisions[s.k]; s.k += 1; return d
        # run_episode doesn't return the board; simulate manually
        from tetris.ai.cheese import _observe
        from tetris.engine.game import Action
        k = 0
        while not g.over:
            if g.active is None: g.tick(); continue
            d = r.decisions[k]; k += 1
            if d.hold: g.tick([Action.HOLD])
            g.active.rot, g.active.x, g.active.y = d.placement.rot, d.placement.x, d.placement.y
            g.last_action = None; g.tick([Action.HARD_DROP])
        R = sum(bin(row).count("1") for row in g.rows); J = g.lines - g.cheese_dug
        Js.append(J); Rs.append(R); Ps.append(r.pieces)
        assert 4 * r.pieces == L + 10 * J + R, (r.pieces, L, J, R)
    return st.fmean(Ps), st.fmean(Js), st.fmean(Rs), len(Ps)

for L in (5, 10):
    for name, fn in (("orig beam20x4", lambda: BeamAgent(20, 4)), ("fixed beam2", lambda: Beam2(20, 4))):
        P, J, R, n = audit(fn, L, 30)
        print(f"L{L} {name}: pieces {P:.1f} = (G={L} + 10*J + R)/4 with junk lines J={J:.2f}, residual cells R={R:.1f}  (n={n}; lower bound ceil(G/4)={-(-L//4)})")
