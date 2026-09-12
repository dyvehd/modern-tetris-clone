"""Diagnose the round-2 pure-V collapse.

Held-out pair concordance is 0.60 (chance 0.5) yet the pure-V beam
still random-walks (L10: 0% win). Question 1: where does the TEACHER's
move rank by V2 among the candidates at each decision (round 1 measured
~16/26)? Question 2: what does the q-hat spread look like across
candidates of one board — and how does it correlate with the label
spread? Question 3: when the V-beam picks its argmin, how often does it
match the teacher's move?
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from tetris.ai.cheese import CheeseEnv, candidate_moves, run_episode
from tetris.ai.distill import TEACHERS
from tetris.ai.policy import encode_candidates
from tetris.ai.value import load_value


def probe(net, env, seeds, name):
    ranks, spreads, agrees = [], [], 0
    n_dec = 0
    for seed in seeds:
        teacher = TEACHERS["beam20x4"]()

        class Rec:
            name = "rec"

            def decide(self, obs):
                nonlocal agrees, n_dec
                dec = teacher.decide(obs)
                moves = candidate_moves(obs)
                x = encode_candidates(obs, moves)
                with torch.no_grad():
                    qh, _ = net(torch.as_tensor(x, dtype=torch.float32))
                q = qh.numpy().ravel()
                played = next(
                    i for i, (pl, h) in enumerate(moves)
                    if pl == dec.placement and h == dec.hold
                )
                order = np.argsort(q)
                rank = int(np.where(order == played)[0][0])
                ranks.append(rank)
                spreads.append(float(q.max() - q.min()))
                n_dec += 1
                agrees += int(np.argmin(q) == played)
                return dec

        run_episode(Rec(), env, seed, navigate=False)
    n_cand = len(moves) if moves else 0
    print(
        f"[{name}] decisions {n_dec} | teacher mean rank {np.mean(ranks):.1f} "
        f"(median {np.median(ranks):.0f}, best {min(ranks)}, worst {max(ranks)}) | "
        f"argmin==teacher {agrees / n_dec:.1%} | mean q-hat spread {np.mean(spreads):.2f} pieces",
        flush=True,
    )


def main():
    net = load_value(Path("models/value/value_round2.json"), device="cpu")
    for lvl in [3, 5, 10]:
        env = CheeseEnv(level=lvl)
        probe(net, env, [800_000_000 + lvl * 1000 + i for i in range(5)], f"L{lvl}")


if __name__ == "__main__":
    main()
