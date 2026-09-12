"""Round-2 deep diagnostic: the argmin-gap structure of V2's confusions.

The pure-V beam picks the argmin candidate. A mean rank of ~15 is fine
if the argmin is *near-tied* with the teacher's pick; collapse happens
if the argmin is sometimes catastrophically WRONG (V2 confidently
prefers a losing move). This probe measures, over teacher episodes:
- distribution of q(argmin) - q(teacher's pick) — how "cheaper" V2
  thinks its own pick is than the move that actually wins;
- the worst confusions and the actual label values involved.
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


def main():
    net = load_value(Path("models/value/value_round2.json"), device="cpu")
    env = CheeseEnv(level=10)
    qgaps = []
    confusions = []
    n_dec = 0
    for seed in [800_010_000 + i for i in range(10)]:
        teacher = TEACHERS["beam20x4"]()

        class Rec:
            name = "rec"

            def decide(self, obs):
                nonlocal n_dec
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
                amin = int(np.argmin(q))
                qgaps.append(float(q[amin] - q[played]))
                if q[amin] < q[played] - 0.5:
                    confusions.append((seed, amin, played, float(q[amin]), float(q[played])))
                n_dec += 1
                return dec

        run_episode(Rec(), env, seed, navigate=False)

    qgaps = np.asarray(qgaps)
    print(f"decisions: {n_dec}", flush=True)
    print(
        f"q(argmin) - q(teacher): mean {qgaps.mean():+.3f} median {np.median(qgaps):+.3f} "
        f"p10 {np.percentile(qgaps, 10):+.3f} p90 {np.percentile(qgaps, 90):+.3f}",
        flush=True,
    )
    print(f"argmin 'cheaper' than teacher by >0.5 pieces: {len(confusions)} cases", flush=True)
    for c in confusions[:10]:
        print(f"  seed {c[0]}: argmin#{c[1]} q̂ {c[3]:.2f} vs teacher#{c[2]} q̂ {c[4]:.2f}", flush=True)


if __name__ == "__main__":
    main()
