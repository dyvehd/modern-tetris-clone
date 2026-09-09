# Advisor review — reproduction scripts

Companion to [../cheese-ai-advisor-review.md](../cheese-ai-advisor-review.md).
Run everything from the repository root with `.venv/py.sh docs/advisor-review-repro/<script>`.
All scripts are CPU-only (e7 uses the GPU if present), write no checkpoints and
touch no production code. They use a 15-process fork pool; lower `Pool(15)` on
smaller machines.

| script | what it measures | runtime (16-core laptop) |
|---|---|---|
| `beam2.py` | the corrected beam teacher used by the experiments (library, not a script) | — |
| `e2_teacher_ab.py L1,L2 N` | original vs corrected beam, paired on the same seeds | 30 s (L2,3 ×30) · 10 min (L5,10 ×100) |
| `e3_representation.py` | encoding collisions and hidden-preview dependence on teacher-visited states | 3 min |
| `e4_label_noise.py` | beam-vs-1-ply agreement, near-tie margins, preview dependence of the corrected teacher | 5 min |
| `e5_checkpoint.py` | shipped `cheese_policy_L1.json` audit + material-conservation audit | 4 min |
| `e6_scaling.py` | corrected beam strength vs width/depth at L10 | 10 min |
| `e7_afterstate.py` | current vs afterstate encoding, matched data and optimizer steps | 3 min |
| `e8_seed35.py` | the other review's L2/seed-35 witness under the corrected beam | 5 s |

`results_e2_L5_L10.txt` is the saved output of the 100-seed teacher A/B.
