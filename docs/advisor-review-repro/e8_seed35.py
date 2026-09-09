import sys; sys.path.insert(0,"src"); sys.path.insert(0,"docs/advisor-review-repro")
from tetris.ai.cheese import CheeseEnv, run_episode
from tetris.ai.search import BeamAgent
from beam2 import Beam2
for name, ag in (("orig", BeamAgent(20,4)), ("fixed", Beam2(20,4)), ("fixed navigate", Beam2(20,4))):
    r = run_episode(ag, CheeseEnv(level=2), 35, navigate=name.endswith("navigate"))
    print(name, r.won, r.pieces)
