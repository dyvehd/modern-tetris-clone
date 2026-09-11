"""Round-2 value data collection — blockfish-supervised contrast labels.

Round 1's measured ceiling: single-visit MC returns calibrate across
boards but carry no within-board signal — every candidate of a decision
shares one realized return, so V cannot learn which placement is better
on this board. Round 2 fixes the LABEL, not the architecture:

1. blockfish (SOTA cheese B*) plays the episode on the harness; every
   visited decision records every candidate's v7 afterstate row and
   blockfish's raw per-candidate rating (lower = better; ~10 rating
   units ≈ 1 piece — its piece_penalty; terminal traces are exact piece
   counts).
2. Root sibling continuations: at each episode's FIRST decision, the
   best-rated alternatives are forced on a ``Game.clone()`` and
   blockfish plays out the rest — realized piece gaps vs the played
   line are measured multi-visit cost labels, exact counterfactuals on
   the same RNG stream.

Labels per candidate row of the root decision:
- played:    q = realized pieces from the root (the MC return)
- siblings:  q = continuation's pieces from the root (forced piece
             included), failed = continuation outcome
- unvisited: q = played q + min((rating - best)/10, 3) — blockfish's
             piece-gap estimate as a soft prior, capped
Non-root decisions keep the V1 labels (per-decision MC return for all
candidates) — the across-board calibration signal round 1 proved good.

One BlockfishAgent (own CDLL) per worker; the Rust search is
single-threaded and stateless between calls. Output npz per source:
x, q, failed, explored, ratings, is_root.

Run:  python3 examples/bf_collect.py --level 10 --episodes 500 \
        --seed0 700000000 --out models/value/round2_L10.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from tetris.ai.cheese import (  # noqa: E402
    CheeseEnv,
    Decision,
    Obs,
    _observe,
    candidate_moves,
    run_episode_from,
)
from tetris.ai.policy import encode_candidates  # noqa: E402

RATING_PER_PIECE = 10.0  # blockfish piece_penalty: 1 piece ≈ 10 rating units
PRIOR_CAP = 3.0          # unvisited candidates: prior gap capped at 3 pieces

_STATE: dict = {}


def _init_pool(level: int, piece_cap: int, search_limit: int, n_sib: int) -> None:
    """Pool initializer: one BlockfishAgent per worker (its own CDLL)."""
    from tetris.ai.blockfish_agent import BlockfishAgent

    _STATE["env"] = CheeseEnv(level=level, piece_cap=piece_cap)
    _STATE["agent"] = BlockfishAgent(search_limit=search_limit)
    _STATE["n_sib"] = n_sib


def _rate_moves(rank, moves) -> np.ndarray:
    """Map blockfish's ranked candidates onto our candidate list.

    ratings[i] = blockfish's raw rating for moves[i] (lower = better),
    +inf when blockfish never considered that placement. Spin variants
    of one cell set share the best rating."""
    by_key: dict = {}
    for piece, hold, cells, score in rank or []:
        key = (piece, hold, tuple(sorted(cells)))
        raw = -score  # the backend negates (score = -rating)
        if key not in by_key or raw < by_key[key]:
            by_key[key] = raw
    ratings = np.full(len(moves), np.inf, dtype=np.float32)
    for i, (placement, hold) in enumerate(moves):
        key = (placement.piece, hold, tuple(sorted(placement.cells)))
        if key in by_key:
            ratings[i] = by_key[key]
    return ratings


class _Recorder:
    """Pass-through that records every visited decision's candidate rows
    and grabs a clone of the engine Game at the root decision (the live
    game only exists inside the episode loop, so the recorder holds a
    reference to the outer Game and clones it on the first decide())."""

    name = "bfrecorder[blockfish]"

    def __init__(self, agent, game):
        self.agent = agent
        self.game = game
        self.decisions: list[dict] = []
        self.root_game = None

    def decide(self, obs: Obs):
        decision = self.agent.decide(obs)
        moves = candidate_moves(obs)
        x = encode_candidates(obs, moves)
        ratings = _rate_moves(self.agent.last_rank, moves)
        played = None
        for i, (placement, hold) in enumerate(moves):
            if placement == decision.placement and hold == decision.hold:
                played = i
                break
        if played is None:
            raise RuntimeError("blockfish decision not among candidate moves")
        if not self.decisions:  # the root — snapshot the engine state
            self.root_game = self.game.clone()
            self.root_moves = moves
            self.root_ratings = ratings
            self.root_played = played
        self.decisions.append(
            {"x": x, "played": played, "k": obs.pieces_placed, "ratings": ratings}
        )
        return decision


class _Forced:
    """Plays one fixed decision, then hands control to the inner agent."""

    name = "forced[blockfish]"

    def __init__(self, placement, hold, agent):
        self.placement = placement
        self.hold = hold
        self.agent = agent
        self.done = False

    def decide(self, obs: Obs):
        if not self.done:
            self.done = True
            return Decision(self.placement, hold=self.hold)
        return self.agent.decide(obs)


def _episode_shard(seed: int) -> list[tuple]:
    """One full episode + root continuations -> packed row tuples.

    Returns a list of (x, q, failed, explored, ratings, is_root) tuples,
    one per row, already labeled."""
    from tetris.engine.game import Game

    agent = _STATE["agent"]
    env: CheeseEnv = _STATE["env"]
    n_sib: int = _STATE["n_sib"]

    game = Game(env.game_config(), seed=seed)
    rec = _Recorder(agent, game)
    result = run_episode_from(game, rec, env, navigate=False)
    pieces_total, failed = result.pieces, not result.won

    rows: list[tuple] = []

    # --- root decision: contrast labels (played / siblings / prior) -----
    root = rec.decisions[0]
    q_base = float(pieces_total - root["k"])
    best = float(np.min(root["ratings"]))
    x0, played0, ratings0 = root["x"], root["played"], root["ratings"]
    # unvisited candidates: soft prior from blockfish's rating gap
    # (finite-rated only); unrated candidates (+inf rating) get the cap
    gap = np.where(
        np.isfinite(ratings0), ratings0 - best, PRIOR_CAP * RATING_PER_PIECE
    )
    q = q_base + np.clip(gap / RATING_PER_PIECE, 0.0, PRIOR_CAP)
    # played + siblings get MEASURED labels
    q[played0] = q_base
    fail = np.zeros(len(q), dtype=np.float32)
    fail[:] = failed
    if n_sib > 0 and rec.root_game is not None:
        finite_order = sorted(
            (i for i in range(len(rec.root_moves)) if np.isfinite(ratings0[i])),
            key=lambda i: ratings0[i],
        )
        n_forced = 0
        for i in finite_order:
            if n_forced >= n_sib:
                break
            if i == played0:
                continue
            placement, hold = rec.root_moves[i]
            g2 = rec.root_game.clone()
            placed0 = g2.pieces_placed
            res = run_episode_from(
                g2, _Forced(placement, hold, agent), env, navigate=False
            )
            cost = float(res.pieces - placed0)
            q[i] = cost
            fail[i] = 0.0 if res.won else 1.0
            n_forced += 1
    explored = np.zeros(len(q), dtype=bool)
    explored[played0] = True
    rows.append((x0, q.astype(np.float32), fail, explored, ratings0, True))

    # --- non-root decisions: V1 labels (per-decision MC return) -----------
    for d in rec.decisions[1:]:
        qd = np.full(len(d["x"]), float(pieces_total - d["k"]), dtype=np.float32)
        fd = np.full(len(d["x"]), failed, dtype=np.float32)
        ed = np.zeros(len(d["x"]), dtype=bool)
        ed[d["played"]] = True
        rows.append((d["x"], qd, fd, ed, d["ratings"], False))
    return rows


def collect_one(
    level: int,
    episodes: int,
    seed0: int,
    workers: int,
    search_limit: int,
    piece_cap: int,
    n_sib: int,
    out_path: str,
) -> None:
    """Collect ``episodes`` rated+continued episodes at ``level``."""
    from multiprocessing import get_context

    t0 = time.time()
    n_workers = workers or 16
    ctx = get_context("forkserver")
    with ctx.Pool(
        n_workers,
        initializer=_init_pool,
        initargs=(level, piece_cap, search_limit, n_sib),
    ) as pool:
        shards = pool.map(_episode_shard, range(seed0, seed0 + episodes), chunksize=1)

    xs, qs, fs, es, rs, irs = [], [], [], [], [], []
    for shard in shards:
        for x, q, f, e, r, is_root in shard:
            xs.append(x)
            qs.append(q)
            fs.append(f)
            es.append(e)
            rs.append(r)
            irs.append(np.full(len(q), is_root, dtype=bool))
    x = np.concatenate(xs)
    q = np.concatenate(qs)
    f = np.concatenate(fs)
    e = np.concatenate(es)
    r = np.concatenate(rs)
    ir = np.concatenate(irs)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        x=x,
        q=q.astype(np.float32),
        failed=f.astype(np.float32),
        explored=e,
        ratings=r,
        is_root=ir,
    )
    n_root = int(ir.sum())
    print(
        f"L{level}: {episodes} eps -> {len(q)} rows ({n_root} root rows, "
        f"fail {f.mean():.3f}) in {time.time() - t0:.0f}s -> {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--episodes", type=int, required=True)
    ap.add_argument("--seed0", type=int, required=True)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--search-limit", type=int, default=50_000)
    ap.add_argument("--piece-cap", type=int, default=400)
    ap.add_argument("--n-sib", type=int, default=3)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    collect_one(
        args.level,
        args.episodes,
        args.seed0,
        args.workers,
        args.search_limit,
        args.piece_cap,
        args.n_sib,
        args.out,
    )
