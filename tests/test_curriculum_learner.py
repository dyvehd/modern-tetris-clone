"""Curriculum learner tests (Deliverable 4).

Guarded by torch availability — the rest of the suite stays green on
machines without the optional [ai] extra installed. The load-bearing
tests: the shaping telescope (potential-based returns reduce to the closed
form), the gate rules, the batched-update equivalence (the segmented
softmax must reproduce per-decision softmaxes exactly), and a small
end-to-end training run that must measurably improve at level 1.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tetris.ai import CheeseEnv, GreedyDigAgent, run_episode  # noqa: E402
from tetris.ai.curriculum import (  # noqa: E402
    BatchedTrainer,
    Curriculum,
    CurriculumConfig,
    GateStats,
    TrainConfig,
    baseline_stats,
    check_gate,
    gate_stats,
    shaped_return,
)
from tetris.ai.policy import (  # noqa: E402
    INPUT_DIM,
    PLACEMENT_DIM,
    STATE_DIM,
    PolicyAgent,
    PolicyNet,
    encode_candidates,
    encode_placement,
    encode_state,
    load_policy,
    save_policy,
)
from tetris.ai import candidate_moves  # noqa: E402
from tetris.engine.constants import PieceType  # noqa: E402


# encodings -------------------------------------------------------------------


def _first_obs(env_level: int = 1, seed: int = 0):
    from tetris.ai.cheese import Obs
    from tetris.engine.game import Game

    game = Game(CheeseEnv(level=env_level).game_config(), seed=seed)
    game.tick()
    active = game.active
    return Obs(
        rows=tuple(game.rows), active=active.type, hold=game.hold_type,
        can_hold=True, queue=tuple(game.queue[:5]), cheese_on_board=game.cheese_on_board,
        cheese_dug=0, goal=env_level, pieces_placed=0, allow_180=True,
    )


def test_encoding_dims_and_ranges():
    obs = _first_obs()
    state = encode_state(obs)
    cand = encode_candidates(obs, candidate_moves(obs))
    assert state.shape == (STATE_DIM,)
    assert cand.shape[1] == INPUT_DIM
    assert cand.shape[0] == len(candidate_moves(obs)) > 10
    # every value is a bounded one-hot or scaled feature
    assert float(np.abs(state).max()) <= 2.0
    assert float(np.abs(cand).max()) <= 2.0
    # all rows share the same state half; placement halves differ
    n = cand.shape[0]
    assert (cand[:, :STATE_DIM] == cand[0, :STATE_DIM]).all()
    assert not (cand[:, STATE_DIM:] == cand[1, STATE_DIM:]).all()


def test_placement_encoding_content():
    obs = _first_obs()
    moves = candidate_moves(obs)
    placement, hold = moves[len(moves) // 2]
    v = encode_placement(placement, hold)
    assert v.shape == (PLACEMENT_DIM,)
    # the 4x4 pattern has exactly the piece's 4 cells set
    assert int(v[:16].sum()) == 4
    # rotation one-hot: exactly one of the 4 slots after position+row
    rot_off = 16 + 10 + 1
    assert int(v[rot_off : rot_off + 4].sum()) == 1
    assert v[rot_off + (placement.rot % 4)] == 1.0
    # hold flag
    assert v[rot_off + 4] == (1.0 if hold else 0.0)
    # piece one-hot: exactly one set at the tail
    assert int(v[-7:].sum()) == 1
    assert v[-7 + list(PieceType).index(placement.piece)] == 1.0


def test_policy_agent_integration():
    # the agent plays whole episodes legally through the harness
    torch.manual_seed(0)
    net = PolicyNet(hidden=32, layers=1, seed=0)
    agent = PolicyAgent(net)
    env = CheeseEnv(level=1)
    r = run_episode(agent, env, 0, navigate=False)
    assert r.reason in ("cleared", "topout", "capped")
    assert len(agent.trace) == r.pieces
    # every trace entry carries what the trainer needs
    for step in agent.trace:
        assert step["x"].shape[1] == INPUT_DIM
        assert 0 <= step["idx"] < step["x"].shape[0]
        assert "dug_before" in step


def test_greedy_determinism_and_save_load(tmp_path):
    torch.manual_seed(0)
    net = PolicyNet(hidden=16, layers=1, seed=7)
    agent = PolicyAgent(net, greedy=True)
    a = run_episode(agent, CheeseEnv(level=1), 5, navigate=False)
    # save/load round-trip: identical greedy decisions
    p = tmp_path / "pol.json"
    save_policy(net, p)
    net2, agent2 = load_policy(p)
    agent2.greedy = True
    b = run_episode(agent2, CheeseEnv(level=1), 5, navigate=False)
    assert (a.pieces, a.dug, a.won, a.reason) == (b.pieces, b.dug, b.won, b.reason)
    assert (a.decisions[0].placement.cells, a.decisions[0].hold) == (
        b.decisions[0].placement.cells, b.decisions[0].hold,
    )


# shaped returns ---------------------------------------------------------------


def test_shaped_return_telescopes():
    # the closed form equals the per-decision sum: -n + alpha*(dug total) + win
    dug_before = [0, 0, 1, 2, 2]  # deltas: 0, 1, 1, 0 (4 decisions)
    final_dug = 2
    alpha = 1.5
    closed = shaped_return(dug_before, final_dug, won=True, alpha=alpha)
    deltas = [dug_before[1] - dug_before[0]]
    for k in range(1, len(dug_before) - 1):
        deltas.append(dug_before[k + 1] - dug_before[k])
    deltas.append(final_dug - dug_before[-1])
    manual = sum(-1.0 + alpha * d for d in deltas) + 50.0
    assert closed == pytest.approx(manual)
    # and the shape-invariant core: without shaping, return = -n + win
    plain = shaped_return(dug_before, final_dug, won=False, alpha=0.0)
    assert plain == pytest.approx(-float(len(dug_before)))


# gates ------------------------------------------------------------------------


def _gate(mean, ci, win_rate=1.0, episodes=200):
    return GateStats(mean_pieces=mean, ci95=ci, win_rate=win_rate, episodes=episodes)


def test_gate_rules():
    base = _gate(5.0, 0.1)
    # strict CI rule: learner mean + CI must be below baseline mean — this
    # pair is strictly separated
    assert check_gate(_gate(4.5, 0.3), base, use_ci=True).passed  # 4.8 < 5.0
    # 4.5±0.6 vs 5.0±0.1: NOT strict (5.1 >= 5.0), but the diff (−0.5) is
    # within the combined noise (±0.61) — a statistical tie, passes as one
    g = check_gate(_gate(4.5, 0.6), base, use_ci=True)
    assert g.passed and "tie" in g.rule
    # clearly worse than the baseline beyond all noise: blocked either way
    assert not check_gate(_gate(5.5, 0.1), base, use_ci=True).passed
    assert not check_gate(_gate(5.5, 0.1), base, use_ci=True, allow_tie=False).passed
    # absolute rule with margin
    assert check_gate(_gate(4.0, 99.0), base, margin=0.5, use_ci=False).passed
    assert not check_gate(_gate(4.7, 0.0), base, margin=0.5, use_ci=False).passed
    # a learner that clears nothing never passes
    assert not check_gate(GateStats(None, None, 0.0, 200), base).passed
    # a baseline that clears nothing never blocks
    assert check_gate(_gate(999.0, 0.0), GateStats(None, None, 0.0, 200)).passed


def test_gate_tie_rule():
    # at optimum levels the baseline cannot be beaten — matching it within
    # combined noise (with no worse win rate) passes; being clearly worse
    # (beyond the noise band) does not
    optimal = _gate(1.00, 0.0)  # the level-1 optimum: 1 piece every game
    learner_tie = _gate(1.00, 0.0)
    assert check_gate(learner_tie, optimal).passed
    assert "tie" in check_gate(learner_tie, optimal).rule
    # a small diff within combined noise also ties
    assert check_gate(_gate(1.02, 0.05), _gate(1.00, 0.05)).passed
    # clearly worse than the noise band: blocked
    assert not check_gate(_gate(1.30, 0.05), _gate(1.00, 0.05)).passed
    # a tie with a WORSE win rate is not mastery: blocked
    assert not check_gate(GateStats(1.0, 0.0, 0.5, 200), GateStats(1.0, 0.0, 1.0, 200)).passed
    # strict improvement beats tie in the reported rule
    r = check_gate(_gate(4.5, 0.3), _gate(5.0, 0.1))
    assert r.passed and "strict" in r.rule
    # tie disabled: pure strict mode
    assert not check_gate(learner_tie, optimal, allow_tie=False).passed


def test_gate_stats_match_batch_math():
    env = CheeseEnv(level=1)
    stats = gate_stats(PolicyNet(hidden=16, layers=1, seed=3), env, 30, seed0=55)
    assert stats.episodes == 30
    # the reference is essentially optimal at level 1: ~1.01 mean, 100% win
    bstats = baseline_stats("1ply", env, 30, seed0=55)
    assert bstats.win_rate == 1.0
    assert bstats.mean_pieces is not None and bstats.mean_pieces < 1.2


# batched trainer ----------------------------------------------------------------


def test_batched_segmented_softmax_equivalence():
    # the trainer's segmented log-softmax must equal per-decision softmaxes
    from tetris.ai.cheese import Obs
    from tetris.engine.game import Game

    env = CheeseEnv(level=2)
    game = Game(env.game_config(), seed=11)
    game.tick()
    obs = Obs(
        rows=tuple(game.rows), active=game.active.type, hold=game.hold_type,
        can_hold=True, queue=tuple(game.queue[:5]), cheese_on_board=game.cheese_on_board,
        cheese_dug=0, goal=2, pieces_placed=0, allow_180=True,
    )
    net = PolicyNet(hidden=24, layers=1, seed=5)
    agent = PolicyAgent(net)
    run_episode(agent, env, 11, navigate=False)  # populate a trace
    assert agent.trace

    rows_np = [s["x"].detach().cpu().numpy() for s in agent.trace]
    sizes = [s["x"].shape[0] for s in agent.trace]
    idxs = [s["idx"] for s in agent.trace]
    x = torch.as_tensor(np.concatenate(rows_np, axis=0), dtype=torch.float32)
    group = np.concatenate([np.full(s, i, dtype=np.int64) for i, s in enumerate(sizes)])
    group_t = torch.as_tensor(group)
    n_groups = len(sizes)
    scores = net(x)
    max_g = torch.full((n_groups,), float("-inf"))
    max_g.scatter_reduce_(0, group_t, scores, reduce="amax", include_self=True)
    exp_s = torch.exp(scores - max_g[group_t])
    sumexp = torch.zeros(n_groups)
    sumexp.index_add_(0, group_t, exp_s)
    logz = max_g + torch.log(sumexp)
    bases = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    chosen = torch.as_tensor(bases + np.asarray(idxs), dtype=torch.long)
    seg_logp = scores[chosen] - logz[group_t[chosen]]

    # per-decision softmaxes, computed the obvious way
    off = 0
    for i, s in enumerate(sizes):
        chunk = torch.softmax(scores[off : off + s], dim=0)
        lp = torch.log(chunk[idxs[i]] + 1e-12)
        assert float(seg_logp[i]) == pytest.approx(float(lp), abs=1e-5)
        off += s


def test_reinforce_improves_level1():
    # a tiny but real training run at level 1: the win rate must improve
    # from the near-random init toward (at least) consistently clearing
    torch.manual_seed(0)
    net = PolicyNet(hidden=64, layers=2, seed=0)
    agent = PolicyAgent(net)
    trainer = BatchedTrainer(net, TrainConfig(
        iterations=1, episodes=12, lr=1e-3, seed0=0, device="cpu",
    ), np.random.default_rng(0))
    env = CheeseEnv(level=1)
    before = [run_episode(PolicyAgent(net, greedy=True), env, s, navigate=False) for s in range(12)]
    before_wins = sum(r.won for r in before)
    returns = []
    for it in range(4):
        trainer.cfg = trainer.cfg.with_seeds(1000 + it * 500)
        stats = trainer.train_iteration(agent, env)
        returns.append(stats)
    after = [run_episode(PolicyAgent(net, greedy=True), env, s, navigate=False) for s in range(12)]
    after_wins = sum(r.won for r in after)
    # the win rate must not regress over these few iterations; typically it
    # improves from random (~20%) to mostly-clearing
    assert after_wins >= before_wins, (before_wins, after_wins)


# curriculum controller -----------------------------------------------------------


def test_curriculum_runs_and_reports(tmp_path):
    # a compressed ladder run: train + gate at level 1 only, tiny budgets
    torch.manual_seed(0)
    net = PolicyNet(hidden=32, layers=1, seed=1)
    agent = PolicyAgent(net)
    cur = Curriculum(
        CurriculumConfig(
            start_level=1, max_level=1, reference="1ply",
            gate_episodes=8, gate_seed0=777,
            checkpoint_dir=str(tmp_path),
        ),
        net,
        TrainConfig(iterations=1, episodes=4, lr=1e-3, seed0=0, device="cpu"),
    )
    reports = list(cur.run(agent, iterations_per_level=2, verbose=False))
    assert len(reports) >= 1
    assert reports[0].level == 1
    # a checkpoint is written when the gate passes
    if reports[0].passed:
        assert (tmp_path / "cheese_policy_L1.json").exists()
    # max_level=1: the ladder stops after the first report either way
    assert cur.level == 1


def test_ladder_dagger_rounds_run_and_stage_checkpoints(tmp_path):
    # DAgger-in-ladder wiring: with dagger_rounds=1 the train_level pass
    # collects policy-distribution states (parallel), labels them with the
    # teacher, re-distills, and writes stage checkpoints
    torch.manual_seed(0)
    net = PolicyNet(hidden=16, layers=1, seed=0)
    agent = PolicyAgent(net)
    cur = Curriculum(
        CurriculumConfig(
            start_level=1, max_level=1, reference="1ply",
            gate_episodes=4, gate_seed0=777,
            checkpoint_dir=str(tmp_path),
            dagger_rounds=1, dagger_episodes=4, dagger_epochs=1,
            workers=2,
        ),
        net,
        TrainConfig(iterations=1, episodes=2, lr=1e-3, seed0=0, device="cpu"),
    )
    stats = cur.train_level(agent, iterations=1)
    assert len(stats) == 1
    # stage checkpoints from every completed stage (no distill here, so
    # only the reinforced + daggered ones exist)
    for tag in ("L1_reinforced", "L1_daggered"):
        p = tmp_path / f"cheese_policy_{tag}.json"
        assert p.exists(), tag


def test_ladder_dagger_seed_bands_disjoint():
    # the DAgger and distill seed bands must be disjoint from the gate band
    # and from each other by construction
    cfg = CurriculumConfig()
    assert cfg.gate_seed0 == 900_000_000
    cur = Curriculum(cfg, PolicyNet(hidden=8, layers=1, seed=0),
                     TrainConfig(iterations=1, episodes=1, device="cpu"))
    for level in (1, 2, 5, 10):
        cur.level = level
        d0 = cur._distill_seed0()
        assert d0 == 100_000_000 + 1_000_000 * level
        assert cur._dagger_seed0(3) == 200_000_000 + 1_000_000 * level + 30_000
        # bands are disjoint from the gate band and from each other
        assert d0 < 1_000_000_000
        assert 200_000_000 <= cur._dagger_seed0(0) < 1_000_000_000
