"""Executable witnesses for cheese-ai-review.md (reviewed commit ee32fb0).

Run from the repository root: .venv/py.sh docs/cheese-ai-review-repro.py

This prints observations, including undesirable behavior; a successful exit
means the witnesses ran, NOT that the implementation is correct. It is an
audit snapshot, not a replacement for regression tests. No models are saved
and no production files are changed. Uses CPU only, no network or workers.
"""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from tetris.ai import search
from tetris.ai.cheese import CheeseEnv, Decision, _observe, candidate_moves, run_episode
from tetris.ai.curriculum import (
    BatchedTrainer, Curriculum, CurriculumConfig, GateStats, IterationStats,
    TrainConfig, check_gate, shaped_return,
)
from tetris.ai.distill import _group_accuracy
from tetris.ai.movegen import enumerate_placements
from tetris.ai.parallel import _dagger_episode, _init_dagger_worker, net_payload
from tetris.ai.pathfinder import find_path
from tetris.ai.policy import (
    INPUT_DIM, PolicyAgent, PolicyNet, encode_candidates, encode_placement,
)
from tetris.engine import board as B
from tetris.engine.constants import PieceType
from tetris.engine.game import Action, Game


def emit(name, **values):
    print(json.dumps({"witness": name, **values}, sort_keys=True), flush=True)


def new_game(level, seed):
    env = CheeseEnv(level=level)
    game = Game(env.game_config(), seed=seed)
    game.tick()
    return env, game, _observe(game, env)


def apply_candidate(game, decision):
    """Use the engine's lock/spawn bookkeeping for a movegen candidate."""
    child = game.clone()
    if decision.hold:
        child.tick([Action.HOLD])
    p = decision.placement
    assert child.active is not None and child.active.type is p.piece
    child.active.rot, child.active.x, child.active.y = p.rot, p.x, p.y
    child.last_action = None
    child.tick([Action.HARD_DROP])
    return child


def encoding_witnesses():
    placements = enumerate_placements([0] * 40, PieceType.I)
    aliases = [
        [a.x, b.x]
        for i, a in enumerate(placements)
        for b in placements[i + 1 :]
        if a.cells != b.cells
        and np.array_equal(encode_placement(a, False), encode_placement(b, False))
    ]
    emit("distinct_I_placements_share_encoding", x_pairs=aliases)

    _, _, obs = new_game(3, 2)
    # Both tails are permutations of the same remaining first-bag pieces.
    other = replace(obs, queue=tuple(PieceType[c] for c in "IOTJS"))
    teacher = search.BeamAgent(width=20, depth=4)
    emit(
        "omitted_previews_change_teacher_label",
        original_queue="".join(p.name for p in obs.queue),
        other_queue="".join(p.name for p in other.queue),
        identical_inputs=bool(np.array_equal(
            encode_candidates(obs, candidate_moves(obs)),
            encode_candidates(other, candidate_moves(other)),
        )),
        different_labels=teacher.decide(obs) != teacher.decide(other),
    )


def hold_witness():
    _, game, obs = new_game(3, 0)
    nodes = []
    original_node = search._Node

    class CaptureNode(original_node):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            nodes.append(self)

    with patch.object(search, "_Node", CaptureNode):
        search.BeamAgent(width=20, depth=1).decide(obs)
    node = next(n for n in nodes if n.first.hold)
    child = apply_candidate(game, node.first)
    emit(
        "empty_hold_child_differs_from_engine",
        placed=node.first.placement.piece.name,
        beam_next=node.queue_rest[0].name,
        engine_next=child.active.type.name,
        beam_can_hold=node.can_hold,
        engine_can_hold=child.can_hold,
    )


def teacher_witness():
    env, game, obs = new_game(2, 35)
    reference = run_episode(search.BeamAgent(width=20, depth=4), env, 35, navigate=False)

    class TwoPieceSolution:
        def __init__(self):
            self.step = 0

        def decide(self, obs):
            # Engine bounding-box coordinates: S then L; no hold.
            target = [(1, 4, 36), (3, 8, 37)][self.step]
            self.step += 1
            return Decision(next(
                p for p in obs.placements() if (p.rot, p.x, p.y) == target
            ))

    witness = run_episode(TwoPieceSolution(), env, 35, navigate=True)
    assert witness.won and witness.pieces == 2
    one_piece_possible = any(
        apply_candidate(game, Decision(p, hold)).won for p, hold in candidate_moves(obs)
    )
    emit(
        "L2_seed35_optimal_two_piece_replay",
        teacher_pieces=reference.pieces,
        witness_pieces=witness.pieces,
        one_piece_win_exists=one_piece_possible,
        inputs=[[a.name for a in path] for path in witness.inputs],
    )


def gate_witnesses():
    unreliable = check_gate(GateStats(1.0, 0.0, 0.01, 300), GateStats(2.0, 0.0, 1.0, 300))
    pieces = np.array([1] * 299 + [40])
    mean = float(pieces.mean())
    ci = float(1.96 * pieces.std(ddof=1) / np.sqrt(len(pieces)))
    noisy = check_gate(GateStats(mean, ci, 1.0, 300), GateStats(1.01, 0.01, 1.0, 300))
    emit("strict_gate_ignores_win_rate", learner_win_rate=0.01, passed=unreliable.passed)
    emit("noisy_policy_passes_tie", mean=mean, ci95=ci, passed=noisy.passed)

    net = PolicyNet(hidden=8, layers=1, seed=0)
    cur = Curriculum(CurriculumConfig(max_level=3), net, TrainConfig(device="cpu"))
    cur.level = 2
    good, bad = GateStats(1.0, 0.0, 1.0, 10), GateStats(None, None, 0.0, 10)
    with (
        patch.object(cur, "train_level", return_value=[]),
        patch.object(cur, "_checkpoint"),
        patch.object(cur, "gate", side_effect=lambda level: check_gate(good if level >= 2 else bad, good)),
    ):
        report = next(cur.run(PolicyAgent(net), iterations_per_level=0, verbose=False))
    emit(
        "retention_failure_still_advances",
        lower_level_passed=report.retention[1].passed,
        report_passed=report.passed,
        next_level=cur.level,
    )


def learning_witnesses():
    base = PolicyNet(hidden=8, layers=1, seed=0)
    x = np.random.default_rng(0).normal(size=(3, INPUT_DIM)).astype(np.float32)
    weights = []
    for coefficient in (0.0, 100.0):
        net = copy.deepcopy(base)
        trainer = BatchedTrainer(
            net, TrainConfig(device="cpu", entropy_coef=coefficient), np.random.default_rng(0)
        )
        trainer.baseline = 0.0
        stats = IterationStats(0, 1, 0.0, 0.0, None, 0.0, 0.0, 0.0, 0.0)
        trainer._update(stats, [x], [0], [3], [1.0], [0], [2.0])
        weights.append(copy.deepcopy(net.state_dict()))
    emit(
        "entropy_coefficient_has_no_gradient_effect",
        coefficients=[0.0, 100.0],
        identical_updated_weights=all(torch.equal(weights[0][k], weights[1][k]) for k in weights[0]),
    )
    emit(
        "return_can_prefer_failure",
        win_100_pieces_L10=shaped_return([0] * 100, 10, True, 1.0),
        failure_20_pieces_zero_dug=shaped_return([0] * 20, 0, False, 1.0),
    )
    emit(
        "accuracy_counts_any_tied_target_as_correct",
        reported_accuracy=_group_accuracy(
            torch.zeros(3), torch.zeros(3, dtype=torch.long), 1, torch.tensor([2])
        ),
        greedy_index=0, teacher_index=2,
    )

    agent = PolicyAgent(PolicyNet(hidden=8, layers=1, seed=0), greedy=True)
    counts = []
    for seed in (0, 1):
        result = run_episode(agent, CheeseEnv(level=1, piece_cap=3), seed, navigate=False)
        counts.append({"pieces_this_episode": result.pieces, "retained_decisions": len(agent.trace)})
    emit("evaluation_retains_candidate_tensors", episodes=counts)

    torch.manual_seed(0)
    net = PolicyNet(hidden=8, layers=1, seed=0)
    _init_dagger_worker(net_payload(net), "1ply", 1, 0.0)
    first = _dagger_episode((0, 123))
    second = _dagger_episode((0, 123))
    same = len(first) == len(second) and all(
        i == j and np.array_equal(a, b) for (a, i), (b, j) in zip(first, second)
    )
    emit("repeated_DAgger_task_seed", identical_data=same, decisions=[len(first), len(second)])


def engine_witnesses():
    emit("initial_garbage_rows", levels={
        str(level): Game(CheeseEnv(level=level).game_config(), seed=0).cheese_on_board
        for level in (1, 2, 9, 10)
    })
    game = Game(CheeseEnv(level=1).game_config(), seed=0)
    game.rows = [0] * 40
    game.rows[17], game.rows[39] = 1 << 4, B.garbage_row(0)
    game.styles = [[-1] * 10 for _ in range(40)]
    game.styles[17][4], game.styles[39] = 1, [-2] * 10
    game.queue = [PieceType.O] * 7
    game.spawn_forced(PieceType.I)
    target = next(p for p in enumerate_placements(game.rows, PieceType.I) if p.rot == 1 and p.x == -2)
    path = find_path(game.rows, PieceType.I, target)
    assert path is not None
    for action in path:
        game.tick([action])
    emit("goal_reached_but_next_spawn_topouts", dug=game.cheese_dug, goal=1, won=game.won)

    emit("garbage_generation_changes_future_piece_stream", next_bags={
        str(level): "".join(p.name for p in Game(CheeseEnv(level=level).game_config(), seed=0).bag.peek(7))
        for level in (1, 2)
    })


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.manual_seed(0)
    for witness in (encoding_witnesses, hold_witness, teacher_witness, gate_witnesses, learning_witnesses, engine_witnesses):
        witness()
