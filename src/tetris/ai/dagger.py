"""DAgger-style iterative distillation: fix the compounding-error gap.

Pure behavioral cloning (``distill.py``) mimics the teacher on the
*teacher's* trajectories. When the policy errs even slightly, it lands on
board states the teacher never visited — and its behavior there is
unguided, so errors compound into topouts (measured: the distilled policy
clears level 2 at 99% but trails the beam's pieces-per-line and loses ~1-3%
of episodes to topouts the teacher would have won). DAgger (Ross et al.
2011) closes exactly this: alternate (1) roll the *current* policy, (2)
at every visited state, ask the teacher for its move (the label), (3)
train on those policy-state / teacher-action pairs. The state distribution
becomes the policy's own, the teacher supervises everywhere the policy can
stray.

A DAgger round here is: run ``PolicyAgent`` for ``episodes``, but let a
``TeacherRecorder`` that *plays the policy's move while recording the
teacher's* collect the labels — the mixture of policy play and teacher
correction that defines DAgger's data. ``beta`` schedules how often the
teacher's own move is played instead (early rounds play teacher moves more
often, later rounds play policy moves — the standard DAgger decay).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from .agents import BaseAgent
from .cheese import CheeseEnv, Decision, Obs, candidate_moves, run_episode
from .distill import DistillTrainer
from .policy import PolicyAgent, PolicyNet, encode_candidates


class DAggerRecorder(BaseAgent):
    """Plays a mixture of policy and teacher moves while recording the
    TEACHER's chosen index at every visited state — whatever is played,
    the label is the teacher's correction."""

    def __init__(
        self,
        policy: PolicyAgent,
        teacher: BaseAgent,
        data: list[tuple[np.ndarray, int]],
        beta: float,
        rng: np.random.Generator,
    ):
        self.name = f"dagger[{policy.name}|{teacher.name}]"
        self.policy = policy
        self.teacher = teacher
        self.data = data
        self.beta = beta  # probability of playing the TEACHER's move
        self.rng = rng

    def decide(self, obs: Obs):
        moves = candidate_moves(obs)
        x = encode_candidates(obs, moves)
        teacher_decision = self.teacher.decide(obs)
        # the teacher's index among the policy's candidate rows
        idx = None
        for i, (placement, hold) in enumerate(moves):
            if placement == teacher_decision.placement and hold == teacher_decision.hold:
                idx = i
                break
        if idx is None:
            raise RuntimeError("teacher decision not among the candidate moves")
        self.data.append((x, idx))
        if self.rng.random() < self.beta:
            return teacher_decision
        # play the policy's own move: the rollout stays on the policy's
        # state distribution while the label stays the teacher's
        device = next(self.policy.net.parameters()).device
        with torch.no_grad():
            scores = self.policy.net(torch.as_tensor(x, dtype=torch.float32, device=device))
            probs = torch.softmax(scores, dim=0)
            chosen = int(torch.multinomial(probs, 1).item())
        placement, hold = moves[chosen]
        return Decision(placement, hold=hold)


from .cheese import Decision  # noqa: E402  (used in DAggerRecorder.decide)


@dataclass
class DAggerStats:
    round_num: int
    beta: float
    episodes: int
    decisions: int
    teacher_move_rate: float  # fraction of states where the teacher's move was played
    loss: float
    accuracy: float
    seconds: float


def dagger_round(
    net: PolicyNet,
    policy: PolicyAgent,
    teacher: BaseAgent,
    env: CheeseEnv,
    episodes: int,
    seed0: int,
    beta: float,
    data: list[tuple[np.ndarray, int]],
    trainer: DistillTrainer,
    rng: np.random.Generator,
    epochs: int = 1,
) -> DAggerStats:
    """One DAgger round: collect ``episodes`` policy-distribution states
    with teacher labels (mixture play, beta = teacher-move probability),
    then ``epochs`` distillation passes over the full accumulated data."""
    t0 = time.time()
    recorder = DAggerRecorder(policy, teacher, data, beta, rng)
    for i in range(episodes):
        run_episode(recorder, env, seed0 + i, navigate=False)
    stats = DAggerStats(
        round_num=0,
        beta=beta,
        episodes=episodes,
        decisions=len(data),
        teacher_move_rate=beta,
        loss=0.0,
        accuracy=0.0,
        seconds=time.time() - t0,
    )
    for _ in range(epochs):
        train_stats = trainer.train_batch(data)
        stats.loss = train_stats.loss
        stats.accuracy = train_stats.accuracy
    stats.seconds = time.time() - t0
    return stats
