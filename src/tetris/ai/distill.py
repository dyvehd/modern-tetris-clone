"""Supervised distillation from the search oracle (fusion-bot style).

Why this exists: measured on this repo, pure cold-start REINFORCE at sane
budgets does not clear the curriculum's first gate — the sampled win rate
climbs (~30% at level 1) but the argmax/greedy policy stays near-random
(~3%), because each (piece x hole-column) state configuration sees roughly
one winning example per iteration. The fusion bot hit the same wall and
solved it the same way: pretrain the policy on decisions of its own
stronger search, then (optionally) fine-tune with policy gradients.

The teacher is any harness agent — the beam search by default. Teacher
decisions are collected through the same observations/candidate encodings
as the policy uses (so the label is a candidate index, no translation),
and trained with a cross-entropy loss over the segmented per-decision
softmax — the identical GPU-batched machinery as the REINFORCE update.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from .agents import BaseAgent
from .cheese import CheeseEnv, Obs, candidate_moves, run_episode
from .policy import encode_candidates
from .search import BeamAgent, OnePlyAgent

TEACHERS = {
    "1ply": lambda: OnePlyAgent(),
    "beam20x4": lambda: BeamAgent(width=20, depth=4),
}


class TeacherRecorder(BaseAgent):
    """A pass-through agent: plays the teacher's decision and records the
    (candidate encoding, teacher index) pair the policy will train on. The
    net is never touched — no torch in the rollout loop."""

    def __init__(self, teacher: BaseAgent, data: list[tuple[np.ndarray, int]]):
        self.name = f"recorder[{teacher.name}]"
        self.teacher = teacher
        self.data = data

    def decide(self, obs: Obs):
        decision = self.teacher.decide(obs)
        moves = candidate_moves(obs)
        idx = None
        for i, (placement, hold) in enumerate(moves):
            if placement == decision.placement and hold == decision.hold:
                idx = i
                break
        if idx is None:
            raise RuntimeError("teacher decision not among the candidate moves")
        self.data.append((encode_candidates(obs, moves), idx))
        return decision


def collect_teacher_data(
    teacher: BaseAgent,
    env: CheeseEnv,
    n_episodes: int,
    seed0: int,
    data: list[tuple[np.ndarray, int]] | None = None,
) -> list[tuple[np.ndarray, int]]:
    """Play ``n_episodes`` with the teacher; for every decision record
    (candidate rows, teacher's chosen index). The observations are exactly
    what the policy would see — on-policy-state, off-policy-action data."""
    data = data if data is not None else []
    recorder = TeacherRecorder(teacher, data)
    for i in range(n_episodes):
        run_episode(recorder, env, seed0 + i, navigate=False)
    return data


@dataclass
class DistillStats:
    episodes: int
    decisions: int
    loss: float
    accuracy: float
    seconds: float


class DistillTrainer:
    """Cross-entropy on teacher decisions, GPU-batched: one forward/
    backward over every decision of the batch, segmented softmax per
    decision (the same machinery as the REINFORCE update).

    ``chunk_decisions`` caps candidate rows per forward/backward — a
    4 GB laptop GPU OOMs on a full 60k-decision pass (measured), while a
    single chunk fits comfortably; each ``train_batch`` call sweeps the
    data in chunks, one optimizer step per chunk (shuffled by ``rng``).
    """

    def __init__(
        self,
        net: PolicyNet,
        lr: float = 1e-3,
        device: str = "cpu",
        chunk_decisions: int = 2000,
        rng: np.random.Generator | None = None,
    ):
        self.net = net.to(device)
        self.lr = lr
        self.device = device
        self.chunk_decisions = chunk_decisions
        self.rng = rng or np.random.default_rng(0)
        self.opt = torch.optim.Adam(net.parameters(), lr=lr)

    def train_batch(self, data: list[tuple[np.ndarray, int]]) -> DistillStats:
        """One epoch over the data in shuffled chunks."""
        t0 = time.time()
        if not data:
            return DistillStats(0, 0, 0.0, 0.0, 0.0)
        order = np.arange(len(data))
        self.rng.shuffle(order)
        total_loss = 0.0
        total_acc = 0.0
        n_chunks = 0
        n_dec = 0
        i = 0
        while i < len(order):
            chunk = [data[j] for j in order[i : i + self.chunk_decisions]]
            stats = self._train_chunk(chunk)
            total_loss += stats.loss * len(chunk)
            total_acc += stats.accuracy * len(chunk)
            n_dec += len(chunk)
            n_chunks += 1
            i += self.chunk_decisions
        return DistillStats(
            episodes=n_dec,
            decisions=n_dec,
            loss=total_loss / n_dec,
            accuracy=total_acc / n_dec,
            seconds=time.time() - t0,
        )

    def _train_chunk(self, data: list[tuple[np.ndarray, int]]) -> DistillStats:
        sizes = [x.shape[0] for x, _ in data]
        x = torch.as_tensor(
            np.concatenate([x for x, _ in data], axis=0), dtype=torch.float32, device=self.device
        )
        group = np.concatenate(
            [np.full(s, i, dtype=np.int64) for i, s in enumerate(sizes)]
        )
        group_t = torch.as_tensor(group, device=self.device)
        n_groups = len(sizes)
        bases = np.concatenate(([0], np.cumsum(sizes)[:-1]))
        targets = torch.as_tensor(
            bases + np.asarray([t for _, t in data]), dtype=torch.long, device=self.device
        )

        scores = self.net(x)
        max_g = torch.full((n_groups,), float("-inf"), device=self.device)
        max_g.scatter_reduce_(0, group_t, scores, reduce="amax", include_self=True)
        exp_s = torch.exp(scores - max_g[group_t])
        sumexp = torch.zeros(n_groups, dtype=torch.float32, device=self.device)
        sumexp.index_add_(0, group_t, exp_s)
        logz = max_g + torch.log(sumexp)

        chosen_logp = scores[targets] - logz[group[targets.cpu().numpy()]]
        loss = -chosen_logp.mean()
        acc = _group_accuracy(scores, group_t, n_groups, targets)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        return DistillStats(
            episodes=len(data),
            decisions=len(data),
            loss=float(loss.item()),
            accuracy=acc,
            seconds=0.0,
        )


def _group_accuracy(scores, group_t, n_groups, targets):
    """argmax within each decision group vs the teacher's choice (ties
    count as correct — a group's max row IS the prediction)."""
    device = scores.device
    max_g = torch.full((n_groups,), float("-inf"), device=device)
    max_g.scatter_reduce_(0, group_t, scores, reduce="amax", include_self=True)
    target_is_max = scores[targets] == max_g[group_t[targets]]
    return float(target_is_max.float().mean().item())


def _group_accuracy_strict(net, data, device: str = "cpu") -> float:
    """Strict (tie-unsatisfying) held-out accuracy: the net's unique
    argmax per decision must equal the teacher's index exactly. Used for
    held-out evaluation during distillation — the training-time
    ``_group_accuracy`` counts ties as correct, which overstates fit on
    the frozen eval split."""
    net = net.to(device)
    net.eval()
    correct = 0
    with torch.no_grad():
        for x, target in data:
            scores = net(torch.as_tensor(x, dtype=torch.float32, device=device))
            a = int(torch.argmax(scores).item())
            correct += int(a == target)
    acc = correct / len(data) if data else 0.0
    net.train()
    return acc
