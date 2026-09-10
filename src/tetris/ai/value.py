"""Cost-to-go value network on afterstates — expert iteration's V.

The reviews' convergence point (both 2026-09-09 reviews, §4 in each),
unblocked by the v7 stack: train a value network on the *afterstate
candidate rows* — the exact input the v7 encoding was built to provide
("exactly the input a value function needs") — and plug it back into the
corrected beam as the leaf evaluator. The search can then improve on
``V``, and ``V`` can be retrained from the improved search (expert
iteration); imitation's ceiling — per-state cloning cannot express the
teacher's planning through hold+queue (runs 7b/8's L3 twice-blocked
diagnosis) — disappears because the *search* does the planning again.

What the net predicts, per candidate row (the v7 encoding:
``[context | afterstate]``, INPUT_DIM 256):

- ``q_pieces``: the cost-to-go — total pieces that will be placed from
  this candidate's lock until the goal clears, its own piece included.
  Grounded by Monte-Carlo returns of corrected-beam episodes (review 2:
  "target = remaining pieces to clear, not the teacher's argmax"); a
  failed episode carries its capped cost, never a dropped label —
  reliability stays a first-class part of the target.
- ``fail``: the probability the episode fails from here (topout or the
  piece cap) — the separate head review 2 asked for, "so that good
  efficiency cannot hide bad reliability."

With ``V(afterstate)`` defined on lock-and-cleared boards and ``V(goal)
= 0``, ``Q(s, a) = 1 + E[V(next)]`` in the limit; regressing the total
directly is what a beam leaf needs (plans compared at a common horizon
by estimated remaining pieces).

Data collection: :class:`MCValueRecorder` plays the (beam) teacher and
records **every** candidate row of every visited decision with the
episode's realized return as the label. The teacher's own played moves
are on-policy rows at full loss weight; the unplayed siblings are
off-policy rows — labeled with the same episode return (the standard
imitation-data caveat, exactly like DAgger's off-distribution states)
— at half weight so the on-policy fit dominates. Failures keep their
labels too: their rows train the failure head, and their (large)
cost-to-go penalizes the move sequences that led to them.

Parallel collection follows ``parallel.py``'s forkserver recipe (plain
fork of a torch parent deadlocks; teachers ship by name; one episode
per task).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .agents import BaseAgent
from .cheese import CheeseEnv, Obs, candidate_moves, run_episode
from .policy import INPUT_DIM, encode_candidates


@dataclass(frozen=True)
class CandidateTarget:
    """One training row: the (context | afterstate) encoding plus the
    Monte-Carlo cost-to-go label and the failure flag the twin heads
    regress. ``explored`` marks whether the teacher actually played this
    candidate (on-policy row, full loss weight) or merely considered it
    (off-policy row, half weight)."""

    x: np.ndarray  # (INPUT_DIM,)
    q_pieces: float  # pieces from this candidate's lock through the episode end
    failed: bool  # the episode failed (topout/capped)
    explored: bool  # the teacher played this candidate


class MCValueRecorder(BaseAgent):
    """Play an agent (the beam teacher) and record every candidate row of
    every visited decision with the episode's eventual Monte-Carlo
    return — the value-net analogue of ``TeacherRecorder``. Decisions are
    buffered until the episode ends because each row's label (the total
    pieces placed from that decision on) is only known then."""

    def __init__(self, agent: BaseAgent, data: list[CandidateTarget]):
        self.name = f"mcrecorder[{agent.name}]"
        self.agent = agent
        self.data = data
        self._pending: list[tuple[np.ndarray, np.ndarray, int]] = []
        # (candidate rows, played mask, pieces already placed at the decision)

    def decide(self, obs: Obs):
        moves = candidate_moves(obs)
        x = encode_candidates(obs, moves)  # (n_candidates, INPUT_DIM)
        decision = self.agent.decide(obs)
        idx = None
        for i, (placement, hold) in enumerate(moves):
            if placement == decision.placement and hold == decision.hold:
                idx = i
                break
        if idx is None:
            raise RuntimeError("teacher decision not among the candidate moves")
        played = np.zeros(x.shape[0], dtype=bool)
        played[idx] = True
        self._pending.append((x, played, obs.pieces_placed))
        return decision

    def flush(self, pieces_total: int, failed: bool) -> int:
        """Commit the buffered decisions now that the episode ended at
        ``pieces_total``: a decision taken after ``k`` pieces were placed
        has cost-to-go ``pieces_total − k`` (the candidate's own piece
        included). Returns the number of rows appended."""
        n = 0
        for x, played, k in self._pending:
            q = float(pieces_total - k)
            for i in range(x.shape[0]):
                self.data.append(CandidateTarget(
                    x=x[i], q_pieces=q, failed=failed, explored=bool(played[i]),
                ))
            n += x.shape[0]
        self._pending = []
        return n


def collect_value_data(
    teacher: BaseAgent,
    env: CheeseEnv,
    n_episodes: int,
    seed0: int,
    data: list[CandidateTarget] | None = None,
) -> list[CandidateTarget]:
    """Play ``n_episodes`` with the teacher; record every candidate row
    of every decision with the episode's realized return as label. The
    sequential reference implementation — the server uses the forkserver
    version in :func:`collect_value_data_parallel`."""
    data = data if data is not None else []
    for i in range(n_episodes):
        recorder = MCValueRecorder(teacher, data)
        result = run_episode(recorder, env, seed0 + i, navigate=False)
        # a won episode's last played row has q = 1 (its own winning
        # piece) — the empirical grounding of "one move from goal"
        recorder.flush(result.pieces, failed=not result.won)
    return data


def _value_shard(args: tuple) -> list[CandidateTarget]:
    """Forkserver worker: one seed with a fresh teacher + recorder (the
    teacher arrives by NAME — pool arguments are pickled, lambdas are
    not; mirrors ``parallel._teacher_shard``)."""
    teacher_name, level, seed = args
    from .distill import TEACHERS

    env = CheeseEnv(level=level)
    teacher = TEACHERS[teacher_name]()
    data: list[CandidateTarget] = []
    recorder = MCValueRecorder(teacher, data)
    result = run_episode(recorder, env, seed, navigate=False)
    recorder.flush(result.pieces, failed=not result.won)
    return data


def collect_value_data_parallel(
    teacher_name: str,
    level: int,
    n_episodes: int,
    seed0: int,
    workers: int = 0,
) -> tuple[list[CandidateTarget], float]:
    """Teacher episodes with candidate-return labels, collected by a
    forkserver pool (one episode per task — episode lengths vary wildly).
    Returns (rows, seconds)."""
    from multiprocessing import get_context
    import os

    n_workers = workers or max(1, (os.cpu_count() or 2) - 1)
    shards = [(teacher_name, level, seed0 + i) for i in range(n_episodes)]
    t0 = time.time()
    ctx = get_context("forkserver")
    with ctx.Pool(n_workers) as pool:
        out = pool.map(_value_shard, shards, chunksize=1)
    data = [row for shard in out for row in shard]
    return data, time.time() - t0


class ValueNet(nn.Module):
    """Twin-headed score-per-candidate MLP on the v7 afterstate rows.
    The shared trunk is the same tanh-MLP shape as ``PolicyNet`` (the
    fusion-bot recipe); the heads split after it:

    - ``q_pieces``: a softplus scalar — cost-to-go is nonneg.
    - ``fail``: a logistic-sigmoid probability.

    Glorot-uniform init, matching ``PolicyNet``.
    """

    def __init__(self, hidden: int = 128, layers: int = 2, seed: int | None = None):
        super().__init__()
        dims = [INPUT_DIM] + [hidden] * layers
        mods: list[nn.Module] = []
        for fan_in, fan_out in zip(dims[:-1], dims[1:]):
            linear = nn.Linear(fan_in, fan_out)
            if seed is not None:  # reproducible init for tests
                g = torch.Generator().manual_seed(seed)
                with torch.no_grad():
                    linear.weight.copy_(
                        (torch.rand(linear.weight.shape, generator=g) * 2 - 1)
                        * torch.sqrt(torch.tensor(6.0 / (fan_in + fan_out)))
                    )
            mods.append(linear)
            mods.append(nn.Tanh())
        self.trunk = nn.Sequential(*mods)
        self.q_head = nn.Linear(hidden, 1)
        self.fail_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        q = nn.functional.softplus(self.q_head(h)).squeeze(-1)  # nonneg pieces
        fail = torch.sigmoid(self.fail_head(h)).squeeze(-1)  # failure prob
        return q, fail


def value_payload(net: ValueNet) -> dict:
    """Picklable snapshot (architecture + plain-list weights), the same
    shape ``save_value`` writes to JSON — fork workers rebuild from it."""
    hidden = net.trunk[0].out_features
    layers = sum(1 for m in net.trunk if isinstance(m, nn.Linear))
    return {
        "hidden": hidden,
        "layers": layers,
        "state_dict": {k: v.detach().cpu().tolist() for k, v in net.state_dict().items()},
        "meta": {"input_dim": INPUT_DIM, "heads": ["q_pieces", "fail"]},
    }


def payload_to_value_net(payload: dict) -> ValueNet:
    net = ValueNet(hidden=payload["hidden"], layers=payload["layers"])
    net.load_state_dict(
        {k: torch.as_tensor(v, dtype=torch.float32) for k, v in payload["state_dict"].items()}
    )
    return net


def save_value(net: ValueNet, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value_payload(net)))


def load_value(path: Path, device: str = "cpu") -> ValueNet:
    """Load a trained value net. Like ``policy.load_policy``, refuses a
    checkpoint whose input layout mismatches the current encoding — old
    dimensions mean old labels, silently mispredicting otherwise."""
    payload = json.loads(Path(path).read_text())
    meta = payload.get("meta", {})
    if meta.get("input_dim") != INPUT_DIM:
        raise ValueError(
            f"checkpoint input_dim {meta.get('input_dim')} != current "
            f"{INPUT_DIM} — this checkpoint predates the v7 afterstate "
            "encoding; retrain from scratch"
        )
    net = payload_to_value_net(payload)
    return net.to(device)


@dataclass
class ValueStats:
    """One epoch's training statistics."""

    loss: float
    q_loss: float
    fail_loss: float
    q_mae: float  # weighted mean |q_hat − q|
    fail_rate_hat: float  # mean predicted failure probability
    seconds: float


class ValueTrainer:
    """Regression of the twin heads on Monte-Carlo returns, GPU-batched
    (all rows of a chunk in one forward/backward — a value row is a
    single candidate, so no segmented softmax is needed).

    Loss: MSE on the softplus q head against pieces-to-go, and binary
    cross-entropy on the fail head, both weighted per row — on-policy
    (teacher-played) rows weigh 1.0, off-policy sibling rows 0.5, and a
    failed episode's rows carry an extra ``fail_boost`` on the q loss so
    the net treats "this line of play lost the episode" as expensive,
    not as "merely took many pieces".
    """

    def __init__(
        self,
        net: ValueNet,
        lr: float = 1e-3,
        device: str = "cpu",
        chunk_rows: int = 65536,
        fail_boost: float = 2.0,
        rng: np.random.Generator | None = None,
    ):
        self.net = net.to(device)
        self.device = device
        self.chunk_rows = chunk_rows
        self.fail_boost = fail_boost
        self.rng = rng or np.random.default_rng(0)
        self.opt = torch.optim.Adam(net.parameters(), lr=lr)

    def train_epoch(
        self,
        data: list[CandidateTarget] | tuple,
    ) -> ValueStats:
        """One shuffled sweep over ``data`` in chunks of ``chunk_rows``
        rows (one Adam step per chunk).

        ``data`` is either a row list or the packed tuple
        ``(x, q, failed, explored)`` the collection stages produce — the
        packed form skips the per-epoch row-stack copy (the rows are
        already one block)."""
        t0 = time.time()
        if isinstance(data, tuple):
            x_all, q_all, f_all, e_all = data
            n = len(q_all)
            if n == 0:
                return ValueStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            w_all = np.where(e_all, 1.0, 0.5).astype(np.float32)
        else:
            n = len(data)
            if n == 0:
                return ValueStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            order = np.arange(n)
            self.rng.shuffle(order)
            x_all = np.stack([data[j].x for j in order])
            q_all = np.asarray([data[j].q_pieces for j in order], dtype=np.float32)
            f_all = np.asarray([data[j].failed for j in order], dtype=np.float32)
            w_all = np.asarray(
                [1.0 if data[j].explored else 0.5 for j in order], dtype=np.float32
            )
        order = np.arange(n)
        self.rng.shuffle(order)
        totals = np.zeros(5, dtype=np.float64)
        i = 0
        while i < n:
            j = min(n, i + self.chunk_rows)
            idx = order[i:j]
            chunk = self._train_chunk(
                x_all[idx], q_all[idx], f_all[idx], w_all[idx]
            )
            rows = j - i
            totals += np.asarray(chunk) * rows
            i = j
        return ValueStats(
            loss=totals[0] / n,
            q_loss=totals[1] / n,
            fail_loss=totals[2] / n,
            q_mae=totals[3] / n,
            fail_rate_hat=totals[4] / n,
            seconds=time.time() - t0,
        )

    def _train_chunk(self, x, q, fail, w) -> tuple[float, ...]:
        xt = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        qt = torch.as_tensor(q, dtype=torch.float32, device=self.device)
        ft = torch.as_tensor(fail, dtype=torch.float32, device=self.device)
        wt = torch.as_tensor(w, dtype=torch.float32, device=self.device)
        # failed rows are precious on the q head too: losing an episode is
        # not "merely slow" — boost their q-loss weight
        qw = wt * torch.where(ft > 0.5, self.fail_boost, 1.0)
        q_hat, fail_hat = self.net(xt)
        q_loss = ((q_hat - qt) ** 2 * qw).sum() / qw.sum().clamp(min=1.0)
        fail_loss = (
            nn.functional.binary_cross_entropy(
                fail_hat.clamp(1e-6, 1 - 1e-6), ft, reduction="none"
            )
            * wt
        ).sum() / wt.sum().clamp(min=1.0)
        loss = q_loss + fail_loss
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        wsum = wt.sum().clamp(min=1.0)
        return (
            float(loss.item()),
            float(q_loss.item()),
            float(fail_loss.item()),
            float(((q_hat - qt).abs() * wt).sum().item() / wsum.item()),
            float(fail_hat.mean().item()),
        )


def evaluate_q_mae(net: ValueNet, data, device: str = "cpu") -> float:
    """Held-out mean absolute error of the q head (all rows) — the
    distillation held-out-accuracy analogue for the value net. Accepts
    the row-list or packed ``(x, q, failed, explored)`` form."""
    net = net.to(device)
    net.eval()
    if isinstance(data, tuple):
        x_all, q_all = data[0], data[1]
        n = len(q_all)
    else:
        n = len(data)
        if n == 0:
            return 0.0
        x_all = np.stack([d.x for d in data])
        q_all = np.asarray([d.q_pieces for d in data], dtype=np.float32)
    if n == 0:
        return 0.0
    errs = []
    with torch.no_grad():
        for i in range(0, n, 65536):
            x = torch.as_tensor(
                x_all[i : i + 65536], dtype=torch.float32, device=device
            )
            q_hat, _ = net(x)
            q = torch.as_tensor(q_all[i : i + 65536], device=device)
            errs.append((q_hat - q).abs().mean().item())
    net.train()
    return float(np.mean(errs))
