"""Learned placement policy for the cheese race (curriculum Deliverable 4).

GPU-first and torch-native: the same code runs on the local RTX 3050 and,
unchanged, on the RTX Pro 6000 server (``device`` is a constructor knob).
The architecture is *score-per-candidate* — the shape the fusion bot uses
for its policy/value net:

    policy(obs) = softmax over candidates of MLP([ context | afterstate ])

Each candidate row carries **the board after that placement locks and
clears** (plus its lines/dug/win outcome), not the placement's shape
coordinates. Review 2's matched A/B measured why: the old
[state | 4x4-pattern | x-one-hot] encoding collides on distinct placements
(x clipped off-field; vertical-I x = -2..0 identical), and its held-out
agreement saturates at 56% while train climbs (memorization); the
afterstate input reaches 72% held-out with hole-class errors 11.5% ->
0.9% — it is also exactly the input a value function needs (the
cost-to-go direction). The shared context half carries what does not
vary within one decision: all 5 previews, hold, active, and the cheese
counters.

Importing torch is this module's job, not the package's: the rest of
``tetris.ai`` stays importable without it (engine, harness, search).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .agents import BaseAgent
from .cheese import Decision, Obs, candidate_moves
from .search import lock_and_count
from ..engine.constants import FIELD_H, FIELD_W, PieceType

PIECES = list(PieceType)  # index order of the one-hot encodings
PIECE_INDEX = {p: i for i, p in enumerate(PIECES)}

# the visible field: 10x20 occupancy at full cell resolution — a cheese
# hole is ONE cell wide, so any coarser view (max-pooling, first-filled-row
# heights) hides the single most important feature of the dig task
BOARD_ROWS = 20
BOARD_COLS = FIELD_W

STATE_DIM = (
    5 * len(PIECES)  # all 5 preview pieces, order-sensitive (35)
    + len(PIECES)  # hold one-hot (7)
    + len(PIECES)  # active one-hot (7)
    + 3  # cheese_on_board, cheese_dug, goal (scaled)
)
PLACEMENT_DIM = (
    BOARD_ROWS * BOARD_COLS  # board AFTER lock+clear, full resolution (200)
    + 3  # lines/4, cheese dug by the placement/4, win flag
    + 1  # hold flag
)
INPUT_DIM = STATE_DIM + PLACEMENT_DIM


def encode_state(obs: Obs) -> np.ndarray:
    """The per-decision context: previews, hold, active, cheese counters.
    Identical for every candidate of one decision."""
    v = np.zeros(STATE_DIM, dtype=np.float32)
    i = 0
    # all visible previews, in order — the teacher plans over every one of
    # them, so the student must see what the teacher sees (the old
    # encoding carried only queue[0]; two queues that permute the tail
    # were literally indistinguishable to the net)
    for k, piece in enumerate(obs.queue[:5]):
        v[i + k * len(PIECES) + PIECE_INDEX[piece]] = 1.0
    i += 5 * len(PIECES)
    if obs.hold is not None:
        v[i + PIECE_INDEX[obs.hold]] = 1.0
    i += len(PIECES)
    v[i + PIECE_INDEX[obs.active]] = 1.0
    i += len(PIECES)
    v[i] = obs.cheese_on_board / 20.0
    v[i + 1] = obs.cheese_dug / 100.0
    v[i + 2] = obs.goal / 100.0
    return v


def encode_afterstate(obs: Obs, placement, hold: bool) -> np.ndarray:
    """One candidate's features: the board after this placement locks and
    clears, its outcome (lines, cheese dug, win), and the hold flag.
    Distinct placements produce distinct boards — no collisions by
    construction."""
    v = np.zeros(PLACEMENT_DIM, dtype=np.float32)
    rows_after, lines, dug = lock_and_count(list(obs.rows), placement, obs.cheese_on_board)
    i = 0
    base = FIELD_H - BOARD_ROWS
    for br in range(BOARD_ROWS):
        row = rows_after[base + br]
        for bc in range(FIELD_W):
            if (row >> bc) & 1:
                v[i + br * BOARD_COLS + bc] = 1.0
    i += BOARD_ROWS * BOARD_COLS
    v[i] = lines / 4.0
    v[i + 1] = dug / 4.0
    v[i + 2] = 1.0 if obs.cheese_dug + dug >= obs.goal else 0.0
    i += 3
    v[i] = 1.0 if hold else 0.0
    return v


def encode_candidates(obs: Obs, moves: list[tuple]) -> np.ndarray:
    """(n_candidates, INPUT_DIM): each row = [ context | afterstate ]."""
    state = encode_state(obs)
    rows = np.zeros((len(moves), INPUT_DIM), dtype=np.float32)
    for i, (placement, hold) in enumerate(moves):
        rows[i, :STATE_DIM] = state
        rows[i, STATE_DIM:] = encode_afterstate(obs, placement, hold)
    return rows


class PolicyNet(nn.Module):
    """Score-per-candidate MLP: tanh hidden layers, scalar score head.
    Glorot-uniform init — the same recipe as the fusion bot's policy."""

    def __init__(self, hidden: int = 128, layers: int = 2, seed: int | None = None):
        super().__init__()
        dims = [INPUT_DIM] + [hidden] * layers + [1]
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
            if fan_out != 1:
                mods.append(nn.Tanh())
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, INPUT_DIM) -> (N,) scores."""
        return self.net(x).squeeze(-1)


class PolicyAgent(BaseAgent):
    """The learned cheese agent. Scores candidates on the net's device,
    softmaxes (temperature), samples (or argmax when greedy). Records the
    (logprob, entropy, decision-batch) tensors for the REINFORCE update —
    the trainer consumes them after the episode."""

    def __init__(
        self,
        net: PolicyNet,
        temperature: float = 1.0,
        greedy: bool = False,
        name: str = "policy",
        rng: np.random.Generator | None = None,
    ):
        super().__init__()
        self.net = net
        self.temperature = temperature
        self.greedy = greedy
        self.name = name
        self.rng = rng or np.random.default_rng()
        self.trace: list[dict] = []  # per-decision tensors, cleared per episode

    def decide(self, obs: Obs) -> Decision:
        moves = candidate_moves(obs)
        x = encode_candidates(obs, moves)
        device = next(self.net.parameters()).device
        xt = torch.as_tensor(x, dtype=torch.float32, device=device)
        with torch.no_grad():
            scores = self.net(xt)
            logits = scores / self.temperature
            probs = torch.softmax(logits, dim=0)
            idx = int(torch.argmax(probs).item()) if self.greedy else int(
                torch.multinomial(probs, 1).item()
            )
        logprob = float(torch.log(probs[idx] + 1e-12).item())
        entropy = float(-(probs * torch.log(probs + 1e-12)).sum().item())
        self.trace.append(
            {
                "x": xt,  # (n_candidates, INPUT_DIM) on device
                "idx": idx,
                "logprob": logprob,
                "entropy": entropy,
                "dug_before": obs.cheese_dug,  # for exact shaped returns
            }
        )
        placement, hold = moves[idx]
        return Decision(placement, hold=hold)

    def reset_trace(self) -> None:
        self.trace = []


def save_policy(net: PolicyNet, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "hidden": net.net[0].out_features,
        "layers": sum(1 for m in net.net if isinstance(m, nn.Linear)) - 1,
        "state_dict": {
            k: v.detach().cpu().tolist() for k, v in net.state_dict().items()
        },
        "meta": {
            "state_dim": STATE_DIM,
            "placement_dim": PLACEMENT_DIM,
            "input_dim": INPUT_DIM,
        },
    }
    path.write_text(json.dumps(payload))


def load_policy(path: Path, device: str = "cpu") -> tuple[PolicyNet, PolicyAgent]:
    """Load a trained policy: returns (net, agent) with the net on device.

    Refuses checkpoints whose input layout does not match the current
    encoding — the v7 afterstate encoding invalidates every old-stack
    checkpoint (old labels and old dims), and loading one would silently
    mispredict rather than error."""
    payload = json.loads(Path(path).read_text())
    meta = payload.get("meta", {})
    if meta.get("input_dim") != INPUT_DIM:
        raise ValueError(
            f"checkpoint input_dim {meta.get('input_dim')} != current "
            f"{INPUT_DIM} — this checkpoint predates the afterstate "
            "encoding and its labels are invalidated (see "
            "docs/ai-direction-and-results.md); retrain from scratch"
        )
    net = PolicyNet(hidden=payload["hidden"], layers=payload["layers"])
    net.load_state_dict({k: torch.as_tensor(v) for k, v in payload["state_dict"].items()})
    net = net.to(device)
    return net, PolicyAgent(net)
