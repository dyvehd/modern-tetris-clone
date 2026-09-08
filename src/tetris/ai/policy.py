"""Learned placement policy for the cheese race (curriculum Deliverable 4).

GPU-first and torch-native: the same code runs on the local RTX 3050 and,
unchanged, on the RTX Pro 6000 server (``device`` is a constructor knob).
The architecture is *score-per-candidate* — the shape the fusion bot uses
for its policy/value net:

    policy(obs) = softmax over candidates of MLP([ state | placement ])

Every candidate of one decision shares the same state vector; the MLP
scores each "this board, this candidate" pair. Rollouts are engine-bound on
CPU regardless of device (pure-Python stepping + movegen), so the training
loop is **CPU rollout / GPU update**: trajectories are collected through
the harness, then all decisions from all episodes are concatenated into
one big forward/backward batch on the device — the seam PPO and
search-oracle distillation (Deliverable 5) reuse.

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
from ..engine.constants import FIELD_H, FIELD_W, PieceType

PIECES = list(PieceType)  # index order of the one-hot encodings
PIECE_INDEX = {p: i for i, p in enumerate(PIECES)}

# the visible field: 10x20 occupancy at full cell resolution — a cheese
# hole is ONE cell wide, so any coarser view (max-pooling, first-filled-row
# heights) hides the single most important feature of the dig task
BOARD_ROWS = 20
BOARD_COLS = FIELD_W

STATE_DIM = (
    BOARD_ROWS * BOARD_COLS  # occupancy, full resolution, row 0 = visible top
    + FIELD_W  # column heights (0..20, /20)
    + 2 * len(PIECES)  # active, hold one-hots
    + len(PIECES)  # next queue piece
    + 3  # cheese_on_board, cheese_dug, goal (scaled)
    + 1  # pieces_placed (scaled)
)
PLACEMENT_DIM = (
    16  # 4x4 cell occupancy pattern of the piece rotation
    + FIELD_W  # column position one-hot (x, clipped to the field)
    + 1  # normalized landing row
    + 4  # rotation one-hot
    + 1  # hold flag
    + len(PIECES)  # piece one-hot
)
INPUT_DIM = STATE_DIM + PLACEMENT_DIM


def encode_state(obs: Obs) -> np.ndarray:
    """Board + piece context + cheese counters. Fixed length, float32."""
    v = np.zeros(STATE_DIM, dtype=np.float32)
    i = 0
    # full-resolution occupancy of the visible field; rows above the
    # skyline (buffer rows) are not encoded — placements there are rare
    # (tucked setups near topout) and the piece context carries the rest
    base = FIELD_H - BOARD_ROWS
    for br in range(BOARD_ROWS):
        row = obs.rows[base + br]
        for bc in range(FIELD_W):
            if (row >> bc) & 1:
                v[i + br * BOARD_COLS + bc] = 1.0
    i += BOARD_ROWS * BOARD_COLS
    # column heights measured from the floor (0 = empty column) — a
    # complementary view: the hole's column reads one cheese-row lower
    for x in range(FIELD_W):
        for ry in range(FIELD_H - BOARD_ROWS, FIELD_H):
            if (obs.rows[ry] >> x) & 1:
                v[i + x] = (FIELD_H - ry) / 20.0
                break
    i += FIELD_W
    v[i + PIECE_INDEX[obs.active]] = 1.0
    i += len(PIECES)
    if obs.hold is not None:
        v[i + PIECE_INDEX[obs.hold]] = 1.0
    i += len(PIECES)
    if obs.queue:
        v[i + PIECE_INDEX[obs.queue[0]]] = 1.0
    i += len(PIECES)
    v[i] = obs.cheese_on_board / 20.0
    v[i + 1] = obs.cheese_dug / 100.0
    v[i + 2] = obs.goal / 100.0
    i += 3
    v[i] = min(obs.pieces_placed / 200.0, 2.0)
    return v


def encode_placement(placement, hold: bool) -> np.ndarray:
    """One candidate: its 4x4 pattern, position, rotation, hold, piece."""
    v = np.zeros(PLACEMENT_DIM, dtype=np.float32)
    i = 0
    cell_set = set(placement.cells)
    for cy in range(4):
        for cx in range(4):
            # cells are absolute (row, col); normalize to the placement box
            rel = (placement.y + cy, placement.x + cx)
            v[i + cy * 4 + cx] = 1.0 if rel in cell_set else 0.0
    i += 16
    x0 = max(0, min(placement.x, FIELD_W - 1))
    v[i + x0] = 1.0
    i += FIELD_W
    v[i] = min(placement.y / FIELD_H, 1.0)
    i += 1
    v[i + (placement.rot % 4)] = 1.0
    i += 4
    v[i] = 1.0 if hold else 0.0
    i += 1
    v[i + PIECE_INDEX[placement.piece]] = 1.0
    return v


def encode_candidates(obs: Obs, moves: list[tuple]) -> np.ndarray:
    """(n_candidates, INPUT_DIM): each row = [ state | candidate ]."""
    state = encode_state(obs)
    rows = np.zeros((len(moves), INPUT_DIM), dtype=np.float32)
    for i, (placement, hold) in enumerate(moves):
        rows[i, :STATE_DIM] = state
        rows[i, STATE_DIM:] = encode_placement(placement, hold)
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
    """Load a trained policy: returns (net, agent) with the net on device."""
    payload = json.loads(Path(path).read_text())
    net = PolicyNet(hidden=payload["hidden"], layers=payload["layers"])
    net.load_state_dict({k: torch.as_tensor(v) for k, v in payload["state_dict"].items()})
    net = net.to(device)
    return net, PolicyAgent(net)
