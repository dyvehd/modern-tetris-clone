"""Curriculum learning for the cheese-race policy (Deliverable 4).

The user's curriculum, made concrete:

- **Levels**: start at 1 cheese line; advance when the learner passes the
  gate at the current level (and, with retention on, keeps passing easier
  ones — forgetting is measured, not hoped away).
- **Gate**: the learner's mean pieces-to-clear over a fresh seed batch must
  beat the reference baseline (the strongest search agent) with statistical
  separation — learner mean + 95% CI below baseline mean (or an absolute
  margin; see :func:`check_gate`).
- **Reward**: -1 per piece (the objective) + potential-based shaping
  (alpha per cheese line dug) that telescope-sums away across the episode
  (see :func:`shaped_return`), plus a terminal win bonus. Dense gradients
  early; the optimal policy is unchanged by the shaping.
- **Algorithm**: REINFORCE with a moving-average baseline and an entropy
  bonus. Rollouts run on the CPU through the harness; every decision from
  every episode of the iteration is concatenated into ONE forward/backward
  batch on the GPU. The per-decision softmax is segmented by candidate
  group (logsumexp via scatter_reduce / index_add), so the batch is a true
  union of independent decisions — one Adam step per iteration.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import torch

from .cheese import CheeseEnv, run_episode
from .policy import PolicyAgent, PolicyNet, save_policy
from .search import BeamAgent, OnePlyAgent

REFERENCE_AGENTS = {
    "1ply": lambda: OnePlyAgent(),
    "beam20x4": lambda: BeamAgent(width=20, depth=4),
}
WIN_BONUS = 50.0


def shaped_return(
    dug_before: list[int], final_dug: int, won: bool, alpha: float
) -> float:
    """Episode return from the trace's per-decision ``dug_before`` counters.

    Each decision k's dug-delta is dug_before[k+1] - dug_before[k] (the
    last uses final_dug), so the shaped sum telescopes:

        R = sum_k (-1 + alpha * delta_k) + win_bonus
          = -n + alpha * (final_dug - dug_before[0]) + win_bonus

    The closed form is exact — no per-decision replay needed."""
    r = -float(len(dug_before)) + alpha * (final_dug - dug_before[0])
    if won:
        r += WIN_BONUS
    return r


@dataclass
class TrainConfig:
    iterations: int = 60
    episodes: int = 64  # rollouts per iteration (CPU-bound)
    lr: float = 3e-4
    entropy_coef: float = 0.01
    alpha: float = 1.0  # shaping weight (potential-based, telescopes away)
    grad_clip: float = 5.0
    baseline_momentum: float = 0.9
    seed0: int = 0
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    def with_seeds(self, seed0: int) -> "TrainConfig":
        """Fresh rollout seeds — a batch is never reused across iterations
        (reusing seeds would fit the batch, not the task)."""
        return replace(self, seed0=seed0)


@dataclass
class IterationStats:
    iteration: int
    episodes: int
    win_rate: float
    mean_return: float
    mean_pieces: float | None  # over winning episodes; None if nothing cleared
    entropy: float
    grad_norm: float
    loss: float
    seconds: float


@dataclass
class GateStats:
    """Pieces-to-clear statistics over a fixed seed batch (wins only)."""

    mean_pieces: float | None
    ci95: float | None
    win_rate: float
    episodes: int


@dataclass
class GateResult:
    passed: bool
    learner: GateStats
    baseline: GateStats
    rule: str

    @property
    def reason(self) -> str:
        if self.learner.mean_pieces is None:
            return "learner cleared nothing"
        if self.baseline.mean_pieces is None:
            return "baseline cleared nothing (misconfigured reference)"
        return (
            f"learner {self.learner.mean_pieces:.2f}"
            f" (win {self.learner.win_rate:.0%}) vs baseline"
            f" {self.baseline.mean_pieces:.2f} — {self.rule}"
        )


def _pieces_stats(results: list) -> GateStats:
    wins = [r.pieces for r in results if r.won]
    mean = float(np.mean(wins)) if wins else None
    ci = float(1.96 * np.std(wins, ddof=1) / np.sqrt(len(wins))) if len(wins) >= 2 else None
    return GateStats(mean, ci, len(wins) / len(results), len(results))


def gate_stats(net: PolicyNet, env: CheeseEnv, n: int, seed0: int) -> GateStats:
    """The learner's gate statistics, evaluated greedily on fresh seeds."""
    agent = PolicyAgent(net, greedy=True)
    results = [run_episode(agent, env, seed0 + i, navigate=False) for i in range(n)]
    return _pieces_stats(results)


def baseline_stats(reference: str, env: CheeseEnv, n: int, seed0: int) -> GateStats:
    """The reference search agent's statistics on the same seeds."""
    agent = REFERENCE_AGENTS[reference]()
    results = [run_episode(agent, env, seed0 + i, navigate=False) for i in range(n)]
    return _pieces_stats(results)


def check_gate(
    learner: GateStats,
    baseline: GateStats,
    *,
    margin: float = 0.0,
    use_ci: bool = True,
    allow_tie: bool = True,
) -> GateResult:
    """The curriculum gate.

    Passes when EITHER:

    - **strict** (``use_ci``): learner mean + 95% CI sits strictly below
      the baseline mean — a statistically separated improvement; or
    - **tie** (``allow_tie``): the learner matches the baseline within the
      combined noise of both batches AND its win rate is not lower.
      Necessary at levels where the baseline already plays the theoretical
      optimum (level 1: mean 1.00) — strict improvement is impossible
      there, and matching the optimum consistently IS mastery.

    Otherwise (``use_ci=False``): learner mean <= baseline mean - margin.
    A learner that clears nothing never passes; a reference that clears
    nothing never blocks.
    """
    if learner.mean_pieces is None or learner.win_rate == 0.0:
        rule = "cleared nothing"
        return GateResult(False, learner, baseline, rule)
    if baseline.mean_pieces is None:
        rule = "baseline cleared nothing"
        return GateResult(True, learner, baseline, rule)

    if use_ci:
        l_ci = learner.ci95 or 0.0
        b_ci = baseline.ci95 or 0.0
        strict = learner.mean_pieces + l_ci < baseline.mean_pieces
        diff = learner.mean_pieces - baseline.mean_pieces
        noise = 1.96 * float(
            np.sqrt(
                ((l_ci / 1.96) ** 2 if learner.episodes > 1 else 0.0)
                + ((b_ci / 1.96) ** 2 if baseline.episodes > 1 else 0.0)
            )
        ) if (l_ci or b_ci) else 0.0
        tie = (
            allow_tie
            and diff <= noise
            and learner.win_rate >= baseline.win_rate
        )
        if strict:
            rule = "strict: mean+CI<baseline"
            return GateResult(True, learner, baseline, rule)
        if tie:
            rule = "tie: matched the baseline within noise"
            return GateResult(True, learner, baseline, rule)
        rule = "mean+CI<baseline (or tie)"
        return GateResult(False, learner, baseline, rule)

    rule = f"mean<=baseline-{margin:g}"
    passed = learner.mean_pieces <= baseline.mean_pieces - margin
    return GateResult(bool(passed), learner, baseline, rule)


class BatchedTrainer:
    """REINFORCE over whole iterations: CPU rollouts, one GPU-batched
    Adam step per iteration over every decision made."""

    def __init__(self, net: PolicyNet, cfg: TrainConfig, rng: np.random.Generator):
        self.net = net.to(cfg.device)
        self.cfg = cfg
        self.rng = rng
        self.opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
        self.baseline: float | None = None

    def train_iteration(self, agent: PolicyAgent, env: CheeseEnv) -> IterationStats:
        cfg = self.cfg
        t0 = time.time()
        n_wins = 0
        win_pieces: list[int] = []
        # per decision: candidate rows, chosen index, entropy, episode id
        rows_np: list[np.ndarray] = []
        idxs: list[int] = []
        sizes: list[int] = []
        entropies: list[float] = []
        dec_episode: list[int] = []  # episode each decision belongs to
        returns: list[float] = []  # per EPISODE (indexed by episode id)

        for ep in range(cfg.episodes):
            agent.trace.clear()
            res = run_episode(agent, env, cfg.seed0 + ep, navigate=False)
            if res.won:
                n_wins += 1
                win_pieces.append(res.pieces)
            dug_before = [s["dug_before"] for s in agent.trace]
            if not dug_before:
                continue  # no decision was made (instant block-out)
            returns.append(shaped_return(dug_before, res.dug, res.won, cfg.alpha))
            for step in agent.trace:
                rows_np.append(step["x"].detach().cpu().numpy())
                idxs.append(step["idx"])
                sizes.append(int(step["x"].shape[0]))
                entropies.append(step["entropy"])
                dec_episode.append(ep)

        mean_ret = float(np.mean(returns)) if returns else 0.0
        self.baseline = (
            mean_ret if self.baseline is None
            else cfg.baseline_momentum * self.baseline
            + (1 - cfg.baseline_momentum) * mean_ret
        )

        stats = IterationStats(
            iteration=0,  # set by the caller
            episodes=cfg.episodes,
            win_rate=n_wins / cfg.episodes,
            mean_return=mean_ret,
            mean_pieces=float(np.mean(win_pieces)) if win_pieces else None,
            entropy=float(np.mean(entropies)) if entropies else 0.0,
            grad_norm=0.0,
            loss=0.0,
            seconds=time.time() - t0,
        )
        if not rows_np:
            return stats
        self._update(stats, rows_np, idxs, sizes, entropies, dec_episode, returns)
        return stats

    def _update(
        self,
        stats: IterationStats,
        rows_np: list[np.ndarray],
        idxs: list[int],
        sizes: list[int],
        entropies: list[float],
        dec_episode: list[int],
        returns: list[float],
    ) -> None:
        """One batched forward/backward: every decision of the iteration,
        segmented softmax per decision group, advantage per episode."""
        cfg = self.cfg
        device = cfg.device
        x = torch.as_tensor(np.concatenate(rows_np, axis=0), dtype=torch.float32, device=device)

        # group id per candidate row (a decision = one softmax group)
        group = np.concatenate(
            [np.full(size, i, dtype=np.int64) for i, size in enumerate(sizes)]
        )
        group_t = torch.as_tensor(group, device=device)
        n_groups = len(sizes)

        # chosen row per decision: the group's base offset + the chosen index
        bases = np.concatenate(([0], np.cumsum(sizes)[:-1]))
        chosen = torch.as_tensor(bases + np.asarray(idxs), dtype=torch.long, device=device)

        # per-decision advantage: the episode's return minus the baseline
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=device)
        dec_ep_t = torch.as_tensor(dec_episode, dtype=torch.long, device=device)
        adv = ret_t[dec_ep_t] - self.baseline

        # segmented log-softmax: logsumexp within each decision group
        scores = self.net(x)
        max_g = torch.full((n_groups,), float("-inf"), device=device)
        max_g.scatter_reduce_(0, group_t, scores, reduce="amax", include_self=True)
        exp_s = torch.exp(scores - max_g[group_t])
        sumexp = torch.zeros(n_groups, dtype=torch.float32, device=device)
        sumexp.index_add_(0, group_t, exp_s)
        logz = max_g + torch.log(sumexp)
        chosen_logp = scores[chosen] - logz[group_t[chosen]]

        ent_t = torch.as_tensor(entropies, dtype=torch.float32, device=device)
        loss = -(adv * chosen_logp).mean() - cfg.entropy_coef * ent_t.mean()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        stats.grad_norm = float(
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.grad_clip).item()
        )
        self.opt.step()
        stats.loss = float(loss.item())


@dataclass
class CurriculumConfig:
    """The ladder: which levels, which reference, how many gate episodes,
    retention (re-gate easier levels), warm-start distillation, DAgger
    refinement rounds, and checkpointing.

    ``distill_episodes``: when > 0, each new level warm-starts with
    supervised distillation from the reference search agent (fresh seeds)
    before REINFORCE begins — measured on this repo, cold-start REINFORCE
    does not clear the level-1 gate at sane budgets, while a distilled
    policy reaches ~4/5 of the teacher's win rate. Set 0 for pure RL.

    ``dagger_rounds``: after REINFORCE, run this many DAgger refinement
    rounds at the current level — roll the policy (mixture play, beta
    0.8 -> 0 by the standard decay), label every visited state with the
    teacher's move, re-distill. Measured at level 2: closes the rare-state
    topout gap that pure behavioral cloning leaves (2.71 -> 2.51 mean
    pieces at 99% win). Each round's data is collected in parallel when
    ``workers`` > 1.

    Seed bands (structurally disjoint by construction):

    - training:    ``train_seed0`` upward
    - distill:     ``distill_seed0`` upward (per level: +1M per level)
    - DAgger:      ``dagger_seed0`` upward (per level: +1M per level)
    - gate:        ``gate_seed0`` upward — never overlaps training data
    """

    start_level: int = 1
    max_level: int = 10
    level_step: int = 1
    reference: str = "beam20x4"  # REFERENCE_AGENTS key; also the teacher
    gate_episodes: int = 200
    gate_margin: float = 0.0
    gate_use_ci: bool = True
    gate_allow_tie: bool = True  # match-the-baseline passes at optimum levels
    retention: bool = True  # re-gate every easier level on advancement
    gate_seed0: int = 900_000_000  # fresh seeds for gates (disjoint band)
    checkpoint_dir: str = "models/curriculum"
    distill_episodes: int = 0  # teacher episodes per level (0 = off)
    distill_epochs: int = 60
    distill_lr: float = 2e-3
    distill_chunk_decisions: int = 20_000  # GPU batch size (2k fit a 4GB laptop; the server has 96)
    distill_level_window: int = 3  # mix teacher data from level-1 .. level+window-1
    dagger_rounds: int = 0  # DAgger refinement rounds per level (0 = off)
    dagger_episodes: int = 400  # policy episodes per DAgger round
    dagger_beta0: float = 0.8  # first round's teacher-move probability
    dagger_beta_decay: float = 0.6  # multiplicative decay per round
    dagger_epochs: int = 2  # distillation passes per round
    workers: int = 0  # fork-pool size for parallel collection (0 = all-1)
    parallel_collect: bool = True  # False: sequential collection (fallback)


@dataclass
class LevelReport:
    level: int
    passed: bool
    gate: GateResult
    retention: dict[int, GateResult] = field(default_factory=dict)


class Curriculum:
    """The curriculum controller: train at the current level, gate against
    the reference, advance (or stay) per the user's rule.

    Usage::

        cur = Curriculum(CurriculumConfig(), net, TrainConfig(...))
        for report in cur.run(agent):
            ...  # LevelReport per level, checkpoints on advancement
    """

    def __init__(
        self,
        cur_cfg: CurriculumConfig,
        net: PolicyNet,
        train_cfg: TrainConfig,
        rng: np.random.Generator | None = None,
    ):
        self.cfg = cur_cfg
        self.net = net
        self.train_cfg = train_cfg
        self.rng = rng or np.random.default_rng()
        self.level = cur_cfg.start_level
        # one trainer (and one Adam state) persists across the whole ladder
        self.trainer = BatchedTrainer(net, train_cfg, self.rng)

    def train_level(self, agent: PolicyAgent, iterations: int) -> list[IterationStats]:
        """Train at the current level: optional distillation warm-start,
        REINFORCE with fresh rollout seeds per iteration, then optional
        DAgger refinement rounds (policy-distribution states, teacher
        labels). A stage checkpoint is saved between stages so a crash
        never loses a converged level."""
        out: list[IterationStats] = []
        if self.cfg.distill_episodes > 0:
            self._distill_level(agent)
            self._stage_checkpoint(agent, f"L{self.level}_distilled")
        for it in range(iterations):
            self.trainer.cfg = self.train_cfg.with_seeds(
                self.train_cfg.seed0 + it * self.train_cfg.episodes * 997
            )
            stats = self.trainer.train_iteration(agent, CheeseEnv(level=self.level))
            stats.iteration = it
            out.append(stats)
        if out:
            self._stage_checkpoint(agent, f"L{self.level}_reinforced")
        if self.cfg.dagger_rounds > 0:
            self._dagger_level(agent)
            self._stage_checkpoint(agent, f"L{self.level}_daggered")
        return out

    # stage checkpointing -----------------------------------------------------

    def _stage_checkpoint(self, agent: PolicyAgent, tag: str) -> None:
        path = Path(self.cfg.checkpoint_dir) / f"cheese_policy_{tag}.json"
        save_policy(self.net, path)

    # seed bands ---------------------------------------------------------------

    def _distill_seed0(self) -> int:
        """Fresh teacher seeds for this level: band start + 1M/level."""
        return 100_000_000 + 1_000_000 * self.level

    def _dagger_seed0(self, round_num: int) -> int:
        """Fresh policy-rollout seeds for this DAgger round: band start
        + 1M/level + 10k/round — every round re-rolls fresh boards."""
        return 200_000_000 + 1_000_000 * self.level + 10_000 * round_num

    # warm-start / refinement ---------------------------------------------------

    def _distill_level(self, agent: PolicyAgent) -> None:
        """Warm-start from the reference search on fresh seeds — the
        v5-measured recipe: MIXED levels, not just the current one.

        Why mixed: level-1 episodes are a single decision each, so an
        L1-only dataset is structurally starved (8k episodes = 8k
        decisions for a 202k-param net — measured: the gate saw 80% win
        and topouts, while v5's 62k mixed-level decisions on a 4x
        smaller net hit the L1 optimum). States from harder levels teach
        the same dig skill with 2.5-8x more decisions per episode and
        cover the rare boards a 1-piece episode never visits.
        ``distill_level_window`` levels participate, sharing the episode
        budget equally; seeds stay in the level's distill band."""
        from .distill import TEACHERS, DistillTrainer, collect_teacher_data

        window = max(1, self.cfg.distill_level_window)
        levels = [
            min(l, self.cfg.max_level) for l in range(self.level, self.level + window)
        ]
        levels = sorted(set(levels))
        per_level = max(1, self.cfg.distill_episodes // len(levels))
        data: list[tuple[np.ndarray, int]] = []
        if self.cfg.parallel_collect and self.cfg.workers != 1:
            from .parallel import collect_teacher_data_parallel

            for i, lvl in enumerate(levels):
                part, _stats = collect_teacher_data_parallel(
                    self.cfg.reference, lvl, per_level,
                    seed0=self._distill_seed0() + 100_000 * i, workers=self.cfg.workers,
                )
                data.extend(part)
        else:
            for i, lvl in enumerate(levels):
                part = collect_teacher_data(
                    TEACHERS[self.cfg.reference](), CheeseEnv(level=lvl),
                    per_level, seed0=self._distill_seed0() + 100_000 * i,
                )
                data.extend(part)
        trainer = DistillTrainer(
            self.net, lr=self.cfg.distill_lr, device=self.train_cfg.device,
            chunk_decisions=self.cfg.distill_chunk_decisions,
        )
        for epoch in range(self.cfg.distill_epochs):
            stats = trainer.train_batch(data)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"  distill L{self.level}: epoch {epoch + 1}/{self.cfg.distill_epochs}"
                    f" | {len(data)} decisions | loss {stats.loss:.3f}"
                    f" | teacher-move accuracy {stats.accuracy:.1%}",
                    flush=True,
                )

    def _dagger_level(self, agent: PolicyAgent) -> None:
        """DAgger refinement at the current level: roll the current policy
        (mixture play), label every visited state with the teacher's move,
        re-distill. Data accumulates across rounds, matching the measured
        v5 protocol."""
        from .distill import TEACHERS, DistillTrainer
        from .parallel import collect_dagger_data_parallel

        trainer = DistillTrainer(
            self.net, lr=self.cfg.distill_lr, device=self.train_cfg.device,
            chunk_decisions=self.cfg.distill_chunk_decisions,
        )
        data: list[tuple[np.ndarray, int]] = []
        beta = self.cfg.dagger_beta0
        for r in range(self.cfg.dagger_rounds):
            new_data, _stats = collect_dagger_data_parallel(
                self.net, self.cfg.reference,
                self.level, self.cfg.dagger_episodes,
                seed0=self._dagger_seed0(r), beta=beta,
                rng_seed=int(self.rng.integers(1 << 31)), workers=self.cfg.workers,
            )
            data.extend(new_data)
            for _ in range(self.cfg.dagger_epochs):
                trainer.train_batch(data)
            beta *= self.cfg.dagger_beta_decay

    def gate(self, level: int) -> GateResult:
        env = CheeseEnv(level=level)
        learner = gate_stats(
            self.net, env, self.cfg.gate_episodes, self.cfg.gate_seed0
        )
        baseline = baseline_stats(
            self.cfg.reference, env, self.cfg.gate_episodes, self.cfg.gate_seed0
        )
        return check_gate(
            learner, baseline,
            margin=self.cfg.gate_margin, use_ci=self.cfg.gate_use_ci,
            allow_tie=self.cfg.gate_allow_tie,
        )

    def run(self, agent: PolicyAgent, iterations_per_level: int = 40, verbose: bool = True):
        """Train + gate level by level. Yields a LevelReport per level;
        stops at max_level or when the gate blocks advancement twice in a
        row (the user's "increase only when it consistently beats")."""
        blocked = 0
        while self.level <= self.cfg.max_level:
            if verbose:
                print(f"=== level {self.level} ===")
            stats = self.train_level(agent, iterations_per_level)
            if verbose and stats:
                last = stats[-1]
                mp = f"{last.mean_pieces:.2f}" if last.mean_pieces is not None else "—"
                print(
                    f"trained {len(stats)} iters | win {last.win_rate:.0%}"
                    f" | pieces {mp} | entropy {last.entropy:.3f}"
                )
            gate = self.gate(self.level)
            report = LevelReport(level=self.level, passed=gate.passed, gate=gate)
            if gate.passed:
                if self.cfg.retention:
                    for lower in range(self.cfg.start_level, self.level):
                        rg = self.gate(lower)
                        report.retention[lower] = rg
                        if not rg.passed and verbose:
                            print(f"retention regression at level {lower}: {rg.reason}")
                self._checkpoint(agent, self.level)
                if self.level == self.cfg.max_level:
                    if verbose:
                        print(f"max level {self.level} passed — done")
                    yield report
                    return
                self.level = min(self.level + self.cfg.level_step, self.cfg.max_level)
                blocked = 0
                if verbose:
                    print(f"gate PASSED — advancing to {self.level}: {gate.reason}")
            else:
                blocked += 1
                if verbose:
                    print(f"gate blocked ({blocked}): {gate.reason}")
                if blocked >= 2:
                    if verbose:
                        print("twice blocked — stopping the ladder")
                    yield report
                    return
            yield report

    def _checkpoint(self, agent: PolicyAgent, level: int) -> None:
        path = Path(self.cfg.checkpoint_dir) / f"cheese_policy_L{level}.json"
        save_policy(self.net, path)
