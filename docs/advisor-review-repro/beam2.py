"""Advisor scratch: a corrected beam teacher for A/B measurement.

Differences from tetris.ai.search.BeamAgent, each independently switchable:
  fix_hold   — engine-exact hold/queue transitions (empty hold consumes
               queue[0]; can_hold resets to True after every lock).
  expand_all — non-clearing placements are expanded too. A placement is a
               leaf only when the engine would actually refill (a no-clear
               lock with cheese_on_board below the capped target), because
               only then is the next board unknowable.
  horizon    — plans are compared at a common horizon: wins ordered by
               fewest pieces first; otherwise deepest leaves by
               eval + w.lines * cheese dug along the plan.
"""
from __future__ import annotations

import sys
sys.path.insert(0, "src")

from tetris.ai.agents import BaseAgent
from tetris.ai.cheese import Decision, Obs, candidate_moves
from tetris.ai.eval import EvalWeights, eval_board
from tetris.ai.movegen import enumerate_placements
from tetris.ai.search import lock_and_count
from tetris.engine.constants import FIELD_H

STACK = 9  # CheeseEnv.stack default (Jstris)


class Node:
    __slots__ = ("rows", "hold", "can_hold", "queue", "dug", "on_board",
                 "score", "first", "pieces", "won", "leaf")

    def __init__(self, rows, hold, can_hold, queue, dug, on_board, score, first, pieces, won, leaf):
        self.rows, self.hold, self.can_hold, self.queue = rows, hold, can_hold, queue
        self.dug, self.on_board, self.score, self.first = dug, on_board, score, first
        self.pieces, self.won, self.leaf = pieces, won, leaf


class Beam2(BaseAgent):
    def __init__(self, width=20, depth=4, weights=None, fix_hold=True, expand_all=True,
                 horizon=True, name=None):
        self.width, self.depth = width, depth
        self.w = weights or EvalWeights()
        self.fix_hold, self.expand_all, self.horizon = fix_hold, expand_all, horizon
        self.name = name or f"beam2_{width}x{depth}{'H' if fix_hold else ''}{'E' if expand_all else ''}{'Z' if horizon else ''}"

    def _children(self, obs: Obs, node: Node):
        """Yield (placement, hold_flag, piece_placed, hold_after, can_hold_after, queue_after)."""
        w = self.w
        if node.queue:
            piece = node.queue[0]
            yield from ((p, False, node.hold, True, node.queue[1:]) for p in
                        enumerate_placements(node.rows, piece, allow_180=obs.allow_180))
        if node.can_hold:
            if node.hold is not None and node.queue:
                # swap: place hold, stash queue[0]
                yield from ((p, True, node.queue[0], self.fix_hold, node.queue[1:]) for p in
                            enumerate_placements(node.rows, node.hold, allow_180=obs.allow_180))
            elif node.hold is None and len(node.queue) >= 2:
                # empty hold: stash queue[0], place queue[1]; next is queue[2]
                yield from ((p, True, node.queue[0], self.fix_hold, node.queue[2:]) for p in
                            enumerate_placements(node.rows, node.queue[1], allow_180=obs.allow_180))

    def _expand(self, obs: Obs, node: Node, first=None):
        w = self.w
        out = []
        for p, hold, hold_after, can_hold_after, queue_after in self._children(obs, node):
            rows2, lines, dug = lock_and_count(node.rows, p, node.on_board)
            total_dug = node.dug + dug
            on_board = node.on_board - dug
            won = total_dug >= obs.goal
            pieces = node.pieces + 1
            # would the engine refill after this lock? (only after a no-clear lock)
            target = min(STACK, obs.goal - total_dug)
            refills = (lines == 0) and (target - on_board > 0)
            leaf = won or refills or (not self.expand_all and lines == 0)
            if self.horizon:
                score = eval_board(rows2, w) + w.lines * total_dug
                if won:
                    score = w.win - 50.0 * pieces
            else:
                score = eval_board(rows2, w) + w.lines * lines
                if won:
                    score += w.win - 50.0 * pieces
            out.append(Node(rows2, hold_after, can_hold_after, queue_after, total_dug, on_board,
                            score, first if first is not None else Decision(p, hold=hold),
                            pieces, won, leaf))
        return out

    def decide(self, obs: Obs) -> Decision:
        # root node: engine state as-is
        root = Node(list(obs.rows), obs.hold, obs.can_hold, (obs.active,) + tuple(obs.queue),
                    obs.cheese_dug, obs.cheese_on_board, 0.0, None, 0, False, False)
        # root children must be exactly candidate_moves(obs) (same action space)
        children = []
        if not self.fix_hold:
            # replicate the original's (buggy) root semantics for A/B fairness
            for p, hold in candidate_moves(obs):
                rows2, lines, dug = lock_and_count(list(obs.rows), p, obs.cheese_on_board)
                total_dug = obs.cheese_dug + dug
                won = total_dug >= obs.goal
                on_board = obs.cheese_on_board - dug
                target = min(STACK, obs.goal - total_dug)
                refills = (lines == 0) and (target - on_board > 0)
                leaf = won or refills or (not self.expand_all and lines == 0)
                score = eval_board(rows2, self.w) + self.w.lines * (total_dug if self.horizon else lines)
                if won:
                    score = self.w.win - 50.0 if self.horizon else score + self.w.win - 50.0
                children.append(Node(rows2, obs.active if hold else obs.hold, not hold, tuple(obs.queue),
                                     total_dug, on_board, score, Decision(p, hold=hold), 1, won, leaf))
        else:
            children = self._expand(obs, root)
        if not children:
            raise RuntimeError("no candidates")
        wins = [n for n in children if n.won]
        leaves = [n for n in children if n.leaf and not n.won]
        frontier = sorted((n for n in children if not n.leaf), key=lambda n: -n.score)[: self.width]
        best_leaf = max(children, key=lambda n: n.score)
        for _ in range(self.depth - 1):
            if not frontier:
                break
            nxt = []
            for node in frontier:
                nxt.extend(self._expand(obs, node, node.first))
            if not nxt:
                break
            for n in nxt:
                if n.won:
                    wins.append(n)
                elif n.leaf:
                    leaves.append(n)
            frontier = sorted((n for n in nxt if not n.leaf), key=lambda n: -n.score)[: self.width]
            if self.horizon:
                # compare at the deepest horizon reached: frontier ∪ leaves that stopped early
                cands = frontier + leaves
                if cands:
                    best_leaf = max(cands, key=lambda n: n.score)
            else:
                cands = frontier + leaves
                best_leaf = max([best_leaf] + cands, key=lambda n: n.score)
        if wins:
            return min(wins, key=lambda n: (n.pieces, -n.score)).first
        return best_leaf.first
