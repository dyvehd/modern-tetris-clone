"""Minimal AI-example: a 'flat stacking' bot over the headless environment.

Run:  .venv/bin/python examples/flat_bot.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tetris.engine.env import Action, GameConfig, TetrisEnv  # noqa: E402


def column_heights(rows: tuple[int, ...]) -> list[int]:
    heights = [0] * 10
    for y, row in enumerate(rows[-20:]):
        for x in range(10):
            if row >> x & 1 and heights[x] == 0:
                heights[x] = 20 - y
    return heights


def evaluate(rows: tuple[int, ...], lines: int) -> float:
    """Classic simple heuristics: keep the stack low and flat."""
    heights = column_heights(rows)
    return (
        -sum(heights)                        # aggregate height
        - 8 * (max(heights) - min(heights))  # bumpiness
        + 400 * lines                        # cleared lines are good
    )


def steer(env: TetrisEnv, target_x: int, target_rot: int, max_steps: int = 25) -> None:
    """Rotate/shift the active piece onto the target (one input per tick)."""
    for _ in range(max_steps):
        piece = env.observe().piece
        if piece is None:
            return
        _, p_rot, p_x, _y = piece
        d_rot = (target_rot - p_rot) % 4
        if d_rot != 0:
            env.step(Action.ROT_CW if d_rot == 1 else Action.ROT_CCW)
        elif p_x < target_x:
            env.step(Action.RIGHT)
        elif p_x > target_x:
            env.step(Action.LEFT)
        else:
            return


def main() -> None:
    env = TetrisEnv(GameConfig(gravity_g=0.1, lock_delay_ms=1000), seed=1234)
    env.reset()
    env.step(Action.NOOP)  # first tick performs the initial spawn
    obs = env.observe()
    while not obs.over:
        best = None
        for x, rot, _y in env.reachable_placements():
            sim = env.clone()
            piece = sim.game.active
            piece.rot = rot
            piece.x = x
            sim.game.last_action = None
            sim.step(Action.HARD_DROP)
            o = sim.observe()
            score = evaluate(o.rows, o.lines)
            if best is None or score > best[0]:
                best = (score, x, rot)
        _, x, rot = best
        steer(env, x, rot)
        obs, _ = env.step(Action.HARD_DROP)
    print(f"game over: pieces={obs.pieces_placed} lines={obs.lines} score={obs.score}")


if __name__ == "__main__":
    main()
