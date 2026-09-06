"""DAS/ARR controller behaviour (engine-independent, deterministic ticks)."""

from tetris.engine.game import Action, Btn
from tetris.input import InputConfig, InputController


def make_controller(das=2, arr=2):
    cfg = InputConfig(das_ms=1000, arr_ms=1000)
    c = InputController(cfg)
    c.das_ticks = das
    c.arr_ticks = arr
    return c


def test_press_gives_immediate_move():
    c = make_controller()
    c.press(Btn.LEFT)
    actions, held = c.update()
    assert actions == [Action.LEFT]
    assert Btn.LEFT in held


def test_das_then_arr():
    c = make_controller(das=2, arr=2)
    c.press(Btn.RIGHT)
    assert c.update()[0] == [Action.RIGHT]  # press
    assert c.update()[0] == []  # charge 1
    assert c.update()[0] == [Action.RIGHT]  # DAS reached (tick 2)
    assert c.update()[0] == []  # ARR phase 1
    assert c.update()[0] == [Action.RIGHT]  # ARR repeat
    assert c.update()[0] == []
    assert c.update()[0] == [Action.RIGHT]


def test_arr_zero_moves_to_wall():
    c = make_controller(das=2, arr=0)
    c.press(Btn.LEFT)
    assert c.update()[0] == [Action.LEFT]
    assert c.update()[0] == []
    actions, _ = c.update()
    assert actions == [Action.LEFT] * 10  # instant: enough to reach the wall


def test_direction_switch_recharges_das():
    c = make_controller(das=3, arr=1)
    c.press(Btn.LEFT)
    c.update()
    c.update()
    c.press(Btn.RIGHT)  # opposite: active dir switches, DAS restarts
    actions, held = c.update()
    assert actions == [Action.RIGHT]
    assert held == frozenset({Btn.LEFT, Btn.RIGHT})
    assert c.update()[0] == []  # charge 1 (restarted)
    assert c.update()[0] == []  # charge 2
    assert c.update()[0] == [Action.RIGHT]  # charge 3 -> repeat


def test_release_resumes_other_direction():
    c = make_controller(das=3, arr=1)
    c.press(Btn.LEFT)
    c.update()
    c.press(Btn.RIGHT)
    c.update()
    c.release(Btn.RIGHT)  # active released -> LEFT resumes with fresh move
    actions, held = c.update()
    assert actions == [Action.LEFT]
    assert held == frozenset({Btn.LEFT})
    assert c.update()[0] == []
    assert c.update()[0] == []


def test_rotation_and_hard_drop_are_edge_actions():
    c = make_controller()
    c.press(Btn.ROT_CW)
    c.press(Btn.HARD)
    assert c.update()[0] == [Action.ROT_CW, Action.HARD_DROP]
    # holding the key must not re-fire (no OS auto-repeat in the engine)
    assert c.update()[0] == []


def test_soft_drop_is_a_held_state_not_an_action():
    c = make_controller()
    c.press(Btn.SOFT)
    actions, held = c.update()
    assert actions == []
    assert Btn.SOFT in held
    c.release(Btn.SOFT)
    _, held = c.update()
    assert Btn.SOFT not in held
