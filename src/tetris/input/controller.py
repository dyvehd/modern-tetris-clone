"""Keyboard handling: DAS/ARR auto-shift, key -> virtual button mapping.

Pure logic (no pygame): the app feeds press/release edges in, and calls
``update()`` once per engine tick to receive the action list plus the set of
currently held buttons (used by the engine for soft drop; IRS/IHS only if
the TGM-style flags are enabled in GameConfig).

Timing conventions (TETR.IO/Jstris style):
- Press: one immediate move, DAS charge starts.
- After ``das`` ticks, a repeat fires, then every ``arr`` ticks.
  (So the effective delay to the *second* move is das; add ARR on top if you
  want TETR.IO's sub-frame convention.)
- arr = 0 means instant: on charge, move all the way to the wall.
- Pressing the opposite direction switches the active direction and restarts
  DAS; releasing the active direction activates the other held one with a
  fresh immediate move + DAS.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from ..engine.game import Action, Btn
from ..engine.constants import ms_to_ticks


@dataclass
class InputConfig:
    das_ms: float = 167.0
    arr_ms: float = 33.0
    sdf: float = 5.0  # informational; the engine applies soft-drop gravity


class InputController:
    def __init__(self, cfg: InputConfig | None = None) -> None:
        cfg = cfg or InputConfig()
        self.das_ticks = max(1, ms_to_ticks(cfg.das_ms))
        self.arr_ticks = max(0, ms_to_ticks(cfg.arr_ms))
        self._events: deque[tuple[bool, Btn]] = deque()  # (pressed, button)
        self._held: set[Btn] = set()
        self._dir_stack: list[Btn] = []  # held directions, oldest first
        self._das_charge = 0
        self._arr_phase = 0

    def release_all(self) -> None:
        """Forget all pressed keys (used when leaving the game context)."""
        self._events.clear()
        self._held.clear()
        self._dir_stack.clear()
        self._das_charge = 0
        self._arr_phase = 0

    def apply_config(self, cfg: InputConfig) -> None:
        """Re-read DAS/ARR (live handling changes from the settings screen)."""
        self.das_ticks = max(1, ms_to_ticks(cfg.das_ms))
        self.arr_ticks = max(0, ms_to_ticks(cfg.arr_ms))

    # called by the app between ticks --------------------------------------
    def press(self, btn: Btn) -> None:
        self._events.append((True, btn))

    def release(self, btn: Btn) -> None:
        self._events.append((False, btn))

    # called once per engine tick -------------------------------------------
    def update(self) -> tuple[list[Action], frozenset[Btn]]:
        actions: list[Action] = []
        just_activated = False
        while self._events:
            pressed, btn = self._events.popleft()
            if btn in (Btn.LEFT, Btn.RIGHT):
                if pressed:
                    if btn not in self._dir_stack:
                        self._dir_stack.append(btn)
                        self._das_charge = 0
                        self._arr_phase = 0
                        just_activated = True
                        actions.append(Action.LEFT if btn is Btn.LEFT else Action.RIGHT)
                else:
                    was_active = self._active_dir() is btn
                    if btn in self._dir_stack:
                        self._dir_stack.remove(btn)
                    if was_active:
                        # Switch back to the other held direction, if any,
                        # with a fresh immediate move and DAS re-charge.
                        self._das_charge = 0
                        self._arr_phase = 0
                        new_active = self._active_dir()
                        if new_active is not None:
                            just_activated = True
                            actions.append(
                                Action.LEFT if new_active is Btn.LEFT else Action.RIGHT
                            )
            else:
                if pressed:
                    self._held.add(btn)
                    action = self._btn_action(btn)
                    if action is not None:
                        actions.append(action)
                else:
                    self._held.discard(btn)

        active = self._active_dir()
        if active is not None and not just_activated:
            # The tick an activation (press or resume) happens on already
            # produced its immediate move; DAS starts charging afterwards.
            self._das_charge += 1
            if self.arr_ticks == 0:
                if self._das_charge >= self.das_ticks:
                    # Instant ARR: one tick moves the piece to the wall.
                    actions.extend([self._dir_action(active)] * 10)
            elif self._das_charge >= self.das_ticks and (
                (self._das_charge - self.das_ticks) % self.arr_ticks == 0
            ):
                # First repeat fires once DAS is charged, then every ARR.
                actions.append(self._dir_action(active))

        held = frozenset(self._held) | set(self._dir_stack)
        return actions, held

    # helpers ---------------------------------------------------------------
    def _active_dir(self) -> Btn | None:
        return self._dir_stack[-1] if self._dir_stack else None

    @staticmethod
    def _dir_action(btn: Btn) -> Action:
        return Action.LEFT if btn is Btn.LEFT else Action.RIGHT

    @staticmethod
    def _btn_action(btn: Btn) -> Action | None:
        return {
            Btn.ROT_CW: Action.ROT_CW,
            Btn.ROT_CCW: Action.ROT_CCW,
            Btn.ROT_180: Action.ROT_180,
            Btn.HARD: Action.HARD_DROP,
            Btn.HOLD: Action.HOLD,
        }.get(btn)  # SOFT has no edge action; it rides on the held set
