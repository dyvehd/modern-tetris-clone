"""Input event log: key presses and piece spawns, one line per event.

Purpose: debugging handling at high speed ("did I actually press that?").
The log records raw key events (down/up with the bound action name) and
which piece becomes active, so a mis-drop can be traced back to either an
input mistake or a missing/duplicated event, e.g.::

     12.345 T piece
     12.401 Left (key down)
     12.455 Left (key up)
     12.501 Harddrop (key down)
     12.518 L piece
     12.610 Harddrop (key up)

Efficiency: one pre-formatted ``os.write`` on an ``O_APPEND`` descriptor
per event — a single page-cache syscall (µs). The per-tick piece check is
an identity comparison and writes nothing unless the piece changed. There
is no userspace buffer, so the log is complete even if the process dies.

The file is created lazily on the first event: sessions that never start
a game leave nothing on disk.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from ..engine.game import ActivePiece, Btn

_BTN_NAMES: dict[Btn, str] = {
    Btn.LEFT: "Left",
    Btn.RIGHT: "Right",
    Btn.SOFT: "Softdrop",
    Btn.ROT_CW: "Rotate CW",
    Btn.ROT_CCW: "Rotate CCW",
    Btn.ROT_180: "Rotate 180",
    Btn.HARD: "Harddrop",
    Btn.HOLD: "Hold",
}


class InputLogger:
    """Append-only event log. pygame-free so it stays unit-testable."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fd = -1  # opened on first write
        self._done = False  # closed or permanently failed
        self._t0 = time.perf_counter_ns()  # timestamps: session-relative
        self._last_active: ActivePiece | None = None

    def line(self, text: str) -> None:
        if self._done:
            return
        if self._fd < 0 and not self._open():
            return
        stamp = (time.perf_counter_ns() - self._t0) / 1e9
        # O_APPEND makes each single write() an atomic append; it lands in
        # the OS page cache and never blocks on the disk.
        os.write(self._fd, f"{stamp:10.3f} {text}\n".encode("ascii", "replace"))

    # app hooks — the hot one (check_piece) allocates only on a new piece ---
    def key(self, btn: Btn, down: bool) -> None:
        name = _BTN_NAMES.get(btn)
        if name is not None:
            self.line(f"{name} ({'key down' if down else 'key up'})")

    def check_piece(self, active: ActivePiece | None) -> None:
        """Log ``X piece`` whenever a new active piece object appears.

        Identity, not equality: the engine replaces ``active`` on every
        spawn and hold swap, and mutates it in place while falling.
        """
        if active is not self._last_active:
            self._last_active = active
            if active is not None:
                self.line(f"{active.type.name} piece")

    def note(self, text: str) -> None:
        """Free-form state line (game start, pause, top out, ...)."""
        self.line(text)

    def close(self) -> None:
        self._done = True
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def _open(self) -> bool:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._fd = os.open(self._path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        except OSError as exc:
            print(f"warning: input log disabled ({exc})", file=sys.stderr)
            self._done = True
            return False
        return True
