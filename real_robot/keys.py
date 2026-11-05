"""Single keypresses from the terminal, read in a background thread.

Puts the terminal in cbreak mode (keys arrive without Enter, no echo) and
restores it on close(). Needs a real terminal: run `make record` in your own
shell, not through a pipe.
"""
from __future__ import annotations

import os
import queue
import select
import sys
import threading


class KeyReader:
    def __init__(self, stream=None):
        import termios
        import tty

        self.stream = stream or sys.stdin
        if not self.stream.isatty():
            raise RuntimeError("keyboard controls need a terminal (stdin is not a TTY); "
                               "run this in your own shell")
        self.fd = self.stream.fileno()
        self.termios = termios
        self.saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        self.keys: queue.Queue[str] = queue.Queue()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while not self.stop.is_set():
            ready, _, _ = select.select([self.fd], [], [], 0.05)
            if ready:
                ch = os.read(self.fd, 1).decode(errors="ignore")
                if ch:
                    self.keys.put(ch)

    def get(self) -> str | None:
        """The next key pressed, or None."""
        try:
            return self.keys.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=0.5)
        self.termios.tcsetattr(self.fd, self.termios.TCSADRAIN, self.saved)
