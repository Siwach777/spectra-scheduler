"""Lightweight, foreground terminal progress without an extra dependency."""

import sys
from time import perf_counter


class LiveProgress:
    def __init__(self, stream=None):
        self.stream = stream or sys.stdout
        self.terminal = self.stream.isatty()
        self.phase = None
        self.started = self.last = perf_counter()
        self.width = 0

    def message(self, text):
        if self.width:
            print(file=self.stream, flush=True)
            self.width = 0
        print(text, file=self.stream, flush=True)

    def update(self, phase, done, total, detail=""):
        now = perf_counter()
        changed = phase != self.phase
        if changed:
            if self.width:
                print(file=self.stream, flush=True)
                self.width = 0
            self.phase, self.started = phase, now
        if not changed and done != total and now - self.last < (0.5 if self.terminal else 2):
            return
        self.last = now
        elapsed = now - self.started
        fraction = done / total if total else 0
        eta = elapsed * (total - done) / done if done else None
        bar = "=" * int(20 * fraction) + "." * (20 - int(20 * fraction))
        text = f"{phase} [{bar}] {done}/{total} ({fraction:.0%}) | elapsed {elapsed:.1f}s"
        if eta is not None:
            text += f" | ETA {eta:.1f}s"
        if detail:
            text += f" | {detail}"
        if self.terminal:
            print("\r" + text.ljust(self.width), end="", file=self.stream, flush=True)
            self.width = len(text)
            if done == total:
                print(file=self.stream, flush=True)
                self.width = 0
        else:
            print(text, file=self.stream, flush=True)

    def callback(self, phase):
        return lambda done, total: self.update(phase, done, total)
