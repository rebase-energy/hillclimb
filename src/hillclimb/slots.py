"""Machine-wide agent-concurrency slots.

Caps concurrent agent calls across every search process sharing a runs_dir
(suites spawn one process per problem; each may run several workers). Slots
are `flock`ed files — the lock dies with the process, so crashes free their
slot with no reclamation logic. Caveat: advisory flock is unreliable on NFS
mounts; keep runs_dir on a local filesystem.
"""

from __future__ import annotations

import fcntl
import os
import threading
import time
from pathlib import Path
from typing import Callable

RETRY_INTERVAL_S = 2.0


class SlotHandle:
    def __init__(self, fd: int | None):
        self._fd = fd

    def release(self) -> None:
        if self._fd is not None:
            os.close(self._fd)  # closing drops the flock
            self._fd = None

    def __enter__(self) -> "SlotHandle":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


class MachineSlots:
    """Acquire one of `limit` machine-wide slots; limit=0 disables the cap.
    `root` is the slot-file directory — pass the machine cache's agent-slots
    dir so the cap spans every search on the machine."""

    def __init__(self, root: Path, limit: int):
        self.limit = limit
        self.root = root
        if limit > 0:
            self.root.mkdir(parents=True, exist_ok=True)

    def try_acquire(self) -> SlotHandle | None:
        """One non-blocking pass over the slot files; None if all busy."""
        if self.limit <= 0:
            return SlotHandle(None)
        for index in range(self.limit):
            fd = os.open(self.root / f"slot-{index:02d}", os.O_CREAT | os.O_RDWR, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                os.close(fd)
                continue
            os.ftruncate(fd, 0)
            os.write(fd, f"pid={os.getpid()} t={time.time():.0f}\n".encode())
            return SlotHandle(fd)
        return None

    def acquire(
        self,
        abort: threading.Event | None = None,
        should_stop: Callable[[], bool] | None = None,
        on_wait: Callable[[], None] | None = None,
    ) -> SlotHandle | None:
        """Block until a slot frees (2s retry). Returns None when aborted or
        should_stop() turns true while waiting."""
        waited = False
        while True:
            handle = self.try_acquire()
            if handle is not None:
                return handle
            if abort is not None and abort.is_set():
                return None
            if should_stop is not None and should_stop():
                return None
            if not waited and on_wait is not None:
                on_wait()
                waited = True
            time.sleep(RETRY_INTERVAL_S)
