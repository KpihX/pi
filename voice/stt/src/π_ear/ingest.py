"""Following the recorder's directory, one second at a time.

The interface between the two servers is a directory of raw PCM chunks written by
``arecord --max-file-time``. That choice buys four things a pipe cannot:

* the recorder is never stopped, so nothing is ever lost;
* the ear can be paused, restarted or rewritten without touching the recorder;
* a pause is a backlog, and a backlog is caught up on resume;
* the same chunks can be replayed later with different parameters.

Names come from ``%Y%m%d-%H%M%S``, so lexicographic order is chronological order:
no mtime sorting, no ambiguity.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from π_ear.physics import load_raw


@dataclass(slots=True)
class Chunk:
    name: str
    path: Path
    samples: np.ndarray
    is_new: bool
    gap_s: float


class ChunkTailer:
    """Reads what is new in the audio directory, never twice, never out of order."""

    def __init__(
        self,
        audio_dir: Path,
        chunk_s: int,
        gap_factor: float,
        from_start: bool = False,
        expected_bytes: int = 0,
    ):
        self.audio_dir = audio_dir
        self.chunk_s = chunk_s
        self.gap_factor = gap_factor
        self.from_start = from_start
        # The exact size of a complete chunk (chunk_s * rate * channels * 2 bytes
        # for s16le). 0 disables the completeness gate, which is what a replay of
        # a frozen directory wants: there, nothing is being written any more.
        self.expected_bytes = int(expected_bytes)
        self.cursor_name: str | None = None
        self.cursor_bytes = 0
        self.seen_wall: float | None = None
        self.chunks_seen = 0
        self.gaps: list[dict] = []

    # -- public ---------------------------------------------------------------
    def poll(self) -> list[Chunk]:
        out: list[Chunk] = []
        files = sorted(self.audio_dir.glob("*.raw"))
        if not files:
            return out

        newest_path = files[-1]
        newest = newest_path.name

        if self.cursor_name is None and not self.from_start:
            # fresh start: skip what is already on disk, start from now. The
            # backlog belongs to `replay` and to resume-after-pause, not here.
            self.cursor_name = newest
            self.cursor_bytes = newest_path.stat().st_size

        for path in files:
            name = path.name
            if self.cursor_name is not None and name < self.cursor_name:
                continue
            size = path.stat().st_size
            # A chunk the writer is still filling is not a chunk yet. Reading it
            # as it grew was the measured 10x: every poll picked up the delta, ran
            # a whole analysis on it, and counted one more observation of the SAME
            # second of voice. Measured on this machine, 2026-10-19: 4179 events
            # for 379 seconds of audio, the evidence bar inflated tenfold, one core
            # at 99.9%, 330 MB of log per day. A file that is no longer the newest
            # has been closed by the recorder: whatever it holds is final, even
            # when the recording stopped in the middle of a chunk.
            if self.expected_bytes and size < self.expected_bytes and name == newest:
                continue
            offset = self.cursor_bytes if name == self.cursor_name else 0
            if size <= offset:
                continue

            mtime = path.stat().st_mtime
            gap = 0.0
            if self.seen_wall is not None:
                gap = max(0.0, mtime - self.seen_wall - self.chunk_s)
            if gap > self.gap_factor * self.chunk_s:
                self.gaps.append({"after": self.cursor_name, "gap_s": round(gap, 2)})
            self.seen_wall = mtime

            samples = load_raw(path, offset)
            is_new = self.chunks_seen == 0
            self.chunks_seen += 1
            self.cursor_name = name
            self.cursor_bytes = size
            if samples.size:
                out.append(Chunk(name, path, samples, is_new, round(gap, 2)))
        return out

    def backlog(self) -> int:
        """How many chunks are waiting: the honest size of a paused backlog."""
        files = sorted(self.audio_dir.glob("*.raw"))
        if self.cursor_name is None:
            return len(files)
        return sum(1 for path in files if path.name > self.cursor_name)

    @staticmethod
    def prune(audio_dir: Path, keep_name: str | None) -> int:
        """Delete consumed chunks. /tmp is RAM (tmpfs) on this machine.

        Only chunks strictly older than the file currently being read are
        removed, so a chunk is never deleted before it has been ingested.
        """
        removed = 0
        for path in sorted(audio_dir.glob("*.raw")):
            if keep_name is not None and path.name >= keep_name:
                break
            try:
                path.unlink()
                removed += 1
            except FileNotFoundError:
                pass
        return removed


def wait(seconds: float) -> None:
    time.sleep(seconds)
