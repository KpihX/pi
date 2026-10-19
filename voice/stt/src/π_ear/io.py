"""Everything that touches the disk, and nothing that decides.

Kept from v0 because none of it is a decision and all of it was measured:

* the **tailer**: a directory of one-second raw chunks is the whole transport.
  A chunk the recorder is still writing is not read (measured 2026-10-19:
  reading a growing file made 4179 events out of 379 seconds of audio);
* the **state** is rewritten atomically, so a reader never sees half a JSON;
* the **record** is append-only;
* the **corpus** is a bounded, durable copy of what was heard, outside RAM;
* **control is a file**: ``PAUSE`` and ``RESET``, visible in the state.

New in v1: the **embedding store**, one append-only float32 row per observation.
Rows are global and never reused, so a correction written today can still point
at a window heard last week, across restarts.
"""

from __future__ import annotations

import difflib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_jsonl(path: Path, offset: int = 0) -> tuple[list[dict], int]:
    """Complete lines after byte ``offset``, and the offset after the last one."""
    try:
        with open(path, "rb") as handle:
            handle.seek(offset)
            raw = handle.read()
    except FileNotFoundError:
        return [], offset
    end = raw.rfind(b"\n") + 1
    lines = [json.loads(line) for line in raw[:end].splitlines() if line.strip()]
    return lines, offset + end


def load_raw(path: Path) -> np.ndarray:
    """Little-endian int16 PCM to float32 in [-1, 1]."""
    raw = path.read_bytes()
    raw = raw[: len(raw) - len(raw) % 2]
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


@dataclass(slots=True)
class Chunk:
    name: str
    path: Path
    samples: np.ndarray


class Tailer:
    """New complete chunks of a directory, in order, each exactly once."""

    def __init__(self, audio_dir: Path, complete_bytes: int, from_start: bool, cursor=None):
        self.audio_dir = audio_dir
        self.complete_bytes = complete_bytes
        self.cursor: str | None = cursor
        self.from_start = from_start or cursor is not None
        self.seen = 0

    def poll(self) -> list[Chunk]:
        files = sorted(self.audio_dir.glob("*.raw"))
        if not files:
            self.from_start = True  # nothing to skip: whatever comes next is "now"
            return []
        if self.cursor is None and not self.from_start:
            self.cursor = files[-1].name  # a fresh live start begins now
            return []
        out = []
        for path in files:
            if self.cursor is not None and path.name <= self.cursor:
                continue
            last = path == files[-1]
            if last and self.complete_bytes and path.stat().st_size < self.complete_bytes:
                break  # still being written by the recorder
            out.append(Chunk(path.name, path, load_raw(path)))
            self.cursor = path.name
            self.seen += 1
        return out

    def backlog(self) -> int:
        return sum(
            1 for p in self.audio_dir.glob("*.raw") if self.cursor is None or p.name > self.cursor
        )

    def prune(self) -> None:
        """Drop consumed chunks: the audio directory is RAM on the reference machine."""
        for path in sorted(self.audio_dir.glob("*.raw")):
            if self.cursor is None or path.name > self.cursor:
                break
            path.unlink(missing_ok=True)


class Corpus:
    """A bounded, durable copy of every ingested chunk, so a session can be replayed."""

    def __init__(self, directory: Path | None, max_chunks: int):
        self.directory = directory
        self.max_chunks = max_chunks
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)

    def keep(self, chunk: Chunk) -> None:
        if self.directory is None:
            return
        target = self.directory / chunk.name
        if not target.exists():
            shutil.copyfile(chunk.path, target)
        files = sorted(self.directory.glob("*.raw"))
        for path in files[: max(0, len(files) - self.max_chunks)]:
            path.unlink(missing_ok=True)

    def stats(self) -> dict:
        if self.directory is None:
            return {"enabled": False}
        files = list(self.directory.glob("*.raw"))
        return {
            "enabled": True,
            "dir": str(self.directory),
            "chunks": len(files),
            "bytes": sum(p.stat().st_size for p in files),
        }


class EmbeddingStore:
    """One float32 row per observation, append-only, reloaded at launch."""

    def __init__(self, path: Path, dim: int):
        self.path = path
        self.dim = dim
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            flat = np.fromfile(path, dtype=np.float32)
            self.rows = list(
                flat[: len(flat) - len(flat) % dim].reshape(-1, dim).astype(np.float64)
            )
        else:
            self.rows = []

    def append(self, vec: np.ndarray) -> int:
        with open(self.path, "ab") as handle:
            handle.write(np.asarray(vec, dtype=np.float32).tobytes())
        self.rows.append(np.asarray(vec, dtype=np.float32).astype(np.float64))
        return len(self.rows) - 1

    def __call__(self, row: int) -> np.ndarray:
        return self.rows[row]

    def __len__(self) -> int:
        return len(self.rows)


def snapshot(state_file: Path, snapshots_dir: Path, tag: str | None = None) -> Path:
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    target = snapshots_dir / f"{stamp()}{'-' + tag if tag else ''}.json"
    shutil.copy2(state_file, target)
    return target


def diff_last_two(snapshots_dir: Path) -> str:
    files = sorted(snapshots_dir.glob("*.json"))
    if len(files) < 2:
        return "need at least two snapshots: run `π-ear snapshot` twice"
    a, b = files[-2], files[-1]
    lines = difflib.unified_diff(
        a.read_text().splitlines(), b.read_text().splitlines(), a.name, b.name, lineterm=""
    )
    return "\n".join(lines) or f"{a.name} and {b.name} are identical"


def archive(target: Path, *paths: Path) -> None:
    """Move aside instead of deleting: a reset is reversible by hand."""
    target.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if path.exists():
            shutil.move(str(path), str(target / f"{stamp()}-{path.name}"))
