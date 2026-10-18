"""The state on disk: one atomic rewrite, one append-only log, two sentinels.

Three rules encoded here, and they are the whole reason this module exists:

1. the state file is always complete: it is written to a temporary file and then
   renamed, so a reader never sees half a JSON (a half-written state read by the
   agentic pole would be worse than a stale one);
2. the event log is append-only: it is the corpus, and the corpus is never
   rewritten;
3. control is a file, not a signal: a pause that is visible *in the state* is a
   pause the downstream poles can reason about.
"""

from __future__ import annotations

import difflib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path


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
    """Write JSON in a way that can never be observed half-written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def append_event(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def snapshot(state_file: Path, snapshots_dir: Path, tag: str | None = None) -> Path:
    """Freeze a copy of the state. A state you cannot compare to another proves nothing."""
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    target = snapshots_dir / f"{stamp()}{'-' + tag if tag else ''}.json"
    shutil.copy2(state_file, target)
    return target


def diff_last_two(snapshots_dir: Path) -> str:
    files = sorted(snapshots_dir.glob("*.json"))
    if len(files) < 2:
        return "need at least two snapshots: run `π-ear snapshot` twice"
    older, newer = files[-2], files[-1]
    before = older.read_text(encoding="utf-8").splitlines()
    after = newer.read_text(encoding="utf-8").splitlines()
    body = "\n".join(
        difflib.unified_diff(before, after, fromfile=older.name, tofile=newer.name, lineterm="")
    )
    return body or f"{older.name} and {newer.name} are identical"


def archive(root_archive: Path, state_file: Path, events_file: Path) -> None:
    """Move the current state and events aside instead of deleting them."""
    root_archive.mkdir(parents=True, exist_ok=True)
    for path in (state_file, events_file):
        if path.exists():
            shutil.move(str(path), str(root_archive / f"{stamp()}-{path.name}"))
