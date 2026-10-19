"""The ear's port: what it publishes, frozen and versioned (``PORT``).

A consumer (the arbiter first) reads THIS, never the ear's internals. The
transport is the record, ``events.jsonl``; every line the ear writes since v1.1
carries ``"port": 1``. Lines without it (v0, or v1.0) are skipped, never guessed.

    Session     a session starts: which weights, how many rows already known
    Decision    one window: who (or unknown / noise), how sure, how loud
    Text        one finished speech run, transcribed, with the rows it covers
    Correction  one case said by the operator, as the ear applied it

Changing a field's meaning is a new PORT number, never an edit of this one.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, fields
from pathlib import Path

from π_ear.io import read_jsonl

PORT = 1


@dataclass(frozen=True, slots=True)
class Session:
    session: str
    utc: str
    weights: str
    rows: int
    first_chunk: str | None


@dataclass(frozen=True, slots=True)
class Decision:
    row: int
    session: str
    t0: float  # stream seconds since the session started
    t1: float
    decided_at: float
    kind: str  # known | new | unknown | noise
    speaker: str | None
    label: str | None
    p: float  # posterior of the decision
    margin: float  # nats between the decision and the runner up: the legible doubt
    evidence: int
    learned: bool  # False for a glimpse, shorter than the encoder window
    run: float
    rms_dbfs: float
    rel_db: float | None  # level above this session's non-speech floor


@dataclass(frozen=True, slots=True)
class Text:
    session: str
    t0: float
    t1: float
    rows: list[int]
    text: str


@dataclass(frozen=True, slots=True)
class Correction:
    session: str
    what: str  # same | different | not_voice | name | undo
    rows: list[int]
    label: str | None
    line: int  # its line number in corrections.jsonl, the handle `undo` uses
    refused: str | None


KINDS = {"session": Session, "text": Text, "correction": Correction}


def parse(line: dict):
    """One record line to its typed event, or None if it is not on this port."""
    if line.get("port") != PORT:
        return None
    cls = KINDS.get(line.get("kind"), Decision if "row" in line else None)
    if cls is None:
        return None
    return cls(**{f.name: line.get(f.name) for f in fields(cls)})


def read(path: Path, session: str | None = None) -> list:
    events = [e for e in map(parse, read_jsonl(path)[0]) if e is not None]
    return [e for e in events if session is None or e.session == session]


def latest_session(path: Path) -> str | None:
    starts = [e.session for e in read(path) if isinstance(e, Session)]
    return starts[-1] if starts else None


def follow(path: Path, poll_s: float = 0.2) -> Iterator:
    """Every new event, as it is written: what a live consumer loops on."""
    offset = 0
    while True:
        lines, offset = read_jsonl(path, offset)
        for line in lines:
            event = parse(line)
            if event is not None:
                yield event
        if not lines:
            time.sleep(poll_s)
