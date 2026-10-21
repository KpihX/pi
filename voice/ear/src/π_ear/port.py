"""The ear's port: what it publishes, frozen and versioned (``PORT``).

Two directions, both files, both append-only:

* **out**, the record ``events.jsonl``: what the ear heard and decided. A
  consumer (the arbiter first) reads THIS, never the ear's internals. Every line
  carries ``"port": 3``; lines of an older port are skipped, never guessed;
* **in**, the inbox ``corrections.jsonl``: what anyone tells the ear, a person
  at a terminal, the arbiter, or a voice through the mouth and the ear itself.
  ``Inbox`` writes it; the running ear obeys at its next tick.

    Session     a session starts: which weights, which recognisers
    Decision    one window: who (voice, condition), person or device, how sure
    Turns       one run, who spoke when, to the half window
    Partial     the text of a piece still being said (provisional)
    Text        a piece's reading, revised as evidence lands: words, trust, doubt

Port 3 (2026-10-20, after the first live test): the unit of text is a *piece* of a
speech run (cut at Silero's dips, shorter than the slow recogniser's window), named
``piece``; ``run`` is the run it is cut from. In port 2 ``run`` named the whole run.
    Question    what the ear wants to know: a name, or which reading was said
    Correction  one case told to the ear, as it applied it (or why it refused)

Changing a field's meaning is a new PORT number, never an edit of this one.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, fields
from pathlib import Path

from π_ear.io import read_jsonl

PORT = 3


@dataclass(frozen=True, slots=True)
class Session:
    session: str
    utc: str
    weights: str
    rows: int
    first_chunk: str | None
    fast: str | None
    slow: str | None


@dataclass(frozen=True, slots=True)
class Decision:
    row: int
    session: str
    t0: float  # stream seconds since the session started
    t1: float
    decided_at: float
    kind: str  # known | new | unknown | noise
    speaker: str | None  # the voice (a person, or a device)
    condition: str | None  # the condition of that voice it was heard in
    label: str | None
    source: str | None  # person | device: told by a correction, else decided
    source_p: float | None
    p: float  # posterior of the decision
    margin: float  # nats between the decision and the runner up: the legible doubt
    trust: float | None  # P(right | margin), calibrated on corrections (None before)
    evidence: int
    learned: bool  # False for a glimpse or a shadow
    shadow: bool  # shifted by half a window: places the turns, never learned from
    run: float
    rms_dbfs: float
    rel_db: float | None  # level above this session's non-speech floor


@dataclass(frozen=True, slots=True)
class Turns:
    session: str
    run: str
    t0: float
    t1: float
    segments: list  # [t0, t1, voice | "unknown" | "noise"]


@dataclass(frozen=True, slots=True)
class Partial:
    session: str
    piece: str  # the piece being said (it becomes a Text of the same id)
    run: str
    t0: float | None
    text: str
    retracted: bool | None


@dataclass(frozen=True, slots=True)
class Text:
    session: str
    piece: str  # the unit of text: its run, "#", its index in the run
    run: str  # the speech run it is cut from
    t0: float
    t1: float
    rows: list[int]
    speaker: str | None  # the voice holding most of the piece's time
    speakers: dict  # every voice heard in it, with its share of the time
    revision: int  # 0 from the fast reading, then each revision (slow, correction)
    text: str
    p: float  # the reader's probability for this reading: its calibrated trust
    margin: float
    doubt: str  # clear | unseen | taught | partial | ambiguous | not_understood
    words: list  # {"w", "trust", "agreed", "new"}
    readings: list  # every reading considered, with its probability
    language: str | None  # the language it was written in
    told: str | None  # the language the slow recogniser was given (None: it detected one)
    speech: float | None  # Silero's mean over the piece
    separated: bool  # read again from the voice separated out of the music


@dataclass(frozen=True, slots=True)
class Question:
    session: str
    id: str
    about: str  # name | text
    piece: str | None  # the piece it is about (text questions)
    rows: list[int]
    speaker: str | None
    choices: list[str]
    doubt: str
    words: list[str] | None = None  # the words it is unsure of, when it is unsure of words


@dataclass(frozen=True, slots=True)
class Correction:
    session: str
    what: str  # same | different | not_voice | name | person | device | text | undo
    rows: list[int]
    label: str | None
    text: str | None
    line: int  # its line number in corrections.jsonl, the handle `undo` uses
    answers: str | None  # the question it answered, if it was an answer
    refused: str | None


KINDS = {
    "session": Session,
    "turns": Turns,
    "partial": Partial,
    "text": Text,
    "question": Question,
    "correction": Correction,
}
CORRECTIONS = ("same", "different", "not_voice", "name", "person", "device", "text")


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


class Inbox:
    """What anyone tells the ear: a person, the arbiter, or a voice. One line each.

    The same calls serve a terminal and a spoken dialogue: when the ear asks
    "which did you say?" the mouth reads the question, the ear hears the answer,
    and ``answer`` writes it here. The running ear applies every line in order.
    """

    def __init__(self, memory: Path):
        self.path = Path(memory) / "corrections.jsonl"

    def _write(self, line: dict) -> dict:
        from π_ear.io import append_jsonl, utc_now

        line = {**line, "utc": utc_now()}
        append_jsonl(self.path, line)
        return line

    def correct(
        self, kind: str, rows, label: str | None = None, text: str | None = None, **extra
    ) -> dict:
        if kind not in CORRECTIONS:
            raise ValueError(f"{kind!r} is not one of {CORRECTIONS}")
        line = {"kind": kind, "rows": [int(r) for r in rows], **extra}
        if label is not None:
            line["label"] = label
        if text is not None:
            line["text"] = text
        return self._write(line)

    def answer(self, question: str, text: str | None = None, choice: int | None = None) -> dict:
        """Answer a question of the record: a name, a reading's text, or a choice's index."""
        line = {"kind": "answer", "question": question, "rows": []}
        if text is not None:
            line["text"] = text
        if choice is not None:
            line["choice"] = int(choice)
        return self._write(line)

    def undo(self, line: int) -> dict:
        return self._write({"kind": "undo", "line": int(line), "rows": []})


def open_questions(path: Path, session: str | None = None) -> list[Question]:
    """The questions of the record that no correction has answered yet."""
    events = read(path, session)
    answered = {
        e.answers for e in events if isinstance(e, Correction) and e.answers and not e.refused
    }
    return [e for e in events if isinstance(e, Question) and e.id not in answered]
