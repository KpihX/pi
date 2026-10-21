"""The ear's port: what it publishes, frozen and versioned (``PORT``).

Two directions, both files, both append-only:

* **out**, the record ``events.jsonl``: what the ear heard and decided. A
  consumer (the arbiter first) reads THIS, never the ear's internals. Every line
  carries ``"port": 4``; lines of an older port are skipped, never guessed;
* **in**, the inbox ``corrections.jsonl``: what anyone tells the ear, a person
  at a terminal, the arbiter, or a voice through the mouth and the ear itself.
  ``Inbox`` writes it; the running ear obeys at its next tick.

    Session     a session starts: which weights, which recognisers
    Decision    one window: who (voice, condition), person or device, how sure
    Turns       one run, who spoke when, to the half window
    Partial     the text of a piece still being said (provisional)
    Text        a piece's reading, revised as evidence lands: words, trust, doubt
    Revision    a window decided again: who the ear says it was now, and why
    Measure     the room measured by the stream: what the microphone read of what
                the machine played, against the machine's own reading
    Question    what the ear wants to know: a name, which reading was said, or
                whether two voices are one
    Correction  one case told to the ear, as it applied it (or why it refused)

Port 3 (2026-10-20, after the first live test): the unit of text is a *piece* of a
speech run (cut at Silero's dips, shorter than the slow recogniser's window), named
``piece``; ``run`` is the run it is cut from. In port 2 ``run`` named the whole run.
A scribe job that fails is written as ``"kind": "error"``, outside the port (a
consumer never needs it; ``ctl status`` counts them).

Port 4 (2026-10-21, the ambient ear): two channels, the microphone (the room) and
the machine's own sound (what this machine plays, captured before the air). A
``Decision`` and a ``Turns`` say which ``channel`` they were heard on; a microphone
decision carries ``echo``, the machine's share of its energy (what the echo canceller
took out); a machine decision and a ``Text`` carry ``apps``, the applications that
were playing. Every voiced decision and every text says its ``scene``: room (a person
here), remote (a person reached through a device), media (a recording), self (π); a
microphone text that repeats what the machine's channel wrote is ``echo_of`` it. No
field of port 3 changed meaning.

Port 5 (2026-10-21, second thoughts, ``revise.py``): a decision once written can be
said again. A ``Revision`` names the window (``revises``, its row), what it was said to
be, what it is now, and why (merged | corrected | learned); a run's ``Turns`` are
written again with ``revision`` counting how many times (0 at first); its pieces'
``Text`` are revised under their voice now. A ``Question`` may be about ``same``: are
these two ``speakers`` one voice, with the model's ``p`` and the ``value`` of an answer
(the decisions it is expected to change); its answer is ``same`` or ``different``. A
consumer keeps the latest of each. A ``Decision`` says its ``overlap``: the segmenter's
P(two voices at once in the window) (``overlap.py``); a window more probably a mix than
not is decided but never learned. A ``Measure`` is written when a microphone piece that
heard the machine through the air is read: its ``truth`` (the machine's own reading of
that stretch) and each recogniser's ``recall`` of it; the reader learns from it as from a
correction. No field of port 4 changed meaning.

Changing a field's meaning is a new PORT number, never an edit of this one.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, fields
from pathlib import Path

from π_ear.io import read_jsonl

PORT = 5


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
    rel_db: float | None  # level above this session's non-speech floor (of its channel)
    channel: str | None  # mic (the room) | machine (what this machine plays)
    echo: float | None  # mic: the machine's share of the window's energy, taken out
    apps: list | None  # machine: the applications playing during the window
    scene: str | None  # room | remote | media | self: where the voice comes from (scene.py)
    scene_p: float | None
    overlap: float | None = None  # P(two voices at once in the window), the segmenter's


@dataclass(frozen=True, slots=True)
class Turns:
    session: str
    run: str
    t0: float
    t1: float
    segments: list  # [t0, t1, voice | "unknown" | "noise"]
    channel: str | None
    revision: int | None  # 0, then once more each time a window of it is revised


@dataclass(frozen=True, slots=True)
class Revision:
    session: str
    revises: int  # the row of the window decided again
    run: str | None  # the run it is in, when closed
    channel: str | None
    t0: float
    t1: float
    was: dict  # {"kind", "speaker"} as it was said
    now: dict  # {"kind", "speaker", "label", "scene"} as it is said now
    why: str  # merged | corrected | learned


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
    channel: str | None  # mic | machine
    apps: list | None  # machine: what played it
    echo_of: list | None  # mic: the machine's pieces it repeats (the machine heard through the air)
    scene: str | None  # where its voice comes from: room | remote | media | self


@dataclass(frozen=True, slots=True)
class Measure:
    session: str
    piece: str  # the microphone piece
    of: list  # the machine's pieces it heard
    truth: str  # what the machine's channel read of that stretch
    recall: dict  # recogniser -> the share of the truth's words it read


@dataclass(frozen=True, slots=True)
class Question:
    session: str
    id: str
    about: str  # name | text | same
    piece: str | None  # the piece it is about (text questions)
    rows: list[int]
    speaker: str | None
    choices: list[str]
    doubt: str
    words: list[str] | None = None  # the words it is unsure of, when it is unsure of words
    speakers: list[str] | None = None  # same: the two voices it asks about
    p: float | None = None  # same: the model's P(one voice)
    value: float | None = None  # same: the decisions an answer is expected to change


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
    "revision": Revision,
    "measure": Measure,
    "correction": Correction,
}
CORRECTIONS = (
    "same",
    "different",
    "not_voice",
    "name",
    "person",
    "device",
    "room",
    "remote",
    "media",
    "self",
    "text",
)


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
