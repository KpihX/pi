"""The scribe: *what* was said, written as it is said, revised in the silences.

One thread, three kinds of work, in this order of priority:

1. **final**: a speech run just ended. The fast recogniser reads it whole, the
   reader decides, and a ``text`` line (revision 0) is written at once;
2. **partial**: a run is still going on. The fast recogniser re-reads it from its
   start whenever it is free (the latest audio wins, nothing queues up), and a
   ``partial`` line is written when the words change: the text appears while it
   is being said;
3. **revision**, only while the room is silent: the slow recogniser reads the
   run again (on the voice separated from the music, if a separator is plugged
   in and the run mixed both),
   the reader decides between all the readings, and a ``text`` line with the
   next revision is written. Corrections, too, are absorbed here: the weights
   are refitted and the corrected run is read again.

What the reader decided is never hidden: each ``text`` line carries the chosen
reading, its probability, its margin, its named doubt, its words with their
own trust, and the other readings. When the doubt is ``ambiguous``, or
``not_understood`` in a run of a known voice, a ``question`` is written: the arbiter, or the operator, or
a voice, may answer it, and the answer is a correction like any other.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from π_ear.asr import Hypothesis
from π_ear.io import append_jsonl, read_jsonl, utc_now
from π_ear.words import NONE, Reader


@dataclass
class RunText:
    run: str
    t0: float
    t1: float
    rows: list[int]
    voice: str | None
    samples: np.ndarray | None
    mixed: bool = False
    speech: float | None = None  # Silero's mean speech probability over the run
    fast: Hypothesis | None = None
    slow: Hypothesis | None = None
    revision: int = -1
    separated: bool = False
    asked: bool = False
    decision: dict = field(default_factory=dict)


class Scribe:
    def __init__(self, reader: Reader, fast, slow, separator, write, memory: Path, echo=None):
        self.reader = reader
        self.fast, self.slow, self.separator = fast, slow, separator
        self.write = write
        self.memory = memory
        self.echo = echo
        self.lock = threading.Condition()
        self.runs: dict[str, RunText] = {}
        self.of_row: dict[int, str] = {}
        self.finals: list[str] = []
        self.revisions: list[str] = []
        self.corrections: list[tuple[str, str, int]] = []
        self.live_partial: tuple[str, np.ndarray, list[int]] | None = None
        self.last_partial: dict[str, str] = {}
        self.closed: set[str] = set()  # runs finished or retracted: no partial after them
        self.speaking = False
        self.busy = False
        self.questions = 0
        threading.Thread(target=self._work, daemon=True).start()

    # -- what the ear hands over ------------------------------------------------------
    def set_speaking(self, speaking: bool) -> None:
        with self.lock:
            self.speaking = speaking
            self.lock.notify_all()

    def partial(self, run: str, samples: np.ndarray, rows: list[int]) -> None:
        if self.fast is None:
            return
        with self.lock:
            self.live_partial = (run, samples, rows)
            self.lock.notify_all()

    def finish(self, text: RunText) -> None:
        if self.fast is None:
            return
        with self.lock:
            self.closed.add(text.run)
            self.runs[text.run] = text
            for row in text.rows:
                self.of_row[row] = text.run
            if self.live_partial and self.live_partial[0] == text.run:
                self.live_partial = None
            self.finals.append(text.run)
            self.lock.notify_all()

    def retract(self, run: str) -> None:
        """A run with no recognised voice: its partial text, if any, was not speech."""
        with self.lock:
            self.closed.add(run)
            if self.live_partial and self.live_partial[0] == run:
                self.live_partial = None
            if self.last_partial.pop(run, None):
                self.write({"kind": "partial", "run": run, "text": "", "retracted": True})

    def correct(self, rows: list[int], truth: str, line: int) -> str | None:
        """A text correction: the run its rows belong to is the one corrected."""
        with self.lock:
            runs = list(dict.fromkeys(self.of_row[r] for r in rows if r in self.of_row))
            runs = runs or self._recall(rows)
            if not runs:
                return "these rows belong to no transcribed run"
            for run in runs:
                self.corrections.append((run, truth, line))
            self.lock.notify_all()
        return None

    def drain(self) -> None:
        """Wait until every pending reading, revision and correction is written."""
        with self.lock:
            self.speaking = False
            self.lock.notify_all()
            while (
                self.finals or self.revisions or self.corrections or self.busy or self.live_partial
            ):
                self.lock.wait()

    def pending(self) -> int:
        return len(self.finals) + len(self.revisions) + len(self.corrections)

    # -- the thread ------------------------------------------------------------------------
    def _next(self):
        with self.lock:
            while True:
                if self.finals:
                    job = ("final", self.finals.pop(0))
                elif self.corrections and not self.speaking:
                    job = ("correction", self.corrections.pop(0))
                elif self.live_partial is not None:
                    job, self.live_partial = ("partial", self.live_partial), None
                elif self.revisions and not self.speaking:
                    job = ("revision", self.revisions.pop(0))
                elif self.reader.dirty and not self.speaking:
                    job = ("refit", None)
                else:
                    self.busy = False
                    self.lock.notify_all()
                    self.lock.wait()
                    continue
                self.busy = True
                return job

    def _work(self) -> None:
        while True:
            kind, what = self._next()
            try:
                getattr(self, f"_{kind}")(what)
            finally:
                with self.lock:
                    self.busy = False
                    self.lock.notify_all()

    def _partial(self, job) -> None:
        run, samples, rows = job
        heard = self.fast(samples)
        with self.lock:  # a run closed while it was being read keeps its last word
            if run not in self.closed and heard.text != self.last_partial.get(run, ""):
                self.last_partial[run] = heard.text
                self.write({"kind": "partial", "run": run, "text": heard.text, "rows": rows})

    def _final(self, run: str) -> None:
        text = self.runs[run]
        text.fast = self.fast(text.samples)
        self.last_partial.pop(run, None)
        self._decide(text)
        if self.slow is not None:
            with self.lock:
                self.revisions.append(run)
                self.lock.notify_all()

    def _revision(self, run: str) -> None:
        text = self.runs[run]
        audio = text.samples
        if text.mixed and self.separator is not None:
            audio, text.separated = self.separator(audio), True
        text.slow = self.slow(audio)
        self._decide(text)
        # after the decision: a word heard for the first time is "new" this time,
        # and known from then on, never confirmed by the very reading it is in
        self.reader.agree(text.fast, text.slow, text.voice)

    def _correction(self, job) -> None:
        run, truth, line = job
        text = self.runs[run]
        self.reader.learn(text.fast, text.slow, text.voice, truth, text.speech)
        append_jsonl(
            self.memory / "taught.jsonl",
            {
                "line": line,
                "run": run,
                "voice": text.voice,
                "fast": text.fast.to_dict(),
                "slow": text.slow.to_dict() if text.slow else None,
                "speech": text.speech,
                "truth": truth,
            },
        )
        self.reader.refit()
        self._decide(text)

    def _refit(self, _) -> None:
        self.reader.refit()

    # -- the decision, written ------------------------------------------------------------
    def _remember(self, text: RunText) -> None:
        """Every reading kept: a correction may come sessions later, and finds it here."""
        append_jsonl(
            self.memory / "runs.jsonl",
            {
                "run": text.run,
                "t0": text.t0,
                "t1": text.t1,
                "rows": text.rows,
                "voice": text.voice,
                "revision": text.revision,
                "speech": text.speech,
                "fast": text.fast.to_dict() if text.fast else None,
                "slow": text.slow.to_dict() if text.slow else None,
                "separated": text.separated,
            },
        )

    def _recall(self, rows: list[int]) -> list[str]:
        """The runs of earlier sessions these rows belong to, from ``runs.jsonl``."""
        found: dict[str, dict] = {}
        wanted = set(rows)
        for line in read_jsonl(self.memory / "runs.jsonl")[0]:
            if wanted & set(line["rows"]):
                found[line["run"]] = line  # the latest reading of each run wins
        for run, line in found.items():
            self.runs[run] = RunText(
                run=run,
                t0=line["t0"],
                t1=line["t1"],
                rows=line["rows"],
                voice=line["voice"],
                samples=None,
                speech=line.get("speech"),
                fast=Hypothesis.from_dict(line["fast"]),
                slow=Hypothesis.from_dict(line["slow"]),
                revision=line["revision"],
                separated=line["separated"],
                asked=True,
            )
            for row in line["rows"]:
                self.of_row[row] = run
        return list(found)

    def _decide(self, text: RunText) -> None:
        decided = self.reader.read(text.fast, text.slow, text.voice, text.speech)
        text.revision += 1
        text.decision = decided
        self._remember(text)
        heard = text.slow or text.fast
        self.write(
            {
                "kind": "text",
                "run": text.run,
                "t0": text.t0,
                "t1": text.t1,
                "rows": text.rows,
                "speaker": text.voice,
                "revision": text.revision,
                "text": decided["text"],
                "p": round(decided["p"], 6),
                "margin": round(decided["margin"], 3),
                "doubt": decided["doubt"],
                "origins": decided["origins"],
                "words": [
                    {
                        "w": w["w"],
                        "trust": round(w["trust"], 4),
                        "agreed": w["agreed"],
                        "new": w["new"],
                    }
                    for w in decided["words"]
                ],
                "readings": [
                    {"text": r["text"], "p": round(r["p"], 6), "origins": r["origins"]}
                    for r in decided["readings"]
                ],
                "language": heard.language if heard else None,
                "separated": text.separated,
                "written": utc_now(),
            }
        )
        if self.echo:
            marked = " ".join(w["w"] if w["agreed"] else f"[{w['w']}?]" for w in decided["words"])
            self.echo(
                f"[{text.t0:8.2f}-{text.t1:8.2f}] text r{text.revision} {decided['doubt']:14s} "
                f"p={decided['p']:.2f} {marked or '∅'}"
            )
        final = text.slow is not None or self.slow is None
        # Ask when the ear hesitates between readings, or when someone it knows spoke
        # and it did not understand; a sound nobody owns, not understood, is no question.
        ask = decided["doubt"] == "ambiguous" or (
            decided["doubt"] == "not_understood" and text.voice is not None
        )
        if ask and not text.asked and final:
            text.asked = True
            self.questions += 1
            choices = [r["text"] for r in decided["readings"] if r["text"] != NONE]
            self.write(
                {
                    "kind": "question",
                    "id": f"{text.run}#text",
                    "about": "text",
                    "run": text.run,
                    "rows": text.rows,
                    "speaker": text.voice,
                    "choices": choices,
                    "doubt": decided["doubt"],
                }
            )


def rebuild_reader(memory: Path, base: dict, undone: set[int]) -> Reader:
    """The reader again, from its base: every kept text correction, in order, then a refit."""
    reader = Reader(base)
    for line in read_jsonl(memory / "taught.jsonl")[0]:
        if line["line"] in undone:
            continue
        reader.learn(
            Hypothesis.from_dict(line["fast"]),
            Hypothesis.from_dict(line["slow"]),
            line["voice"],
            line["truth"],
            line.get("speech"),
        )
    reader.refit()
    return reader
