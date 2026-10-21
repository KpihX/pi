"""The scribe: *what* was said, written as it is said, revised as soon as it can be.

The unit of text is a **piece**: a speech run cut at Silero's deepest dips, so that no
piece is longer than what the slow recogniser reads whole (Whisper: under 30 s). The
cuts are made as the run grows (the open part is cut at the deepest dip of its first
window as soon as it is longer than a window), so a piece is final while its run
still goes on: on studio talk that never pauses, the text of the first minute is
written in the first minute, not after the run ends.

Measured live (2026-10-20, two hours of a football broadcast heard by a laptop):
Whisper dropped everything after 30 s of a run (a 70.9 s run lost half its words), and
read worse the shorter its input (word recall 0.84 on 30 s of match commentary, 0.64 on
8 s); Parakeet answered in English on French studio talk the longer the piece (2 % of
the words at 4 s, 77 % at 32 s), while on match commentary it needed long pieces (0.30
at 2 s, 0.65 at 12 s). No length is right for every recogniser and every regime, so:

* pieces are cut where the voice changes (the turns of the model of who) and where the
  slow recogniser's window would overflow, at Silero's deepest dip, never shorter than
  the VAD's own shortest speech;
* the slow recogniser reads as many consecutive pieces as its window holds, and its
  timed sentences go back to the pieces they were said in;
* a translation by the fast recogniser is recognised by its language (``language.py``)
  and loses to the slow reading. A first build also re-read a piece in halves when the
  halves were "more in the voice's language": on short phrases the estimate was noise,
  the fast reading's recall on KπX's own segment fell from 0.26 to 0.09; removed.

One thread, the work in this order:

1. **settle**: a run grew or ended: new pieces are cut, read by the fast recogniser,
   decided, and written (``text``, revision 0);
2. **partial**: the open part of a run in progress is read again whenever the thread is
   free (latest audio wins): the text appears while it is said;
3. **correct**: what KπX said a piece was, absorbed (not while a person is speaking:
   the thread stays free for them);
4. **revise**: the slow recogniser reads a piece in the language its voice speaks
   (``language.py``), and the reader decides again (``text``, next revision). Most
   valuable first: pieces of people before pieces of devices, the most doubtful first.
   Not only in silences (the television never pauses), but never while a person is
   speaking.

Every decision is written with its doubt. A ``question`` is written when it is worth
someone's attention: the reader could not choose (``ambiguous``), understood only bits
(``partial``), or nothing of a known voice, or holds a word more probably wrong than
right (the question names those words), *and* the voice is a person (a commentator on
television is nobody to ask), known before the piece (a voice born of one window may be
nobody: on the replay of a goal, KπX's shouts made three such voices, each asked about;
a real one comes back), and that voice has no question still open.
"""

from __future__ import annotations

import heapq
import itertools
import threading
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from π_ear import asr, contract
from π_ear.asr import Hypothesis, Word
from π_ear.io import append_jsonl, read_jsonl, utc_now
from π_ear.lexicon import tokens
from π_ear.sense import Silero
from π_ear.words import NONE, Context, Heard, Reader, mean_logp

VOICED = ("known", "new")


@dataclass
class Piece:
    id: str
    run: str
    t0: float
    t1: float
    samples: np.ndarray | None
    speech: float | None
    rows: list[int] = field(default_factory=list)
    voices: dict[str, float] = field(default_factory=dict)  # share of the piece's time
    voice: str | None = None
    mixed: bool = False
    known: bool = False  # its voice was known before it: not a voice born in this piece
    fast: Hypothesis | None = None
    slow: Hypothesis | None = None
    language: str | None = None  # what the slow recogniser was told, if anything
    revision: int = -1
    asked: bool = False
    separated: bool = False
    decision: dict = field(default_factory=dict)


@dataclass
class Open:
    """A run in progress: its audio, Silero's frames over it, what is settled."""

    id: str
    start: float  # stream seconds
    silero: Silero
    audio: np.ndarray
    decisions: list[dict] = field(default_factory=list)
    settled: int = 0  # frames already in pieces
    count: int = 0
    finished: bool = False
    group: list[str] = field(default_factory=list)  # settled pieces still to be read slowly


def cut(probs: np.ndarray, start: int, most: int, floor: int = 1) -> int:
    """Where the open part [start, ...) is cut: Silero's deepest dip inside its first
    ``most`` frames, at least ``floor`` frames from its start (the VAD's own shortest
    speech: a first build cut 32 ms pieces)."""
    lo = start + min(max(floor, 1), most - 1)
    return lo + int(np.argmin(probs[lo : start + most]))


class Scribe:
    def __init__(
        self,
        reader: Reader,
        fast,
        slow,
        separator,
        write,
        memory: Path,
        models: Path,
        source_of=None,
        floor_s: float = 0.0,
        echo=None,
    ):
        self.reader = reader
        self.fast, self.slow, self.separator = fast, slow, separator
        self.write = write
        self.memory = memory
        self.models = models
        self.source_of = source_of or (lambda voice: (None, None))
        self.echo = echo
        windows = [w for w in (asr.window_s(fast), asr.window_s(slow) if slow else None) if w]
        # strictly shorter than the shortest window: Silero frames of 512 samples
        self.frame = Silero(models).block
        self.most = (
            int(min(windows) * contract.RATE / self.frame) - 1 if windows else None
        )  # frames
        self.floor = max(1, int(floor_s * contract.RATE / self.frame))
        self.lock = threading.Condition()
        self.open: dict[str, Open] = {}
        self.dirty: set[str] = set()  # open runs with new audio, to cut
        self.partials: set[str] = set()  # open runs whose open part is to be read again
        self.alternate = False  # a partial, then a revision, when both wait
        self.pieces: dict[str, Piece] = {}
        self.of_row: dict[int, list[str]] = {}  # a window straddling a cut is in two pieces
        self.revisions: list[tuple] = []  # a heap of (priority, order, piece id)
        self.order = itertools.count()
        self.corrections: list[tuple[str, str, int]] = []
        self.last_partial: dict[str, str] = {}
        self.retracted: set[str] = set()
        self.done: set[str] = set()  # runs settled to their end: a late `hear` is ignored
        self.speaking = False
        self.person_speaking = False
        self.busy = False
        self.questions = 0
        self.errors: list[str] = []
        self.open_questions: dict[str | None, str] = {}  # voice -> its open question
        self.pace = {"audio": 0.0, "spent": 0.0}  # the slow recogniser's own speed, measured
        # what the sensors usually say of the known voices of this place: Silero's mean,
        # and the fast recogniser's mean token log-probability
        self.regime = {"sum": 0.0, "count": 0.0, "logp": 0.0, "read": 0.0}
        threading.Thread(target=self._work, daemon=True).start()

    # -- what the ear hands over -----------------------------------------------------
    def set_speaking(self, speaking: bool, person: bool = False) -> None:
        with self.lock:
            self.speaking, self.person_speaking = speaking, speaking and person
            self.lock.notify_all()

    def hear(self, run: str, start: float, samples: np.ndarray, decisions: list[dict]) -> None:
        """The run in progress, all of it so far (the latest call wins)."""
        if self.fast is None:
            return
        with self.lock:
            state = self.open.get(run)
            if state is None:
                if run in self.retracted or run in self.done:
                    return
                state = self.open[run] = Open(run, start, Silero(self.models), samples[:0])
            state.audio, state.decisions = samples, list(decisions)
            self.dirty.add(run)
            self.lock.notify_all()

    def finish(self, run: str, start: float, samples: np.ndarray, decisions: list[dict]) -> None:
        if self.fast is None:
            return
        self.hear(run, start, samples, decisions)
        with self.lock:
            if run in self.open:
                self.open[run].finished = True
                self.dirty.add(run)
                self.lock.notify_all()

    def retract(self, run: str) -> None:
        """A run of known sounds only: nothing of it is text, and a partial is taken back."""
        with self.lock:
            self.retracted.add(run)
            self.open.pop(run, None)
            self.dirty.discard(run)
            for key in [k for k in self.last_partial if k.startswith(run + "#")]:
                if self.last_partial.pop(key):
                    self.write(
                        {"kind": "partial", "piece": key, "run": run, "text": "", "retracted": True}
                    )

    def correct(
        self, rows: list[int], truth: str, line: int, piece: str | None = None
    ) -> str | None:
        """A text correction: of one piece (an answer names it), or of the pieces its
        rows belong to, each then taught its share of the words."""
        with self.lock:
            if piece is not None:
                pieces = [piece] if piece in self.pieces or self._recall(rows, piece) else []
            else:
                pieces = list(dict.fromkeys(k for r in rows for k in self.of_row.get(r, [])))
                pieces = sorted(pieces or self._recall(rows), key=lambda k: self.pieces[k].t0)
            if not pieces:
                return "these rows belong to no transcribed piece"
            parts = [truth] if len(pieces) == 1 else self._share(pieces, truth)
            self.corrections += [(p, part, line) for p, part in zip(pieces, parts, strict=True)]
            self.lock.notify_all()
        return None

    def answered(self, voice: str | None) -> None:
        with self.lock:
            self.open_questions.pop(voice, None)

    def drain(self) -> None:
        """Wait until every run is settled and every piece is read by both recognisers."""
        with self.lock:
            self.speaking = self.person_speaking = False
            for state in self.open.values():
                state.finished = True
                self.dirty.add(state.id)
            self.lock.notify_all()
            while self.dirty or self.partials or self.revisions or self.corrections or self.busy:
                self.lock.wait()

    def pending(self) -> int:
        with self.lock:
            return (
                len(self.revisions) + len(self.corrections) + len(self.dirty) + len(self.partials)
            )

    def behind_s(self) -> float:
        """Seconds of audio waiting for the slow recogniser."""
        with self.lock:
            return sum(
                self.pieces[k].t1 - self.pieces[k].t0 for _, _, keys in self.revisions for k in keys
            )

    # -- the thread --------------------------------------------------------------------
    def _next(self):
        """The next job. Cutting runs into pieces comes first (cheap, and it gives every
        piece its first reading); then corrections; then partials and slow revisions in
        turn: live, a run grows every second, and a first build that always put the
        partial first never revised anything while the television spoke. Nothing slow
        while a person speaks: the scribe stays free for them."""
        with self.lock:
            while True:
                self.dirty &= set(self.open)
                self.partials &= set(self.open)
                finished = [r for r in self.dirty if self.open[r].finished]
                revise = bool(self.revisions) and not self.person_speaking
                if finished or self.dirty:
                    run = finished[0] if finished else next(iter(self.dirty))
                    self.dirty.discard(run)
                    job = ("settle", run)
                elif self.corrections and not self.person_speaking:
                    job = ("correction", self.corrections.pop(0))
                elif self.partials and not (revise and self.alternate):
                    job = ("partial", self.partials.pop())
                    self.alternate = True
                elif revise:
                    job = ("revision", heapq.heappop(self.revisions)[2])
                    self.alternate = False
                elif (
                    self.reader.dirty or self.reader.languages.dirty
                ) and not self.person_speaking:
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
            kind = what = None
            try:
                kind, what = self._next()
                getattr(self, f"_{kind}")(what)
            except Exception as error:  # a job that fails must not take the scribe with it
                self.errors.append(f"{kind}: {error!r}")
                self.write(
                    {
                        "kind": "error",
                        "job": kind,
                        "what": str(what)[:200],
                        "error": traceback.format_exc()[-2000:],
                    }
                )
            finally:
                with self.lock:
                    self.busy = False
                    self.lock.notify_all()

    def _settle(self, run: str) -> None:
        with self.lock:
            state = self.open.get(run)
            if state is None:
                return
            audio, decisions, finished = state.audio, state.decisions, state.finished
        fed = len(state.silero.probs) * self.frame + len(state.silero.pending)
        state.silero.feed(audio[fed:])
        probs = np.asarray(state.silero.probs)
        total = len(audio) // self.frame
        turns = self._turn_cuts(state, probs, decisions, total)
        while True:
            turns = [t for t in turns if t > state.settled]
            if turns and (not self.most or turns[0] - state.settled <= self.most):
                end = turns[0]  # the voice changes: a piece is one voice's
            elif self.most and total - state.settled > self.most:
                end = cut(probs, state.settled, self.most, self.floor)  # the window is full
            else:
                break
            self._piece(state, state.settled, end, total, audio, probs, decisions)
            state.settled = end
        if finished:
            self._piece(state, state.settled, total, total, audio, probs, decisions)
            self._flush(state)
            with self.lock:
                self.open.pop(run, None)
                self.dirty.discard(run)
                self.done.add(run)
                self.last_partial.pop(f"{run}#{state.count}", None)
            return
        with self.lock:
            self.partials.add(run)
            self.lock.notify_all()

    def _turn_cuts(self, state: Open, probs, decisions, total: int) -> list[int]:
        """Where the voice changes inside the open part: between the middles of two
        consecutive learned windows given to two different voices, at Silero's deepest
        dip, when the change is more probably real than not. A piece is one voice's
        phrase, so its text is that voice's (an interview's question and answer,
        measured live in one run, become two pieces)."""
        frames = contract.RATE / self.frame
        voiced = sorted(
            (d for d in decisions if d["learned"] and d["kind"] in VOICED and d["speaker"]),
            key=lambda d: d["t0"],
        )
        cuts, last = [], state.settled
        for a, b in itertools.pairwise(voiced):
            if a["speaker"] == b["speaker"]:
                continue
            # a change is a cut only if it is more probably real than not, by the model of
            # who's own calibrated trust (Q1): on fast match commentary, cuts between two
            # unsure windows made fragments nobody could read (measured on the replay)
            # (a window the model could not calibrate, trust None, keeps its argmax; a
            # trust of 0 is a trust, not a missing one: a first build read it as 1)
            both = np.prod([1 if d.get("trust") is None else d["trust"] for d in (a, b)])
            if both < 1 - both:
                continue
            lo = int(((a["t0"] + a["t1"]) / 2 - state.start) * frames)
            hi = int(((b["t0"] + b["t1"]) / 2 - state.start) * frames)
            lo, hi = max(lo, last + self.floor), min(hi, total - self.floor)
            if hi - lo < 1:
                continue
            at = lo + int(np.argmin(probs[lo:hi]))
            cuts.append(at)
            last = at
        return cuts

    def _piece(self, state: Open, a: int, b: int, total: int, audio, probs, decisions) -> None:
        """Frames [a, b) of a run become a piece (the last one takes the run's tail)."""
        lo, hi = a * self.frame, len(audio) if b >= total else b * self.frame
        t0, t1, samples = (
            state.start + lo / contract.RATE,
            state.start + hi / contract.RATE,
            audio[lo:hi],
        )
        if not len(samples):
            return
        key = f"{state.id}#{state.count}"
        state.count += 1
        covering = [d for d in decisions if d["t0"] < t1 and t0 < d["t1"]]
        share: Counter = Counter()
        for d in covering:
            if d["kind"] in VOICED and d["speaker"]:
                share[d["speaker"]] += min(t1, d["t1"]) - max(t0, d["t0"])
        mass = sum(share.values())
        frames = probs[a : max(b, a + 1)]
        piece = Piece(
            id=key,
            run=state.id,
            t0=round(t0, 3),
            t1=round(t1, 3),
            samples=samples,
            speech=float(np.mean(frames)) if len(frames) else None,
            rows=[d["row"] for d in covering],
            voices={k: round(v / mass, 3) for k, v in share.most_common()} if mass else {},
            voice=share.most_common(1)[0][0] if share else None,
            mixed=any(d["kind"] in ("noise", "unknown") for d in covering if d["learned"]),
        )
        piece.known = any(d["kind"] == "known" and d["speaker"] == piece.voice for d in covering)
        if piece.voice is not None and piece.speech is not None:
            self.regime["sum"] += piece.speech
            self.regime["count"] += 1
        piece.fast = asr.read(self.fast, samples)
        self.reader.heard(piece.fast.text)
        if piece.voice is not None and any(w.logp is not None for w in piece.fast.words):
            self.regime["logp"] += mean_logp(piece.fast)
            self.regime["read"] += 1
        with self.lock:
            self.pieces[key] = piece
            for row in piece.rows:
                self.of_row.setdefault(row, []).append(key)
        self._decide(piece)
        if self.slow is not None:
            self._group(state, piece)
        else:
            piece.samples = None  # nothing will read it again

    def _partial(self, run: str) -> None:
        with self.lock:
            state = self.open.get(run)
            if state is None:
                return
            audio = state.audio
        lo = state.settled * self.frame
        if len(audio) - lo <= 0:
            return
        heard = asr.read(self.fast, audio[lo:])
        key = f"{state.id}#{state.count}"
        with self.lock:
            if state.id in self.retracted or state.id not in self.open:
                return
            if heard.text != self.last_partial.get(key, ""):
                self.last_partial[key] = heard.text
                t0 = round(state.start + lo / contract.RATE, 3)
                self.write(
                    {"kind": "partial", "piece": key, "run": state.id, "t0": t0, "text": heard.text}
                )

    def _group(self, state: Open, piece: Piece) -> None:
        """Pieces are read slowly together, as many consecutive ones as the slow
        recogniser's window holds: Whisper reads better with context (word recall 0.84
        on 30 s of match commentary, 0.64 on 8 s pieces, measured live) and wrote "..."
        for short pieces it heard alone; its timed sentences then go back to their pieces."""
        window = (self.most or 0) * self.frame / contract.RATE
        if state.group and piece.t1 - self.pieces[state.group[0]].t0 >= window:
            self._flush(state)
        state.group.append(piece.id)

    def _flush(self, state: Open) -> None:
        keys, state.group = tuple(state.group), []
        if not keys:
            return
        pieces = [self.pieces[k] for k in keys]
        person = any(self.source_of(p.voice)[0] == "person" for p in pieces)
        doubt = min(p.decision.get("p", 0.0) for p in pieces)
        # people first, then the most doubtful, then the oldest
        priority = (0 if person else 1, doubt, pieces[0].t0)
        with self.lock:
            heapq.heappush(self.revisions, (priority, next(self.order), keys))
            self.lock.notify_all()

    def _revision(self, keys: tuple[str, ...]) -> None:
        pieces = [self.pieces[k] for k in keys]
        start = pieces[0].t0
        audio = np.concatenate([p.samples for p in pieces])
        separated = any(p.mixed for p in pieces) and self.separator is not None
        if separated:
            audio = self.separator(audio)
        seconds = len(audio) / contract.RATE
        share: Counter = Counter()
        for p in pieces:
            if p.voice is not None:
                share[p.voice] += p.t1 - p.t0
        voice = share.most_common(1)[0][0] if share else None
        told = self.reader.languages.choose(voice, seconds) if seconds else None
        started = time.monotonic()
        heard = asr.read(self.slow, audio, told or "")
        self.pace["spent"] += time.monotonic() - started
        self.pace["audio"] += seconds
        self.reader.heard(heard.text)
        languages = self.reader.languages
        # a detection is evidence before the decision, counted by how often this voice's
        # detections this long are right: on the first group already, Parakeet's English
        # for a French player is a translation (measured live, cold start). A told
        # reading only teaches what its language is written with.
        languages.heard(voice, heard.language or told, heard.text, told is None, seconds)
        if languages.dirty:
            # now, not when the scribe is idle: the television never lets it be, and on
            # the replay of KπX's segment the voice's regime was never learned in time
            # (59 groups of 60 left to detect); it costs a tenth of what Whisper does
            languages.refit()
        for piece in pieces:
            lo, hi = piece.t0 - start, piece.t1 - start
            mine = [w for w in heard.words if lo <= (w.t0 + w.t1) / 2 < hi]
            if len(pieces) == 1:
                mine = list(heard.words)
            words = tuple(Word(w.text, w.t0 - lo, w.t1 - lo, w.logp) for w in mine)
            text = " ".join(w.text for w in mine) if len(pieces) > 1 else heard.text
            piece.slow = Hypothesis(text, words, heard.language, heard.source)
            piece.language, piece.separated = told, separated
            self._decide(piece)
            self.reader.agree(piece.fast, piece.slow, piece.voice)
            piece.samples = None  # read by both: a two-hour session kept 460 MB of them

    def _correction(self, job) -> None:
        key, truth, line = job
        piece = self.pieces[key]
        heard = Heard(piece.fast, piece.slow, truth, self._context(piece))
        self.reader.learn(heard)
        append_jsonl(
            self.memory / "taught.jsonl",
            {
                "line": line,
                "piece": key,
                "voice": piece.voice,
                "speech": piece.speech,
                "regime": heard.context.regime,
                "confidence": heard.context.confidence,
                "fast": piece.fast.to_dict(),
                "slow": piece.slow.to_dict() if piece.slow else None,
                "truth": truth,
            },
        )
        self.reader.refit()
        piece.asked = True
        self._decide(piece)

    def _refit(self, _) -> None:
        try:
            self.reader.refit()
        finally:  # a refit that fails is written as an error, never retried in a loop
            self.reader.dirty = self.reader.languages.dirty = False

    # -- the decision, written ------------------------------------------------------------
    def _context(self, piece: Piece) -> Context:
        """The piece, against what the sensors usually say here: the means over the
        known voices of this place, with the base's own as one pseudo-piece before any
        (the reader's base pieces were measured that way)."""
        base = self.reader.base

        def usual(total: float, count: float, prior: float | None) -> float | None:
            mass = count + (1 if prior is not None else 0)
            return (total + (prior or 0.0)) / mass if mass else None

        return Context(
            piece.voice,
            piece.speech,
            usual(self.regime["sum"], self.regime["count"], base.get("regime")),
            piece.t1 - piece.t0,
            usual(self.regime["logp"], self.regime["read"], base.get("confidence")),
        )

    def _remember(self, piece: Piece) -> None:
        """Every reading kept: a correction may come sessions later, and finds it here."""
        append_jsonl(
            self.memory / "pieces.jsonl",
            {
                "piece": piece.id,
                "run": piece.run,
                "t0": piece.t0,
                "t1": piece.t1,
                "rows": piece.rows,
                "voice": piece.voice,
                "voices": piece.voices,
                "revision": piece.revision,
                "speech": piece.speech,
                "language": piece.language,
                "fast": piece.fast.to_dict() if piece.fast else None,
                "slow": piece.slow.to_dict() if piece.slow else None,
                "separated": piece.separated,
            },
        )

    def _recall(self, rows: list[int], piece: str | None = None) -> list[str]:
        """The pieces of earlier sessions these rows (or this piece) belong to (``pieces.jsonl``)."""
        found: dict[str, dict] = {}
        wanted = set(rows)
        for line in read_jsonl(self.memory / "pieces.jsonl")[0]:
            if line["piece"] == piece or (piece is None and wanted & set(line["rows"])):
                found[line["piece"]] = line  # the latest reading of each piece wins
        for key, line in found.items():
            self.pieces[key] = Piece(
                id=key,
                run=line["run"],
                t0=line["t0"],
                t1=line["t1"],
                samples=None,
                speech=line.get("speech"),
                rows=line["rows"],
                voices=line.get("voices", {}),
                voice=line["voice"],
                fast=Hypothesis.from_dict(line["fast"]),
                slow=Hypothesis.from_dict(line["slow"]),
                language=line.get("language"),
                revision=line["revision"],
                separated=line["separated"],
                asked=True,
            )
            for row in line["rows"]:
                self.of_row.setdefault(row, []).append(key)
        return sorted(found, key=lambda k: self.pieces[k].t0)

    def _share(self, keys: list[str], truth: str) -> list[str]:
        """One truth over several pieces: its words shared in proportion to their readings."""
        words = truth.split()
        sizes = [
            max(len((self.pieces[k].slow or self.pieces[k].fast).text.split()), 1) for k in keys
        ]
        bounds = np.round(np.cumsum([0] + sizes) / sum(sizes) * len(words)).astype(int)
        return [" ".join(words[a:b]) for a, b in itertools.pairwise(bounds)]

    def _decide(self, piece: Piece) -> dict:
        decided = self.reader.read(piece.fast, piece.slow, self._context(piece))
        piece.revision += 1
        piece.decision = decided
        self._remember(piece)
        heard = piece.slow or piece.fast
        self.write(
            {
                "kind": "text",
                "piece": piece.id,
                "run": piece.run,
                "t0": piece.t0,
                "t1": piece.t1,
                "rows": piece.rows,
                "speaker": piece.voice,
                "speakers": piece.voices,
                "revision": piece.revision,
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
                "told": piece.language,
                "speech": None if piece.speech is None else round(piece.speech, 3),
                "separated": piece.separated,
                "written": utc_now(),
            }
        )
        if self.echo:
            marked = " ".join(w["w"] if w["agreed"] else f"[{w['w']}?]" for w in decided["words"])
            self.echo(
                f"[{piece.t0:8.2f}-{piece.t1:8.2f}] text r{piece.revision} {decided['doubt']:14s} "
                f"p={decided['p']:.2f} {marked or '∅'}"
            )
        self._ask(piece, decided)
        return decided

    def _ask(self, piece: Piece, decided: dict) -> None:
        final = piece.slow is not None or self.slow is None
        doubt = decided["doubt"]
        # the words more probably wrong than right: a sure sentence around one unsure
        # name is still worth one question, about that name
        doubtful = [
            w["w"] for w in decided["words"] if w["trust"] < 1 - w["trust"] and w["new"]
        ]  # a word KπX confirmed for this voice is not asked again, however unsure
        worth = (
            doubt in ("ambiguous", "partial")
            or (doubt == "not_understood" and piece.voice is not None)
            or bool(doubtful)
        )
        source, _ = self.source_of(piece.voice)
        with self.lock:
            if (
                not final
                or piece.asked
                or not worth
                or source != "person"
                or not piece.known  # a voice of one window may be nobody: if real, it returns
                or piece.voice in self.open_questions
            ):
                return
            piece.asked = True
            self.questions += 1
            self.open_questions[piece.voice] = f"{piece.id}#text"
        choices = [r["text"] for r in decided["readings"] if r["text"] != NONE]
        self.write(
            {
                "kind": "question",
                "id": f"{piece.id}#text",
                "about": "text",
                "piece": piece.id,
                "rows": piece.rows,
                "speaker": piece.voice,
                "choices": choices,
                "doubt": doubt,
                "words": doubtful,
            }
        )

    def stats(self) -> dict:
        pace = self.pace["spent"] / self.pace["audio"] if self.pace["audio"] else None
        return {
            "pieces": len(self.pieces),
            "behind_s": round(self.behind_s(), 1),
            "slow_seconds_per_second": None if pace is None else round(pace, 3),
            "questions": self.questions,
            "errors": len(self.errors),
            "regime": None
            if not self.regime["count"]
            else round(self.regime["sum"] / self.regime["count"], 3),
            "confidence": None
            if not self.regime["read"]
            else round(self.regime["logp"] / self.regime["read"], 3),
        }


def rebuild_reader(
    memory: Path, base: dict, undone: set[int], current: Reader | None = None
) -> Reader:
    """The reader again, from its base: every kept text correction, in order, then a refit.

    What the stream taught without a correction survives (``current``): the languages
    of its voices, how its readings repeat, the words both recognisers confirmed. A first
    build rebuilt from the base alone, and one undo made every voice's language and
    every confirmed word forgotten."""
    taught = read_jsonl(memory / "taught.jsonl")[0]
    reader = Reader(base)
    if current is not None:
        reader.languages.load(current.languages.to_dict())
        reader.repeats = Counter(current.repeats)
        lexicon = current.memory.to_dict()["lexicon"]
        reader.memory.load({"lexicon": lexicon})
        for line in taught:  # the corrections' own words, replayed below if kept
            words = reader.memory.lexicon[line["voice"] or ""]
            words.subtract(Counter(tokens(line["truth"])))
            for word in [w for w, n in words.items() if n <= 0]:
                del words[word]
    for line in taught:
        if line["line"] in undone:
            continue
        reader.learn(
            Heard(
                Hypothesis.from_dict(line["fast"]),
                Hypothesis.from_dict(line["slow"]),
                line["truth"],
                Context(
                    line["voice"],
                    line.get("speech"),
                    line.get("regime"),
                    None,
                    line.get("confidence"),
                ),
            )
        )
    reader.refit()
    return reader
