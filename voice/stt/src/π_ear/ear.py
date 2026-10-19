"""The ear server: one loop, live, with its memory on disk.

    chunks -> Sensor (VAD + CAM++) -> Model (who) -> record + state
                     \\-> Scribe thread -> Whisper (what was said, per finished run)
    corrections.jsonl -> Model.correct  (obeyed at the next tick; `undo` replays)

Two places on disk, on purpose:

* ``root`` (``/tmp/π``, RAM): the live view. ``state.json``, ``events.jsonl``,
  the audio directory, ``PAUSE`` and ``RESET``. Losing it loses nothing learned;
* ``memory`` (``~/.cache/π/ear``): what the ear has learned. ``model.json`` (the
  checkpoint), ``embeddings.f32`` (one row per observation), ``observations.jsonl``
  (when and how each row was decided) and ``corrections.jsonl``. It is read back
  at launch, so a voice keeps its key across restarts, and the whole model can be
  rebuilt by replay, which is what makes ``undo`` exact.

The two clocks, in this module: the decision loop never waits for Whisper. A
finished speech run is queued to a scribe thread, and its text lands in the
record later, as its own line (measured in v1.0 on the live microphone: 144 % of
a core while the loop transcribed inline).
"""

from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from π_ear import contract
from π_ear.io import (
    Corpus,
    EmbeddingStore,
    Tailer,
    append_jsonl,
    archive,
    read_json,
    read_jsonl,
    stamp,
    utc_now,
    write_atomic,
)
from π_ear.model import Model, Weights
from π_ear.port import PORT
from π_ear.sense import Sensor, Transcriber, rms_dbfs

VOICED = ("known", "new")


@dataclass(slots=True)
class Places:
    root: Path = contract.ROOT
    memory: Path = contract.MEMORY
    models: Path = contract.MODELS
    audio: Path | None = None
    corpus: Path | None = contract.CORPUS
    corpus_max_chunks: int = contract.CORPUS_MAX_CHUNKS
    weights: Path = field(default_factory=lambda: base_weights_path(contract.ENCODER))

    def __post_init__(self):
        self.root, self.memory, self.models = Path(self.root), Path(self.memory), Path(self.models)

    @property
    def audio_dir(self) -> Path:
        return Path(self.audio) if self.audio else self.root / "audio"

    state = property(lambda self: self.root / "state.json")
    events = property(lambda self: self.root / "events.jsonl")
    pause = property(lambda self: self.root / "PAUSE")
    reset = property(lambda self: self.root / "RESET")
    snapshots = property(lambda self: self.root / "snapshots")
    checkpoint = property(lambda self: self.memory / "model.json")
    embeddings = property(lambda self: self.memory / "embeddings.f32")
    observations = property(lambda self: self.memory / "observations.jsonl")
    corrections = property(lambda self: self.memory / "corrections.jsonl")

    def ensure(self) -> None:
        for path in (self.root, self.audio_dir, self.memory):
            path.mkdir(parents=True, exist_ok=True)


def base_weights_path(encoder: str) -> Path:
    return contract.WEIGHTS / f"base-{Path(encoder).stem}.json"


def load_memory(places: Places) -> tuple[Model, EmbeddingStore, dict]:
    """The learned state, read-only: what `why`, `correct --speaker` and tests look at."""
    weights = Weights.from_dict(read_json(places.weights))
    store = EmbeddingStore(places.embeddings, weights.dim)
    model = Model(weights, store)
    saved = read_json(places.checkpoint, {}) or {}
    if saved:
        model.load(saved["model"])
    return model, store, saved


class Scribe:
    """Whisper on its own thread: the slow clock never holds the fast one."""

    def __init__(self, transcriber: Transcriber, write):
        self.transcriber = transcriber
        self.write = write
        self.jobs: queue.Queue = queue.Queue()
        threading.Thread(target=self._work, daemon=True).start()

    def submit(self, samples: np.ndarray, event: dict) -> None:
        self.jobs.put((samples, event))

    def drain(self) -> None:
        self.jobs.join()

    def _work(self) -> None:
        while True:
            samples, event = self.jobs.get()
            try:
                self.write({**event, "text": self.transcriber(samples), "written": utc_now()})
            finally:
                self.jobs.task_done()


class Ear:
    def __init__(self, places: Places, from_start: bool = False, live: bool = True, asr=None):
        places.ensure()
        self.places = places
        self.live = live
        self.lock = threading.Lock()
        self.sensor = Sensor(places.models)
        want = asr is not False and Transcriber.available(places.models)
        self.scribe = Scribe(Transcriber(places.models), self._write) if want else None
        self.corpus = Corpus(places.corpus if live else None, places.corpus_max_chunks)
        self.session = stamp()
        self.started = time.monotonic()
        self.origin = time.time()  # the wall time of stream second 0 of this session
        self.paused = False
        self.now: dict = {}
        self.echo = None
        self._load(from_start)
        self._retire_foreign_record()

    # -- memory ------------------------------------------------------------------
    def _load(self, from_start: bool) -> None:
        self.model, self.store, saved = load_memory(self.places)
        self.revision = saved.get("revision", 0)
        self.corrections_offset = saved.get("corrections_offset", 0)
        self.corrections_seen = saved.get("corrections_seen", 0)
        self.applied: list[list[int]] = saved.get("applied", [])  # [line, rows at the time]
        self.counters = saved.get("counters", {})
        self.tailer = Tailer(
            self.places.audio_dir,
            contract.chunk_bytes() if self.live else 0,
            from_start=from_start,
        )
        self.runs: dict[int, dict] = {}
        self.floor = {"power": 0.0, "chunks": 0}
        self.opened = False

    def _save(self) -> None:
        write_atomic(
            self.places.checkpoint,
            {
                "revision": self.revision,
                "corrections_offset": self.corrections_offset,
                "corrections_seen": self.corrections_seen,
                "applied": self.applied,
                "counters": self.counters,
                "weights": self.model.base.source,
                "model": self.model.to_dict(),
            },
        )

    def _retire_foreign_record(self) -> None:
        """A record whose first line is not on the port (v0, v1.0) is archived whole."""
        lines, _ = read_jsonl(self.places.events)
        if lines and lines[0].get("port") != PORT:
            archive(self.places.root / "archive", self.places.events)

    def _write(self, event: dict) -> None:
        with self.lock:
            append_jsonl(self.places.events, {"port": PORT, "session": self.session, **event})

    def _open(self, first_chunk: str | None) -> None:
        """The session line: the one place a reader learns where a session starts."""
        if not self.opened:
            self.opened = True
            self._write(
                {
                    "kind": "session",
                    "utc": utc_now(),
                    "weights": self.model.base.source,
                    "rows": len(self.store),
                    "first_chunk": first_chunk,
                }
            )

    # -- the loop -----------------------------------------------------------------
    def run(self, max_seconds: float | None = None, drain: bool = False, echo=None) -> None:
        self.echo = echo
        last_persist = float("-inf")
        while max_seconds is None or self.clock() < max_seconds:
            if self.places.reset.exists():
                self._reset()
            self.paused = self.places.pause.exists()
            worked = False
            if not self.paused:
                worked = self.tick()
            if worked or time.monotonic() - last_persist >= contract.PERSIST_EVERY_S:
                self.persist()
                last_persist = time.monotonic()
            if drain and not worked and not self.paused:
                self.finish()
                break
            time.sleep(contract.IDLE_SLEEP_S)
        self.persist()

    def tick(self) -> bool:
        worked = self._apply_corrections()
        for chunk in self.tailer.poll():
            worked = True
            self._open(chunk.name)
            self.corpus.keep(chunk)
            self._bump("chunks")
            pieces, runs = self.sensor.feed(chunk.samples)
            if not pieces and not runs and not self.sensor.speaking():
                self.floor["power"] += float(np.mean(np.square(chunk.samples)))
                self.floor["chunks"] += 1
            self._handle(pieces, runs)
        if worked and self.live:
            self.tailer.prune()
        return worked

    def finish(self) -> None:
        """End of a frozen stream: close the run in progress, wait for its text."""
        self._handle(*self.sensor.flush())
        if self.scribe:
            self.scribe.drain()
        self.persist()

    def clock(self) -> float:
        return time.monotonic() - self.started

    def stream_s(self, sample: int) -> float:
        return round(sample / contract.RATE, 3)

    def floor_dbfs(self) -> float | None:
        if not self.floor["chunks"]:
            return None
        return round(10 * math.log10(max(self.floor["power"] / self.floor["chunks"], 1e-24)), 1)

    # -- one observation --------------------------------------------------------------
    def _handle(self, pieces, runs) -> None:
        for piece in pieces:
            self._observe(piece)
        for run in runs:
            self._transcribe(run)

    def _observe(self, piece) -> None:
        run = self.runs.setdefault(piece.run, {"rows": [], "voiced": False, "last": None})
        vec = self.sensor.embed(piece.samples)
        if not np.all(np.isfinite(vec)):
            # Measured 2026-10-19: run tails of 6 to 22 ms, shorter than one fbank
            # frame, came out of CAM++ as NaN and wrote "p": NaN into the record. A
            # sensor that cannot see produces no observation, and says so in a counter.
            self._bump("unseeable")
            return
        row = self.store.append(vec)
        x = self.store(row)  # the stored float32 vector: what a replay will see, bit for bit
        follows = run["last"] if piece.full else None
        t = round(self.origin + piece.start / contract.RATE, 3)  # when it was said, not decided
        decision = self.model.observe(row, x, t, piece.full, follows)
        append_jsonl(
            self.places.observations, {"row": row, "t": t, "learn": piece.full, "follows": follows}
        )
        if piece.full:
            run["last"] = row
        run["rows"].append(row)
        run["voiced"] |= decision["kind"] in VOICED
        self._bump(decision["kind"])
        level = round(rms_dbfs(piece.samples), 1)
        floor = self.floor_dbfs()
        event = {
            "row": row,
            "t0": self.stream_s(piece.start),
            "t1": self.stream_s(piece.start + len(piece.samples)),
            "decided_at": self.stream_s(self.sensor.fed),
            "utc": utc_now(),
            "kind": decision["kind"],
            "speaker": decision["speaker"],
            "label": decision["label"],
            "p": round(decision["p"], 6),
            "margin": round(decision["margin"], 3),
            "runner_up": {
                "id": decision["runner_up"]["id"],
                "p": round(decision["runner_up"]["p"], 6),
            },
            "evidence": decision["evidence"],
            "learned": decision["learned"],
            "run": self.stream_s(piece.run),
            "rms_dbfs": level,
            "rel_db": None if floor is None else round(level - floor, 1),
        }
        self._write(event)
        self.now = event
        if self.echo:
            who = event["label"] or event["speaker"] or "-"
            self.echo(
                f"[{event['t0']:8.2f}-{event['t1']:8.2f}] {event['kind']:8s} {who:18s} "
                f"p={event['p']:.3f} margin={event['margin']:<9.1f} n={event['evidence']:<4d} "
                f"row={row}" + ("" if piece.full else "  (glimpse)")
            )

    def _transcribe(self, run) -> None:
        """Only a run in which a voice was recognised is worth Whisper's time.

        Not a threshold: the model's own decision. Measured in v1.0, whisper on
        music runs wrote `[MUSIC]`, `(music playing)` and invented Japanese.
        """
        info = self.runs.pop(run.start, None)
        if not info or not info["rows"]:
            return
        if not info["voiced"]:
            self._bump("runs_without_voice")
            return
        if self.scribe is None:
            return
        self._bump("runs_transcribed")
        self.scribe.submit(
            run.samples,
            {
                "kind": "text",
                "t0": self.stream_s(run.start),
                "t1": self.stream_s(run.start + len(run.samples)),
                "decided_at": self.stream_s(self.sensor.fed),
                "rows": info["rows"],
            },
        )

    # -- corrections ----------------------------------------------------------------
    def _apply_corrections(self) -> bool:
        lines, self.corrections_offset = read_jsonl(
            self.places.corrections, self.corrections_offset
        )
        for line in lines:
            self.corrections_seen += 1
            number = self.corrections_seen
            refused = None
            if line["kind"] == "undo":
                refused = self._undo(int(line["line"]))
            else:
                self.model.correct(line["kind"], [int(r) for r in line["rows"]], line.get("label"))
                self.applied.append([number, len(self.store)])
            self._bump("corrections")
            self._write(
                {
                    "kind": "correction",
                    "what": line["kind"],
                    "rows": line.get("rows", []),
                    "label": line.get("label"),
                    "line": number,
                    "undoes": line.get("line"),
                    "refused": refused,
                    "applied": utc_now(),
                }
            )
            if self.echo:
                self.echo(
                    f"[correction {number}] {line}" + (f"  REFUSED: {refused}" if refused else "")
                )
        return bool(lines)

    def _undo(self, target: int) -> str | None:
        """Forget one correction exactly: rebuild the model by replaying everything else."""
        if target not in [line for line, _ in self.applied]:
            return f"line {target} is not an applied correction"
        replayed = rebuild(self.places, [a for a in self.applied if a[0] != target])
        if replayed is None:
            return "this memory predates observations.jsonl: it cannot be replayed exactly"
        self.model = replayed
        self.applied = [a for a in self.applied if a[0] != target]
        return None

    # -- state ----------------------------------------------------------------------
    def _bump(self, key: str) -> None:
        self.counters[key] = self.counters.get(key, 0) + 1

    def persist(self) -> None:
        with self.lock:
            self.revision += 1
            self._save()
            write_atomic(self.places.state, self.state())

    def state(self) -> dict:
        model = self.model
        now = time.time()
        voices = sorted(
            (c for c in model.clusters.values() if c.kind != "candidate"),
            key=lambda c: (c.kind, -c.n),
        )
        candidates = sorted(
            (c for c in model.clusters.values() if c.kind == "candidate"),
            key=lambda c: c.first_seen,
        )
        return {
            "port": PORT,
            "revision": self.revision,
            "session": self.session,
            "utc": utc_now(),
            "uptime_s": round(self.clock(), 1),
            "stream_s": self.stream_s(self.sensor.fed),
            "paused": self.paused,
            "audio": {
                "dir": str(self.places.audio_dir),
                "chunks_seen": self.tailer.seen,
                "backlog_chunks": self.tailer.backlog(),
                "last_chunk": self.tailer.cursor,
                "floor_dbfs": self.floor_dbfs(),
            },
            "corpus": self.corpus.stats(),
            "now": self.now,
            "model": {
                k: round(v, 6) if isinstance(v, float) else v for k, v in model.summary().items()
            },
            "speakers": [
                {
                    "id": c.key,
                    "kind": c.kind,
                    "label": c.label,
                    "evidence": c.n,
                    "first_seen_unix": c.first_seen or None,
                    "silent_for_s": round(now - c.last_seen, 1) if c.last_seen else None,
                }
                for c in voices
            ],
            "candidates": [
                {
                    "id": c.key,
                    "evidence": c.n,
                    "rows": c.members,
                    "waiting_s": round(now - c.first_seen, 1),
                    "weight": round(model.survival(c), 6),
                }
                for c in candidates
            ],
            "absorbed": model.absorbed,
            "rows": len(self.store),
            "transcriptions_pending": self.scribe.jobs.qsize() if self.scribe else 0,
            "counters": self.counters,
        }

    def _reset(self) -> None:
        """Forget everything learned: archived, never deleted."""
        target = self.places.root / "archive"
        if self.scribe:
            self.scribe.drain()
        archive(
            target,
            self.places.state,
            self.places.events,
            self.places.checkpoint,
            self.places.embeddings,
            self.places.observations,
            self.places.corrections,
        )
        self.places.reset.unlink(missing_ok=True)
        self.sensor = Sensor(self.places.models)
        self._load(from_start=True)


def rebuild(places: Places, applied: list[list[int]]) -> Model | None:
    """The model again, from the base: every row replayed, every kept correction
    re-applied at the moment it was first applied. Deterministic, so exact."""
    weights = Weights.from_dict(read_json(places.weights))
    store = EmbeddingStore(places.embeddings, weights.dim)
    observations, _ = read_jsonl(places.observations)
    if len(observations) != len(store):
        return None
    lines, _ = read_jsonl(places.corrections)
    model = Model(weights, store)
    pending = sorted(applied, key=lambda a: (a[1], a[0]))
    for obs in observations:
        while pending and pending[0][1] <= obs["row"]:
            line = lines[pending.pop(0)[0] - 1]
            model.correct(line["kind"], [int(r) for r in line["rows"]], line.get("label"))
        model.observe(obs["row"], store(obs["row"]), obs["t"], obs["learn"], obs["follows"])
    for number, _ in pending:
        line = lines[number - 1]
        model.correct(line["kind"], [int(r) for r in line["rows"]], line.get("label"))
    return model


def correction(
    places: Places, kind: str, rows: list[int], label: str | None = None, **extra
) -> dict:
    """Write one correction. The running ear obeys it at its next tick."""
    line = {"kind": kind, "rows": rows, "utc": utc_now(), **extra}
    if label is not None:
        line["label"] = label
    append_jsonl(places.corrections, line)
    return line
