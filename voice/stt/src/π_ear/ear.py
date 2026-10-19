"""The ear server: one loop, live, with its memory on disk.

    chunks -> Sensor (VAD + CAM++) -> Model (who) -> record + state
                     \\-> Transcriber (what was said, per finished run)
    corrections.jsonl -> Model.correct  (obeyed at the next tick)

Two places on disk, on purpose:

* ``root`` (``/tmp/π``, RAM): the live view. ``state.json``, ``events.jsonl``,
  the audio directory, ``PAUSE`` and ``RESET``. Losing it loses nothing learned;
* ``memory`` (``~/.cache/π/ear``): what the ear has learned. ``model.json`` (the
  checkpoint: clusters, links, learned statistics, the record cursor) and
  ``embeddings.f32`` (one row per observation). It is read back at launch, so
  a voice keeps its key across restarts, and a correction can name any row.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

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
from π_ear.sense import Sensor, Transcriber, rms_dbfs


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
    corrections = property(lambda self: self.memory / "corrections.jsonl")

    def ensure(self) -> None:
        for path in (self.root, self.audio_dir, self.memory):
            path.mkdir(parents=True, exist_ok=True)


def base_weights_path(encoder: str) -> Path:
    return contract.WEIGHTS / f"base-{Path(encoder).stem}.json"


class Ear:
    def __init__(self, places: Places, from_start: bool = False, live: bool = True, asr=None):
        places.ensure()
        self.places = places
        self.live = live
        self.sensor = Sensor(places.models)
        self.transcribe = (
            Transcriber(places.models)
            if asr is not False and Transcriber.available(places.models)
            else None
        )
        self.corpus = Corpus(places.corpus if live else None, places.corpus_max_chunks)
        self.session = stamp()
        self.started = time.monotonic()
        self.paused = False
        self.now: dict = {}
        self._load(from_start)

    # -- memory ------------------------------------------------------------------
    def _load(self, from_start: bool) -> None:
        saved = read_json(self.places.checkpoint, {}) or {}
        weights = Weights.from_dict(read_json(self.places.weights))
        self.store = EmbeddingStore(self.places.embeddings, weights.dim)
        self.model = Model(weights, self.store)
        if saved:
            self.model.load(saved["model"])
        self.revision = saved.get("revision", 0)
        self.corrections_offset = saved.get("corrections_offset", 0)
        self.counters = saved.get("counters", {})
        self.tailer = Tailer(
            self.places.audio_dir,
            contract.chunk_bytes() if self.live else 0,
            from_start=from_start,
        )
        self.rows_of_run: dict[int, list[int]] = {}

    def _save(self) -> None:
        write_atomic(
            self.places.checkpoint,
            {
                "revision": self.revision,
                "corrections_offset": self.corrections_offset,
                "counters": self.counters,
                "weights": self.model.base.source,
                "model": self.model.to_dict(),
            },
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
            self.corpus.keep(chunk)
            self._bump("chunks")
            self._handle(*self.sensor.feed(chunk.samples))
        if worked and self.live:
            self.tailer.prune()
        return worked

    def finish(self) -> None:
        """End of a frozen stream: close the run in progress."""
        self._handle(*self.sensor.flush())
        self.persist()

    def clock(self) -> float:
        return time.monotonic() - self.started

    def stream_s(self, sample: int) -> float:
        return round(sample / contract.RATE, 3)

    # -- one observation --------------------------------------------------------------
    def _handle(self, pieces, runs) -> None:
        for piece in pieces:
            self._observe(piece)
        for run in runs:
            self._transcribe(run)

    def _observe(self, piece) -> None:
        x = self.sensor.embed(piece.samples)
        row = self.store.append(x)
        decision = self.model.observe(row, x, round(time.time(), 3), piece.full)
        self.rows_of_run.setdefault(piece.run, []).append(row)
        self._bump(decision["kind"])
        event = {
            "row": row,
            "session": self.session,
            "t0": self.stream_s(piece.start),
            "t1": self.stream_s(piece.start + len(piece.samples)),
            "decided_at": self.stream_s(self.sensor.fed),
            "utc": utc_now(),
            "kind": decision["kind"],
            "speaker": decision["speaker"],
            "label": decision["label"],
            "p": round(decision["p"], 6),
            "runner_up": {
                "id": decision["runner_up"]["id"],
                "p": round(decision["runner_up"]["p"], 6),
            },
            "evidence": decision["evidence"],
            "learned": decision["learned"],
            "run": self.stream_s(piece.run),
            "rms_dbfs": round(rms_dbfs(piece.samples), 1),
        }
        append_jsonl(self.places.events, event)
        self.now = event
        if self.echo:
            who = event["label"] or event["speaker"] or "-"
            self.echo(
                f"[{event['t0']:8.2f}-{event['t1']:8.2f}] {event['kind']:8s} {who:18s} "
                f"p={event['p']:.3f} n={event['evidence']:<4d} row={row}"
                + ("" if piece.full else "  (glimpse)")
            )

    def _transcribe(self, run) -> None:
        rows = self.rows_of_run.pop(run.start, [])
        if self.transcribe is None or not rows:
            return
        text = self.transcribe(run.samples)
        event = {
            "kind": "text",
            "session": self.session,
            "t0": self.stream_s(run.start),
            "t1": self.stream_s(run.start + len(run.samples)),
            "decided_at": self.stream_s(self.sensor.fed),
            "rows": rows,
            "text": text,
        }
        append_jsonl(self.places.events, event)
        if self.echo and text:
            self.echo(f"[{event['t0']:8.2f}-{event['t1']:8.2f}] text     {text}")

    # -- corrections ----------------------------------------------------------------
    def _apply_corrections(self) -> bool:
        lines, self.corrections_offset = read_jsonl(
            self.places.corrections, self.corrections_offset
        )
        for line in lines:
            self.model.correct(line["kind"], [int(r) for r in line["rows"]], line.get("label"))
            self._bump("corrections")
            append_jsonl(
                self.places.events,
                {**line, "kind": "correction", "what": line["kind"], "applied": utc_now()},
            )
            if self.echo:
                self.echo(f"[correction] {line}")
        return bool(lines)

    # -- state ----------------------------------------------------------------------
    def _bump(self, key: str) -> None:
        self.counters[key] = self.counters.get(key, 0) + 1

    def persist(self) -> None:
        self.revision += 1
        self._save()
        write_atomic(self.places.state, self.state())

    def state(self) -> dict:
        model = self.model
        voices = sorted(
            (c for c in model.clusters.values() if c.kind != "candidate"),
            key=lambda c: (c.kind, -c.n),
        )
        return {
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
                    "silent_for_s": round(time.time() - c.last_seen, 1) if c.last_seen else None,
                }
                for c in voices
            ],
            "absorbed": model.absorbed,
            "rows": len(self.store),
            "counters": self.counters,
        }

    def _reset(self) -> None:
        """Forget everything learned: archived, never deleted."""
        target = self.places.root / "archive"
        archive(
            target,
            self.places.state,
            self.places.events,
            self.places.checkpoint,
            self.places.embeddings,
            self.places.corrections,
        )
        self.places.reset.unlink(missing_ok=True)
        self.sensor = Sensor(self.places.models)
        self._load(from_start=True)


def correction(places: Places, kind: str, rows: list[int], label: str | None = None) -> dict:
    """Write one correction. The running ear obeys it at its next tick."""
    line = {"kind": kind, "rows": rows, "utc": utc_now()}
    if label is not None:
        line["label"] = label
    append_jsonl(places.corrections, line)
    return line
