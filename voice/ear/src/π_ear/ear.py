"""The ear server: one loop, live, with its memory on disk. Who, and what.

    mic chunks ----> Echo (the machine taken out) --+
    machine chunks (what this machine plays) -------+-> a Channel each: Sensor (VAD,
                                                       CAM++, spectrum) -> Model (who:
                                                       voice, condition, doubt) ->
                                                       Source (person or device)
           -> Scribe thread (what: fast reading while it is said, slow revision
              in the silences, the reader's decision and its named doubt)
    record  <- decisions, turns, partial and revised texts, questions, corrections
    inbox   -> corrections and answers, from a person, a command or a voice

Two channels, one model of who: the microphone hears the room; the machine's own
output (``contract.OUTPUT``, ``make rec-out``) is what this machine plays, exactly,
before it reaches the air. It is read as a channel of its own (its voices are the
machine's, by construction, with the application that played them, ``device.py``)
and taken out of the microphone (``echo.py``), so the room is heard over it. With
the output off, or its recorder not running, the microphone alone is the ear, as
before.

Two places on disk, on purpose:

* ``root`` (``/tmp/π``, RAM): the live view. ``state.json``, ``events.jsonl``,
  the audio directory, ``PAUSE`` and ``RESET``. Losing it loses nothing learned;
* ``memory`` (``~/.cache/π/ear``): what the ear has learned. ``model.json`` (the
  checkpoint), ``embeddings.f32`` and ``spectra.f32`` (one row per window),
  ``observations.jsonl``, ``runs.jsonl``, ``taught.jsonl`` and
  ``corrections.jsonl``. It is read back at launch, and every learned part can
  be rebuilt by replay, which is what makes ``undo`` exact.

The two clocks: the decision loop never waits for a recogniser. The scribe has
its own thread, and its slow work waits for the room to be silent.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from π_ear import asr, contract, echo
from π_ear.device import Apps
from π_ear.io import (
    Corpus,
    EmbeddingStore,
    Tailer,
    append_jsonl,
    archive,
    chunk_index,
    read_json,
    read_jsonl,
    stamp,
    utc_now,
    write_atomic,
)
from π_ear.model import Model, Weights
from π_ear.port import PORT
from π_ear.scribe import Scribe, rebuild_reader
from π_ear.sense import BANDS, Sensor, Separator, rms_dbfs, spectrum
from π_ear.source import Source
from π_ear.words import Reader

VOICED = ("known", "new")
WHO = ("same", "different", "not_voice", "name", "person", "device")
MIC, MACHINE = "mic", "machine"


@dataclass
class Channel:
    """One stream the ear hears: the microphone (the room), or what the machine plays.
    Each has its sensor, its speech runs and its floor; both feed one model of who."""

    name: str
    tailer: Tailer
    sensor: Sensor
    corpus: Corpus
    runs: dict = field(default_factory=dict)
    floor: dict = field(default_factory=lambda: {"power": 0.0, "chunks": 0})
    first: int | None = None  # the second of its first chunk, on the shared clock
    last: int | None = None  # the second of its last chunk

    def floor_dbfs(self) -> float | None:
        if not self.floor["chunks"]:
            return None
        power = max(self.floor["power"] / self.floor["chunks"], 1e-24)
        return round(10 * math.log10(power), 1)


@dataclass(slots=True)
class Places:
    root: Path = contract.ROOT
    memory: Path = contract.MEMORY
    models: Path = contract.MODELS
    audio: Path | None = None
    output: Path | None = None  # the machine's own sound; None: root/out
    corpus: Path | None = contract.CORPUS
    corpus_max_chunks: int = contract.CORPUS_MAX_CHUNKS
    weights: Path = field(default_factory=lambda: base_weights_path(contract.ENCODER))

    def __post_init__(self):
        self.root, self.memory, self.models = Path(self.root), Path(self.memory), Path(self.models)

    @property
    def audio_dir(self) -> Path:
        return Path(self.audio) if self.audio else self.root / "audio"

    @property
    def output_dir(self) -> Path:
        return Path(self.output) if self.output else self.root / "out"

    apps = property(lambda self: self.root / "apps.jsonl")

    state = property(lambda self: self.root / "state.json")
    events = property(lambda self: self.root / "events.jsonl")
    pause = property(lambda self: self.root / "PAUSE")
    reset = property(lambda self: self.root / "RESET")
    snapshots = property(lambda self: self.root / "snapshots")
    checkpoint = property(lambda self: self.memory / "model.json")
    embeddings = property(lambda self: self.memory / "embeddings.f32")
    spectra = property(lambda self: self.memory / "spectra.f32")
    observations = property(lambda self: self.memory / "observations.jsonl")
    corrections = property(lambda self: self.memory / "corrections.jsonl")

    def ensure(self) -> None:
        for path in (self.root, self.audio_dir, self.output_dir, self.memory):
            path.mkdir(parents=True, exist_ok=True)

    def learned(self) -> tuple[Path, ...]:
        names = (
            "model.json",
            "embeddings.f32",
            "spectra.f32",
            "observations.jsonl",
            "runs.jsonl",
            "taught.jsonl",
            "corrections.jsonl",
        )
        return tuple(self.memory / n for n in names)


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


class Ear:
    def __init__(self, places: Places, from_start: bool = False, live: bool = True, asr=None):
        places.ensure()
        self.places = places
        self.live = live
        self.lock = threading.RLock()
        self.want_text = asr is not False
        self.output = contract.OUTPUT == "on"
        self.session = stamp()
        self.started = time.monotonic()
        self.origin = time.time()  # the wall time of stream second 0 of this session
        self.paused = False
        self.draining = False
        self.origin_second: int | None = None  # the chunk second stream time 0 is
        self.now: dict = {}
        self.echo = None
        self._load(from_start)
        # The memory's time never runs backwards: a replay faster than real time
        # would otherwise start its session before the last one ended, and "the
        # oldest voice keeps its name" would pick the wrong one.
        self.origin = max(self.origin, self.model.now)
        self._retire_foreign_record()
        self.started = time.monotonic()  # the ear's clock starts when it is ready to hear

    # -- memory ------------------------------------------------------------------
    def _load(self, from_start: bool) -> None:
        self.model, self.store, saved = load_memory(self.places)
        self.spectra = EmbeddingStore(self.places.spectra, BANDS)
        self.revision = saved.get("revision", 0)
        self.corrections_offset = saved.get("corrections_offset", 0)
        self.corrections_seen = saved.get("corrections_seen", 0)
        # [line, rows at the time, kind, {rows, label} as applied (an answer resolved)]
        self.applied: list[list] = saved.get("applied", [])
        self.undone: set[int] = set(saved.get("undone", []))
        # every correction the ear could not obey, and why: shown by `ctl status`, never silent
        self.refused: list[dict] = saved.get("refused", [])
        self.counters = saved.get("counters", {})
        self._align_spectra()
        self.seen: dict[str, str] = saved.get("seen", {})  # voice -> the session it was first heard
        self.channels = {MIC: self._channel(MIC, self.places.audio_dir, from_start)}
        if self.output:
            self.channels[MACHINE] = self._channel(MACHINE, self.places.output_dir, from_start)
        self.canceller = echo.Echo() if self.output and echo.available() else None
        self.apps = Apps(self.places.apps)
        self.waiting: list = []  # microphone chunks waiting for the machine's (echo.covers)
        self.machine_waiting: list = []  # the machine's chunks, merged in by their second
        self.share: list[
            tuple[int, np.ndarray]
        ] = []  # (first sample, the machine's share per frame)
        self.asked: set[str] = set()
        self.source = self._rebuild_source()
        self.reader = Reader(self.model.base.reader)
        if "reader" in saved:
            self.reader.load(saved["reader"])
        self.opened = False
        self.scribe = None
        if self.want_text:
            fast = asr.build(contract.FAST, self.places.models)
            if fast is not None:
                slow = asr.build(contract.SLOW, self.places.models)
                separator = (
                    Separator(self.places.models)
                    if Separator.available(self.places.models)
                    else None
                )
                self.scribe = Scribe(
                    self.reader,
                    fast,
                    slow,
                    separator,
                    self._write,
                    self.places.memory,
                    self.places.models,
                    source_of=self.source_of,
                    floor_s=self.sensor.min_speech_s,
                )

    def _save(self) -> None:
        write_atomic(
            self.places.checkpoint,
            {
                "revision": self.revision,
                "corrections_offset": self.corrections_offset,
                "corrections_seen": self.corrections_seen,
                "applied": self.applied,
                "undone": sorted(self.undone),
                "refused": self.refused,
                "counters": self.counters,
                "seen": self.seen,
                "weights": self.model.base.source,
                "model": self.model.to_dict(),
                "reader": self.reader.to_dict(),
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
                    "fast": contract.FAST if self.scribe else None,
                    "slow": contract.SLOW if self.scribe and self.scribe.slow else None,
                }
            )

    # -- the loop -----------------------------------------------------------------
    def run(self, max_seconds: float | None = None, drain: bool = False, echo=None) -> None:
        self.echo = echo
        if self.scribe:
            self.scribe.echo = echo
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

    def _channel(self, name: str, directory: Path, from_start: bool) -> Channel:
        corpus = None
        if self.live and self.places.corpus is not None:
            corpus = self.places.corpus if name == MIC else self.places.corpus / "out"
        return Channel(
            name,
            Tailer(directory, contract.chunk_bytes() if self.live else 0, from_start=from_start),
            Sensor(self.places.models),
            Corpus(corpus, self.places.corpus_max_chunks),
        )

    # the microphone channel, under the names the rest of the ear (and its tests) know
    mic = property(lambda self: self.channels[MIC])
    sensor = property(lambda self: self.mic.sensor)
    tailer = property(lambda self: self.mic.tailer)
    runs = property(lambda self: self.mic.runs)
    corpus = property(lambda self: self.mic.corpus)

    def tick(self) -> bool:
        worked = self._apply_corrections()
        machine = self.channels.get(MACHINE)
        for channel, queue in ((self.mic, self.waiting), (machine, self.machine_waiting)):
            if channel is None:
                continue
            for chunk in channel.tailer.poll():
                channel.corpus.keep(chunk)  # now: a waiting chunk's file may be pruned
                queue.append(chunk)
        # The two streams, merged in the order of their seconds: the machine's chunk k
        # goes first while it is not further ahead of the microphone than an offset
        # can reach, so that the canceller holds the machine's sound each microphone
        # chunk may hear, however the chunks arrive (one by one live, all at once in a
        # replay).
        reach = int(np.ceil(contract.ECHO_MOST_LAG_S / contract.CHUNK_S))
        while self.waiting or self.machine_waiting:
            head = chunk_index(self.waiting[0].name) if self.waiting else None
            if self.machine_waiting and (
                head is None or chunk_index(self.machine_waiting[0].name) <= head + reach
            ):
                chunk = self.machine_waiting.pop(0)
                if self.canceller is not None:
                    self.canceller.push_ref(chunk_index(chunk.name), chunk.samples)
                self._feed(machine, chunk, chunk.samples)
            elif self.waiting and self._mic_ready(self.waiting[0]):
                chunk = self.waiting.pop(0)
                samples = chunk.samples
                if self.canceller is not None and machine is not None and machine.tailer.seen:
                    samples, share = self.canceller.cancel(chunk_index(chunk.name), samples)
                    self._remember_share(chunk, share)
                self._feed(self.mic, chunk, samples)
            else:
                break
            worked = True
        if self.scribe:
            speaking = any(c.sensor.speaking() for c in self.channels.values())
            self.scribe.set_speaking(speaking, self._person_speaking())
        if worked and self.live:
            for channel in self.channels.values():
                channel.tailer.prune()
        return worked

    def _mic_ready(self, chunk) -> bool:
        """A microphone chunk is read once the machine's sound it may hear is there,
        or once the next microphone chunk exists (the machine's recorder is late, or
        stopped: never more than a chunk of delay), or when the stream is frozen."""
        machine = self.channels.get(MACHINE)
        if self.canceller is None or machine is None or not machine.tailer.seen:
            return True
        return (
            self.canceller.covers(chunk_index(chunk.name)) or len(self.waiting) > 1 or self.draining
        )

    def _feed(self, channel: Channel, chunk, samples: np.ndarray) -> None:
        self._open(chunk.name)
        self._bump("chunks" if channel.name == MIC else "machine_chunks")
        second = chunk_index(chunk.name)
        if channel.first is None:
            channel.first = second
            channel.sensor.offset = self._sample_of(second)
        elif channel.last is not None and second > channel.last + 1:
            # a hole in this stream (its recorder stopped, then resumed): its clock
            # keeps the hole, so its times stay the other channel's times
            gap = second - channel.last - 1
            silence = np.zeros(gap * contract.RATE * contract.CHUNK_S, dtype=np.float32)
            self._handle(channel, *channel.sensor.feed(silence))
        channel.last = second
        pieces, runs = channel.sensor.feed(samples)
        if not pieces and not runs and not channel.sensor.speaking():
            channel.floor["power"] += float(np.mean(np.square(samples)))
            channel.floor["chunks"] += 1
        self._handle(channel, pieces, runs)
        self._hear(channel)

    def _sample_of(self, second: int) -> int:
        """The stream sample a chunk second starts at: the first second heard is 0."""
        if self.origin_second is None:
            self.origin_second = second
        return (second - self.origin_second) * contract.RATE * contract.CHUNK_S

    def _remember_share(self, chunk, share: np.ndarray) -> None:
        self.share.append((self._sample_of(chunk_index(chunk.name)), share))
        del self.share[: max(0, len(self.share) - contract.SHARE_KEEP_CHUNKS)]

    def machine_share(self, a: int, b: int) -> float | None:
        """The machine's share of the microphone's energy over stream samples [a, b)."""
        values = []
        for first, share in self.share:
            lo = max(0, (a - first) // contract.ECHO_FRAME)
            hi = min(len(share), -(-(b - first) // contract.ECHO_FRAME))
            if hi > lo:
                values.append(share[lo:hi])
        return float(np.mean(np.concatenate(values))) if values else None

    def finish(self) -> None:
        """End of a frozen stream: close the runs in progress, wait for their text."""
        self.draining = True
        self.tick()
        for channel in self.channels.values():
            self._handle(channel, *channel.sensor.flush())
        if self.scribe:
            self.scribe.drain()
        self.persist()

    def clock(self) -> float:
        return time.monotonic() - self.started

    def stream_s(self, sample: int, channel: Channel | None = None) -> float:
        """Stream seconds on the shared clock: a channel's sample, past its offset."""
        offset = (channel or self.mic).sensor.offset
        return round((offset + sample) / contract.RATE, 3)

    def floor_dbfs(self, channel: Channel | None = None) -> float | None:
        return (channel or self.mic).floor_dbfs()

    def run_id(self, start: int, channel: Channel | None = None) -> str:
        channel = channel or self.mic
        mark = "" if channel.name == MIC else "m"
        return f"{self.session}:{mark}{self.stream_s(start, channel)}"

    # -- one observation --------------------------------------------------------------
    def _handle(self, channel: Channel, pieces, runs) -> None:
        for piece in pieces:
            self._observe(channel, piece)
        for run in runs:
            self._close(channel, run)

    def _observe(self, channel: Channel, piece) -> None:
        run = channel.runs.setdefault(
            piece.run, {"rows": [], "learned": [], "decisions": [], "last": None}
        )
        vec = channel.sensor.embed(piece.samples)
        if not np.all(np.isfinite(vec)):
            # Measured 2026-10-19: run tails of 6 to 22 ms, shorter than one fbank
            # frame, came out of CAM++ as NaN and wrote "p": NaN into the record. A
            # sensor that cannot see produces no observation, and says so in a counter.
            self._bump("unseeable")
            return
        row = self.store.append(vec)
        x = self.store(row)  # the stored float32 vector: what a replay will see, bit for bit
        self.spectra.append(spectrum(piece.samples))
        learn = piece.full and not piece.shadow
        follows = run["last"] if learn else None
        start = channel.sensor.offset + piece.start
        t = round(self.origin + start / contract.RATE, 3)  # when it was said
        # what the machine plays is heard in a place of its own: its channel is not the room's
        place = self.session if channel.name == MIC else f"{self.session}#{MACHINE}"
        decision = self.model.observe(row, x, t, learn, follows, place)
        append_jsonl(
            self.places.observations,
            {"row": row, "t": t, "learn": learn, "follows": follows, "place": place},
        )
        if learn:
            run["last"] = row
            run["learned"].append(row)
        run["rows"].append(row)
        self._bump(decision["kind"] if not piece.shadow else "shadow")
        level = round(rms_dbfs(piece.samples), 1)
        floor = channel.floor_dbfs()
        source, source_p = (
            self.source_of(decision["speaker"]) if decision["kind"] in VOICED else (None, None)
        )
        share = (
            self.machine_share(start, start + len(piece.samples)) if channel.name == MIC else None
        )
        apps = None
        if channel.name == MACHINE:
            first = self.origin_second + start // contract.RATE
            apps = self.apps.during(first, first + len(piece.samples) // contract.RATE)
        event = {
            "row": row,
            "t0": self.stream_s(piece.start, channel),
            "t1": self.stream_s(piece.start + len(piece.samples), channel),
            "decided_at": self.stream_s(channel.sensor.fed, channel),
            "utc": utc_now(),
            "channel": channel.name,
            "kind": decision["kind"],
            "speaker": decision["speaker"],
            "condition": decision["condition"],
            "label": decision["label"],
            "source": source,
            "source_p": None if source_p is None else round(source_p, 4),
            "p": round(decision["p"], 6),
            "margin": round(decision["margin"], 3),
            "trust": None if decision["trust"] is None else round(decision["trust"], 4),
            "runner_up": {
                "id": decision["runner_up"]["id"],
                "p": round(decision["runner_up"]["p"], 6),
            },
            "evidence": decision["evidence"],
            "learned": learn,
            "shadow": piece.shadow,
            "run": self.stream_s(piece.run, channel),
            "rms_dbfs": level,
            "rel_db": None if floor is None else round(level - floor, 1),
            "echo": None if share is None else round(share, 4),
            "apps": apps,
        }
        run["decisions"].append(event)
        self._write(event)
        self.now = event
        self._ask_name(event)
        if self.echo and not piece.shadow:
            who = event["label"] or event["speaker"] or "-"
            self.echo(
                f"[{event['t0']:8.2f}-{event['t1']:8.2f}] {channel.name[:3]} {event['kind']:8s} "
                f"{who:18s} p={event['p']:.3f} margin={event['margin']:<8.1f} "
                f"n={event['evidence']:<4d} row={row}"
                + ("" if piece.full else "  (glimpse)")
                + (f"  {source}" if source == "device" else "")
                + (f"  {', '.join(apps)}" if apps else "")
            )

    def _hear(self, channel: Channel) -> None:
        """The run in progress, handed to the scribe: it cuts and reads it as it grows."""
        if self.scribe is None:
            return
        current = channel.sensor.run_audio()
        if current is None:
            return
        start, samples = current
        decisions = channel.runs.get(start, {}).get("decisions", [])
        self.scribe.hear(
            self.run_id(start, channel), self.stream_s(start, channel), samples, decisions
        )

    def _person_speaking(self) -> bool:
        """Is the voice of the room's run in progress a person (the scribe keeps free for them)?"""
        current = self.mic.sensor.run_audio()
        if current is None:
            return False
        decisions = self.mic.runs.get(current[0], {}).get("decisions", [])
        voiced = [d["speaker"] for d in decisions if d["kind"] in VOICED]
        return bool(voiced) and self.source_of(voiced[-1])[0] == "person"

    def _close(self, channel: Channel, run) -> None:
        """A speech run ended: its turns, then its text, unless every window of it was
        a known sound. A voice not yet known is still read (its first sentence is
        a sentence); whether anything was said is the reader's to say, and it can
        say "I did not understand" (the empty reading, learned on ESC-50 sounds)."""
        info = channel.runs.pop(run.start, None)
        if not info or not info["rows"]:
            return
        decisions = info["decisions"]
        self._turns(channel, run, decisions)
        identifier = self.run_id(run.start, channel)
        if all(d["kind"] == "noise" for d in decisions):
            self._bump("runs_of_sounds")
            if self.scribe:
                self.scribe.retract(identifier)
            return
        if self.scribe is None:
            return
        self._bump("runs_transcribed")
        self.scribe.finish(identifier, self.stream_s(run.start, channel), run.samples, decisions)

    def _turns(self, channel: Channel, run, decisions: list[dict]) -> None:
        """Who spoke when, to the half window (Q4): each half window takes the voice of
        the more confident of the two windows (learned and shadow) that cover it."""
        half = contract.ENCODER_WINDOW_S / 2
        t0 = self.stream_s(run.start, channel)
        t1 = self.stream_s(run.start + len(run.samples), channel)
        slots = []
        start = t0
        while start < t1:
            end = min(start + half, t1)
            covering = [d for d in decisions if d["t0"] < end and start < d["t1"]]
            if covering:
                best = max(covering, key=lambda d: d["margin"])
                who = best["speaker"] if best["kind"] in VOICED else best["kind"]
                if slots and slots[-1][2] == who:
                    slots[-1][1] = round(end, 3)
                else:
                    slots.append([round(start, 3), round(end, 3), who])
            start = end
        self._write(
            {
                "kind": "turns",
                "run": self.run_id(run.start, channel),
                "channel": channel.name,
                "t0": t0,
                "t1": t1,
                "segments": slots,
            }
        )

    # -- where a voice comes from (Q3) ---------------------------------------------------
    def voice_rows(self, key: str) -> list[int]:
        voice = self.model.voices.get(key)
        if voice is None:
            return []
        return [r for c in voice.conditions for r in self.model.clusters[c].members]

    def _align_spectra(self) -> None:
        """One spectral row per embedding row, whatever the memory's history.

        Measured live (2026-10-20): a memory written before v2 had 1449 rows and no
        spectra; v2 appended spectra from row 1449 on, at index 0, so every source
        decision of that session read another window's spectrum (KπX came out
        a device), and the voices heard after the last spectral row had none
        (``source_p``: NaN). Both files only ever grow together since v2, so the rows
        without a spectrum are the oldest ones: they are put back in front, as unseen
        rows (NaN), kept out of every mean.
        """
        missing = len(self.store) - len(self.spectra)
        if missing > 0:
            unseen = [np.full(BANDS, np.nan)] * missing
            self.spectra.rows = unseen + self.spectra.rows
            self._bump("rows_without_spectrum")
        elif missing < 0:  # a crash between the two appends: the extra rows are no one's
            self.spectra.rows = self.spectra.rows[: len(self.store)]
        else:
            return
        np.asarray(self.spectra.rows, dtype=np.float32).reshape(-1, BANDS).tofile(
            self.places.spectra
        )

    def shape_of(self, rows) -> np.ndarray | None:
        """The mean spectral shape of these windows, over those that were seen; None if none was."""
        seen = [self.spectra(r) for r in rows if r < len(self.spectra)]
        seen = [x for x in seen if np.all(np.isfinite(x))]
        return np.mean(seen, axis=0) if seen else None

    def source_of(self, key: str | None) -> tuple[str | None, float | None]:
        """Person or device, for a voice: told by a correction, else decided from its shape.
        A voice whose windows were never seen spectrally is not decided: (None, None)."""
        voice = self.model.voices.get(key) if key else None
        if voice is None:
            return None, None
        if voice.source:
            return voice.source, 1.0
        shape = self.shape_of(self.voice_rows(key))
        return self.source.decide(shape) if shape is not None else (None, None)

    def _rebuild_source(self) -> Source:
        source = Source(self.model.base.spectra)
        lines, _ = read_jsonl(self.places.corrections)
        for number, line in enumerate(lines, 1):
            if line["kind"] in ("person", "device") and number not in self.undone:
                shape = self.shape_of(line.get("rows", []))
                if shape is not None:
                    source.add(line["kind"], shape)
        return source

    # -- questions (Q5) ----------------------------------------------------------------------
    def _ask_name(self, event: dict) -> None:
        """A person heard again in another session, still nameless: ask who it is, once."""
        key = event["speaker"]
        if event["kind"] not in VOICED or key is None:
            return
        first = self.seen.setdefault(key, self.session)
        voice = self.model.voices.get(key)
        if first == self.session or voice is None or voice.label or key in self.asked:
            return
        if self.source_of(key)[0] != "person":  # a voice on television is nobody to ask
            return
        self.asked.add(key)
        self._write(
            {
                "kind": "question",
                "id": f"{key}#name#{self.session}",
                "about": "name",
                "speaker": key,
                "rows": [event["row"]],
                "choices": [],
                "doubt": "unnamed",
            }
        )

    # -- corrections, answers, undo ---------------------------------------------------------
    def _apply_corrections(self) -> bool:
        lines, self.corrections_offset = read_jsonl(
            self.places.corrections, self.corrections_offset
        )
        for line in lines:
            self.corrections_seen += 1
            number = self.corrections_seen
            kind, refused = line["kind"], None
            rows = [int(r) for r in line.get("rows", [])]
            if kind == "answer":
                kind, rows, line, refused = self._answer(line)
            if refused:
                pass
            elif kind == "undo":
                refused = self._undo(int(line["line"]))
            elif kind == "text":
                refused = (
                    self.scribe.correct(rows, line["text"], number, line.get("piece"))
                    if self.scribe
                    else "no scribe"
                )
                if refused is None:
                    self.applied.append([number, len(self.store), "text"])
            elif kind in WHO:
                if kind in ("person", "device"):
                    shape = self.shape_of(rows)
                    if shape is not None:
                        self.source.add(kind, shape)
                self.model.correct(kind, rows, line.get("label"))
                self.applied.append(
                    [number, len(self.store), kind, {"rows": rows, "label": line.get("label")}]
                )
            else:
                refused = f"unknown kind {kind!r}"
            self._bump("corrections")
            if refused:
                self.refused.append(
                    {"line": number, "kind": kind, "why": refused, "utc": utc_now()}
                )
            self._write(
                {
                    "kind": "correction",
                    "what": kind,
                    "rows": rows,
                    "label": line.get("label"),
                    "text": line.get("text"),
                    "line": number,
                    "undoes": line.get("line") if kind == "undo" else None,
                    "answers": line.get("question"),
                    "refused": refused,
                    "applied": utc_now(),
                }
            )
            if self.echo:
                self.echo(
                    f"[correction {number}] {line}" + (f"  REFUSED: {refused}" if refused else "")
                )
        return bool(lines)

    def _answer(self, line: dict):
        """An answer to a question is the correction the question asked for."""
        question = line.get("question", "")
        asked = next(
            (
                e
                for e in read_jsonl(self.places.events)[0]
                if e.get("kind") == "question" and e.get("id") == question
            ),
            None,
        )
        if asked is None:
            return "answer", [], line, f"no question {question!r} in this record"
        if asked["about"] == "name":
            name = (line.get("label") or line.get("text") or "").strip()
            if not name:
                return "answer", [], line, "an answer to a name question names someone"
            return "name", asked["rows"], {**line, "label": name}, None
        text = line.get("text")
        if text is None and line.get("choice") is not None:
            choices = asked["choices"]
            if not 0 <= int(line["choice"]) < len(choices):
                return "answer", [], line, f"choice {line['choice']} of {len(choices)}"
            text = choices[int(line["choice"])]
        if text is None:  # "" is an answer: nothing was said
            return "answer", [], line, "an answer gives a text or a choice"
        if self.scribe:
            self.scribe.answered(asked.get("speaker"))
        return "text", asked["rows"], {**line, "text": text, "piece": asked.get("piece")}, None

    def _undo(self, target: int) -> str | None:
        """Forget one correction exactly: rebuild what it taught by replaying the rest."""
        entry = next((a for a in self.applied if a[0] == target), None)
        if entry is None:
            return f"line {target} is not an applied correction"
        kept = [a for a in self.applied if a[0] != target]
        self.undone.add(target)
        if entry[2] == "text":
            self.reader = rebuild_reader(
                self.places.memory, self.model.base.reader, self.undone, self.reader
            )
            if self.scribe:
                self.scribe.reader = self.reader
        else:
            replayed = rebuild(self.places, [a for a in kept if a[2] in WHO])
            if replayed is None:
                self.undone.discard(target)
                return "this memory predates observations.jsonl: it cannot be replayed exactly"
            self.model = replayed
            self.source = self._rebuild_source()
        self.applied = kept
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
        voices = sorted(model.voices.values(), key=lambda v: -len(self.voice_rows(v.key)))
        candidates = sorted(
            (c for c in model.clusters.values() if c.kind == "candidate"),
            key=lambda c: c.first_seen,
        )
        noise = [c for c in model.clusters.values() if c.kind == "noise"]

        def last_heard(v):
            seen = [model.clusters[c].last_seen for c in v.conditions]
            return round(now - max(seen), 1) if seen and max(seen) else None

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
            "output": self._output_state(),
            "corpus": self.corpus.stats(),
            "now": self.now,
            "model": {
                k: round(v, 6) if isinstance(v, float) else v for k, v in model.summary().items()
            },
            "speakers": [
                {
                    "id": v.key,
                    "label": v.label,
                    "source": dict(zip(("is", "p"), self.source_of(v.key), strict=True)),
                    "conditions": len(v.conditions),
                    "evidence": len(self.voice_rows(v.key)),
                    "first_seen_unix": v.first_seen or None,
                    "silent_for_s": last_heard(v),
                }
                for v in voices
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
            "noise": [{"id": c.key, "evidence": c.n} for c in noise],
            "reader": {
                "fast": contract.FAST if self.scribe else None,
                "slow": contract.SLOW if self.scribe and self.scribe.slow else None,
                "examples": len(self.reader.examples),
                "lexicon_words": sum(len(v) for v in self.reader.memory.lexicon.values()),
                "substitutions": len(self.reader.memory.substitutions),
                "pending": self.scribe.pending() if self.scribe else 0,
                **(self.scribe.stats() if self.scribe else {}),
            },
            "corrections": {
                "seen": self.corrections_seen,
                "applied": len(self.applied),
                "undone": sorted(self.undone),
                "refused": self.refused,
            },
            "absorbed": model.absorbed,
            "rows": len(self.store),
            "counters": self.counters,
        }

    def _output_state(self) -> dict:
        """What the ear knows of the machine's own sound, and why it may not be using it."""
        machine = self.channels.get(MACHINE)
        if machine is None:
            return {"on": False}
        state = {
            "on": True,
            "dir": str(self.places.output_dir),
            "chunks_seen": machine.tailer.seen,
            "last_chunk": machine.tailer.cursor,
            "waiting_mic_chunks": len(self.waiting),
        }
        if not machine.tailer.seen:
            state["degraded"] = "no chunk of the machine's sound yet: is `make rec-out` running?"
        if self.canceller is None:
            state["degraded"] = "no echo canceller (pyaec): the machine is read, not cancelled"
        else:
            state["echo"] = self.canceller.stats()
        return state

    def _reset(self) -> None:
        """Forget everything learned: archived, never deleted."""
        if self.scribe:
            self.scribe.drain()
        archive(
            self.places.root / "archive",
            self.places.state,
            self.places.events,
            *self.places.learned(),
        )
        self.places.reset.unlink(missing_ok=True)
        self._load(from_start=True)


def rebuild(places: Places, applied: list[list]) -> Model | None:
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

    def apply(entry):
        if len(entry) > 3:  # as it was applied: an answer is the correction it resolved to
            model.correct(entry[2], [int(r) for r in entry[3]["rows"]], entry[3]["label"])
            return
        line = lines[entry[0] - 1]
        model.correct(line["kind"], [int(r) for r in line["rows"]], line.get("label"))

    for obs in observations:
        while pending and pending[0][1] <= obs["row"]:
            apply(pending.pop(0))
        model.observe(
            obs["row"], store(obs["row"]), obs["t"], obs["learn"], obs["follows"], obs.get("place")
        )
    for entry in pending:
        apply(entry)
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
