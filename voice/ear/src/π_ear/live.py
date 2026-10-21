"""Live recordings: real sessions of the ear, kept on the operator's machine, replayed.

A live recording is what ``make rec`` wrote during a real session (the corpus of 1 s raw
chunks) next to what the ear produced then (``report/``: events, state, memory). It is
never committed: it holds broadcast audio and the voices of the people in the room. The
repository holds only its **manifest** (``tests/live/<name>.json``): where the corpus
sits on the stream, and the segments worth hearing again, each with what makes it hard.

Layout under ``contract.LIVE`` (``PI_EAR_LIVE``, default ``~/.cache/π/data/live``)::

    <name>/corpus/**/*.raw      the chunks, any depth
    <name>/report/runtime/      events.jsonl, state.json of the session
    <name>/report/memory/       the memory as it was at the end
    <name>/listen-*.jsonl       readings by other systems, for comparison (optional)

No single reading is taken as the truth (the live test showed every recogniser fail
somewhere: Parakeet translated studio talk, Whisper invented languages on short
phrases, both dropped words). Comparisons are against several systems, and what they
say is a measurement, not a target.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from π_ear import contract
from π_ear.lexicon import align, tokens

MANIFESTS = Path(__file__).parent.parent.parent / "tests" / "live"


@dataclass
class Segment:
    name: str
    start: float  # corpus seconds
    end: float
    what: str
    hard: str
    expect: dict = field(default_factory=dict)


@dataclass
class Recording:
    name: str
    root: Path
    offset: float  # stream seconds of corpus second 0
    segments: dict[str, Segment]
    chunks: list[Path]

    @classmethod
    def open(cls, name: str, live: Path | None = None) -> Recording | None:
        manifest = MANIFESTS / f"{name}.json"
        root = (live or contract.LIVE) / name
        if not manifest.exists() or not root.exists():
            return None
        raw = json.loads(manifest.read_text())
        chunks = sorted(root.glob("corpus/**/*.raw"), key=lambda p: p.name)
        if len(chunks) < raw["chunks"]:
            return None
        segments = {s["name"]: Segment(**s) for s in raw["segments"]}
        return cls(name, root, raw["offset_s"], segments, chunks)

    def audio(self, start: float, end: float) -> np.ndarray:
        first, last = int(start), int(np.ceil(end))
        parts = [
            np.frombuffer(c.read_bytes(), "<i2").astype(np.float32) / 32768
            for c in self.chunks[first:last]
        ]
        x = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        lo = int(round((start - first) * contract.RATE))
        return x[lo : lo + int(round((end - start) * contract.RATE))]

    def write(self, segment: str | Segment, directory: Path) -> Path:
        """The chunks of a segment, copied as they were recorded (bit for bit)."""
        s = self.segments[segment] if isinstance(segment, str) else segment
        directory.mkdir(parents=True, exist_ok=True)
        for chunk in self.chunks[int(s.start) : int(np.ceil(s.end))]:
            shutil.copy2(chunk, directory / chunk.name)
        return directory

    def memory(self, directory: Path) -> Path:
        """A copy of the memory the ear had at the end of the session.

        It already heard every segment once: the voices of a replay are known from the
        very windows replayed, and *who* is easier than it was live. Of *what* it holds
        little: no correction, and a lexicon of 230 words confirmed by both recognisers
        over two hours (v2.0 confirmed a word only when two readings agreed whole)."""
        source = self.root / "report" / "memory"
        if directory.exists():
            shutil.rmtree(directory)
        shutil.copytree(source, directory)
        return directory

    def events(self) -> list[dict]:
        path = self.root / "report" / "runtime" / "events.jsonl"
        return [json.loads(line) for line in path.open()] if path.exists() else []

    def listening(self, system: str) -> list[dict]:
        """Another system's reading of the corpus, block by block (``t`` in corpus seconds)."""
        path = self.root / f"listen-{system}.jsonl"
        return [json.loads(line) for line in path.open()] if path.exists() else []


def recall(reference: list[str], written: list[str]) -> float:
    """The share of a reference's words found, in order, in what was written."""
    if not reference:
        return 1.0
    return sum(op == "=" for op, _, _ in align(reference, written)) / len(reference)


def words_between(blocks: list[dict], start: float, end: float) -> list[str]:
    """The words of the blocks that start inside [start, end): a block straddling the
    end counts whole, one straddling the start not at all (the same for every run
    compared, so a bias of the reference, not of a run)."""
    return [w for b in blocks if start <= b["t"] < end for w in tokens(b["text"])]
