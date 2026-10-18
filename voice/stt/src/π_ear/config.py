"""Every knob in one place.

Nothing in this package hardcodes a path, a device or a threshold where it is
used: the defaults live here, they can be overridden by an environment variable
(prefix ``PI_EAR_``), and finally by a CLI flag. The runtime root defaults to
``/tmp/π`` on purpose: the state, the audio and the events are session
artefacts, never documents.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ENV_PREFIX = "PI_EAR_"


def _env(name: str, default: str) -> str:
    return os.environ.get(ENV_PREFIX + name, default)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(ENV_PREFIX + name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(slots=True)
class Config:
    # --- runtime root: everything the POC produces lives under it ------------
    root: Path = field(default_factory=lambda: Path(_env("ROOT", "/tmp/π")))
    # Where the chunks are read from. None means ``<root>/audio``, the live
    # recorder. Pointing it at the corpus is what makes a session replayable: the
    # same ear, the same parameters, a frozen directory.
    audio_dir_override: Path | None = field(
        default_factory=lambda: (
            Path(os.path.expanduser(_env("AUDIO_DIR", ""))) if _env("AUDIO_DIR", "") else None
        )
    )

    # --- capture contract ----------------------------------------------------
    # 16 kHz mono s16le is the format every model downstream expects
    # (CAM++, YAMNet, sherpa-onnx). Verified on this machine: plughw:0,6 accepts
    # exactly this configuration natively.
    device: str = field(default_factory=lambda: _env("DEVICE", "plughw:0,6"))
    rate: int = field(default_factory=lambda: int(_env("RATE", "16000")))
    channels: int = field(default_factory=lambda: int(_env("CHANNELS", "1")))
    chunk_s: int = field(default_factory=lambda: int(_env("CHUNK_S", "1")))

    # --- analysis: a sliding window, one decision per hop --------------------
    window_s: float = field(default_factory=lambda: float(_env("WINDOW_S", "3.0")))
    hop_s: float = field(default_factory=lambda: float(_env("HOP_S", "1.0")))
    frame_ms: int = field(default_factory=lambda: int(_env("FRAME_MS", "40")))
    frame_hop_ms: int = field(default_factory=lambda: int(_env("FRAME_HOP_MS", "20")))
    f0_min: float = field(default_factory=lambda: float(_env("F0_MIN", "60")))
    f0_max: float = field(default_factory=lambda: float(_env("F0_MAX", "450")))
    level_floor_db: float = field(default_factory=lambda: float(_env("LEVEL_FLOOR_DB", "-55")))
    # Fraction of frames in a window that must look voiced before the window is
    # treated as speech at all. Measured 2026-10-18 on this real microphone: a
    # silent room still yields 0.08 to 0.15 (noise slipping through the
    # periodicity test), a normal voice yields 0.4 to 0.7. 0.25 sits between the
    # two. It is the first knob to move for whispered speech (E9).
    voiced_floor: float = field(default_factory=lambda: float(_env("VOICED_FLOOR", "0.25")))
    voiced_min_peak: float = field(default_factory=lambda: float(_env("VOICED_MIN_PEAK", "0.30")))

    # --- inventory (the state that learns) ----------------------------------
    # A speaker is registered only once k_evidence mutually compatible
    # observations have accumulated: one noisy chunk cannot create a phantom
    # speaker.
    #
    # The admission criterion is MEASURED, never counted. A window belongs to a
    # known voice when two things hold at once: it is at least tau_floor close to
    # the closest one, AND it is closer to that one than to the runner-up by
    # margin_required, where margin_required is z_margin times the spread of the
    # margins this sensor has itself been producing. The spread calibrates the ear,
    # so there is nothing to hand-tune and nothing that can run away.
    #
    # What this replaced, and why (measured 2026-10-19, 379 seconds of real audio
    # with two real voices in it): tau = min(tau_base + lam * N_registered,
    # tau_max) made the admission threshold a function of the inventory size, so
    # every registration made the next one easier, which registered someone else.
    # 56 identities for two voices, with a median best-minus-runner-up margin of
    # 0.0005. The counter is gone; the measurement stayed.
    k_evidence: int = field(default_factory=lambda: int(_env("K_EVIDENCE", "3")))
    tau_floor: float = field(default_factory=lambda: float(_env("TAU_FLOOR", "0.5")))
    z_margin: float = field(default_factory=lambda: float(_env("Z_MARGIN", "3.0")))
    margin_floor: float = field(default_factory=lambda: float(_env("MARGIN_FLOOR", "0.0")))
    margin_samples: int = field(default_factory=lambda: int(_env("MARGIN_SAMPLES", "512")))
    tau_candidate: float = field(default_factory=lambda: float(_env("TAU_CANDIDATE", "0.970")))
    max_candidates: int = field(default_factory=lambda: int(_env("MAX_CANDIDATES", "16")))
    f0_history: int = field(default_factory=lambda: int(_env("F0_HISTORY", "200")))
    max_speakers: int = field(default_factory=lambda: int(_env("MAX_SPEAKERS", "16")))

    # --- durable corpus: RAM keeps nothing, the home keeps a bounded window ---
    # Measured 2026-10-19: with no corpus and the audio deleted as it was consumed,
    # six minutes of real test (a hum, then a film) survived only as numbers. The
    # audio was gone, so that exact session could not be replayed against CAM++,
    # which is exactly what the next step needs. The corpus is a durable, bounded
    # copy of the chunks that were actually ingested.
    corpus: bool = field(default_factory=lambda: _env_bool("CORPUS", True))
    corpus_dir: Path = field(
        default_factory=lambda: Path(os.path.expanduser(_env("CORPUS_DIR", "~/.cache/π/corpus")))
    )
    corpus_max_chunks: int = field(default_factory=lambda: int(_env("CORPUS_MAX_CHUNKS", "3600")))

    # --- persistence ---------------------------------------------------------
    persist_every_s: float = field(default_factory=lambda: float(_env("PERSIST_EVERY_S", "1.0")))
    idle_sleep_s: float = field(default_factory=lambda: float(_env("IDLE_SLEEP_S", "0.10")))
    gap_factor: float = field(default_factory=lambda: float(_env("GAP_FACTOR", "2.5")))

    # --- normalisation -------------------------------------------------------
    def __post_init__(self) -> None:
        """Coerce ``root`` to a Path, whatever handed it over.

        A real failure produced this method: every unit test built its Config from
        a ``Path``, while the CLI hands a ``str`` over, so ``self.root / "audio"``
        raised ``TypeError: unsupported operand type(s) for /: 'str' and 'str'``
        and ``make run`` crashed. The dataclass is the single place that can
        guarantee the type, so it does, instead of every caller remembering to.
        """
        if not isinstance(self.root, Path):
            self.root = Path(self.root)
        if not isinstance(self.corpus_dir, Path):
            self.corpus_dir = Path(self.corpus_dir)
        if self.audio_dir_override is not None and not isinstance(self.audio_dir_override, Path):
            self.audio_dir_override = Path(self.audio_dir_override)

    # --- derived paths -------------------------------------------------------
    @property
    def expected_chunk_bytes(self) -> int:
        """The exact size of a complete chunk, in bytes.

        This is what "the recorder has moved on to the next chunk" means, and it is
        what the tailer uses to refuse to read a file that is still being written.
        s16le: two bytes per sample, ``channels`` interleaved.
        """
        return int(self.chunk_s * self.rate * self.channels * 2)

    @property
    def audio_dir(self) -> Path:
        """Where the chunks are read: the live recorder, or a frozen corpus."""
        if self.audio_dir_override is not None:
            return self.audio_dir_override
        return self.root / "audio"

    @property
    def state_file(self) -> Path:
        return self.root / "state.json"

    @property
    def events_file(self) -> Path:
        return self.root / "events.jsonl"

    @property
    def pause_file(self) -> Path:
        return self.root / "PAUSE"

    @property
    def reset_file(self) -> Path:
        return self.root / "RESET"

    @property
    def snapshots_dir(self) -> Path:
        return self.root / "snapshots"

    @property
    def archive_dir(self) -> Path:
        return self.root / "archive"

    def ensure(self) -> None:
        """Create the runtime root. Idempotent, called by every entry point."""
        for path in (self.root, self.audio_dir, self.snapshots_dir, self.archive_dir):
            path.mkdir(parents=True, exist_ok=True)
        if self.corpus:
            self.corpus_dir.mkdir(parents=True, exist_ok=True)

    def as_dict(self) -> dict:
        """The subset of the configuration that belongs in the state file."""
        return {
            "root": str(self.root),
            "device": self.device,
            "rate": self.rate,
            "channels": self.channels,
            "chunk_s": self.chunk_s,
            "window_s": self.window_s,
            "hop_s": self.hop_s,
            "k_evidence": self.k_evidence,
            "tau_floor": self.tau_floor,
            "z_margin": self.z_margin,
            "margin_floor": self.margin_floor,
            "tau_candidate": self.tau_candidate,
            "level_floor_db": self.level_floor_db,
            "corpus": self.corpus,
            "corpus_dir": str(self.corpus_dir),
            "corpus_max_chunks": self.corpus_max_chunks,
        }
