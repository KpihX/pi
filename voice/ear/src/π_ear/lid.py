"""Which language a voice speaks, heard over all it said: a detection of its pooled speech.

Measured live (2026-10-21, the PSG - Barça recording, KπX's own French over the
television): Whisper's language detection on KπX's speech is right on 68 % of 4 s
stretches, 88 % of 8 s, and on all six 30 s stretches; on the studio's and the player's
voices, 97 % at 4 s, all at 8 s and 30 s (lab ``slid``, whisper-base). A language is a
property of a voice, and a voice says much more than one piece: its speech is pooled,
piece after piece, until it fills the detector's window, and detected then, once.

**The detector** is sherpa-onnx's spoken language identification on Whisper base (the
encoder and one step of the decoder: 0.6-0.85 s for 30 s on two threads, about 3 % of the
voice's own time). Whisper turbo's export does not serve it (its 128 mel bands; measured:
every call came back empty). It is one more detector of ``language.py`` (``POOLED``): its
reliability is measured on the base like the others' (``public.measure_pooled``), and
then on the stream, each of its detections judged against the language its voice turned
out to speak without it.

Only pieces of one voice are pooled: a piece shared between voices would teach each the
other's language.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

from π_ear import asr, contract

POOLED = "pooled"


@functools.cache
def _identifier(path: str, prefix: str):
    import sherpa_onnx

    root = Path(path)
    config = sherpa_onnx.SpokenLanguageIdentificationConfig(
        whisper=sherpa_onnx.SpokenLanguageIdentificationWhisperConfig(
            encoder=str(root / f"{prefix}-encoder.int8.onnx"),
            decoder=str(root / f"{prefix}-decoder.int8.onnx"),
        ),
        num_threads=contract.THREADS,
    )
    return sherpa_onnx.SpokenLanguageIdentification(config)


def available(models: Path) -> bool:
    if contract.LID == "none":
        return False
    return (models / contract.LID).exists()


def identifier(models: Path):
    """``samples -> language code`` ("" when it cannot say)."""
    prefix = contract.LID.removeprefix("sherpa-onnx-whisper-")
    lid = _identifier(str(models / contract.LID), prefix)

    def detect(samples: np.ndarray) -> str:
        stream = lid.create_stream()
        stream.accept_waveform(contract.RATE, np.asarray(samples, dtype=np.float32))
        return lid.compute(stream) or ""

    return detect


class Pooled:
    """Each voice's speech, pooled until it fills the detector's window."""

    def __init__(self, models: Path):
        self.detect = identifier(models)
        self.window = int(asr.WINDOW_S["whisper-base"] * contract.RATE)
        self.heard: dict[str, list[np.ndarray]] = {}

    def add(self, voice: str, samples: np.ndarray) -> tuple[str, float] | None:
        """A piece of ``voice`` alone; when its pool is full, (language, seconds)."""
        pool = self.heard.setdefault(voice, [])
        pool.append(np.asarray(samples, dtype=np.float32))
        if sum(len(p) for p in pool) < self.window:
            return None
        audio = np.concatenate(pool)
        self.heard[voice] = [audio[self.window :]] if len(audio) > self.window else []
        language = self.detect(audio[: self.window])
        return (language, self.window / contract.RATE) if language else None
