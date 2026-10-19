"""The sensors: frozen, published, measured. They perceive; they never decide who.

* **Silero VAD** says where speech is, at its published operating point.
* **CAM++** turns ``ENCODER_WINDOW_S`` seconds of speech into a direction.
* **Whisper** (optional) writes down what a finished speech run said.

The cut is live: a window is emitted as soon as the speech run in progress has
grown by one encoder window, without waiting for the run to end. What is left
when a run ends, shorter than a window, is emitted too, as a *glimpse*: the
model decides who it is, but never learns from it, because the encoder is not
reliable below its window (``contract.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sherpa_onnx

from π_ear import contract


@dataclass(slots=True)
class Piece:
    start: int  # absolute sample index in the session stream
    samples: np.ndarray
    run: int  # the start of the speech run it belongs to
    index: int  # its rank inside that run
    full: bool  # a whole encoder window, as opposed to a glimpse


@dataclass(slots=True)
class Run:
    start: int
    samples: np.ndarray


class Sensor:
    def __init__(self, models: Path, encoder: str = contract.ENCODER):
        vad = sherpa_onnx.VadModelConfig()
        vad.silero_vad.model = str(models / contract.VAD_MODEL)
        vad.sample_rate = contract.RATE
        self.vad = sherpa_onnx.VoiceActivityDetector(vad, buffer_size_in_seconds=120)
        self.block = vad.silero_vad.window_size
        self.encoder = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(models / encoder), num_threads=contract.THREADS
            )
        )
        self.dim = self.encoder.dim
        self.window = int(contract.ENCODER_WINDOW_S * contract.RATE)
        self.pending = np.empty(0, dtype=np.float32)
        self.fed = 0
        self.history = np.empty(0, dtype=np.float32)  # recent audio, for slicing pieces
        self.history_start = 0
        self.run_start: int | None = None
        self.used = 0
        self.index = 0

    # -- audio in ---------------------------------------------------------------
    def feed(self, samples: np.ndarray) -> tuple[list[Piece], list[Run]]:
        """Push audio; return the windows ready now and the runs that just ended."""
        data = np.concatenate([self.pending, samples.astype(np.float32)])
        usable = len(data) - len(data) % self.block
        self.pending = data[usable:]
        self._remember(data[:usable])
        pieces: list[Piece] = []
        runs: list[Run] = []
        for offset in range(0, usable, self.block):
            self.vad.accept_waveform(data[offset : offset + self.block])
            self.fed += self.block
            self._collect(pieces, runs)
        return pieces, runs

    def _remember(self, data: np.ndarray) -> None:
        self.history = np.concatenate([self.history, data])
        keep = int(contract.RATE * 60) + self.window  # a resource bound, not a judgement
        drop = max(0, len(self.history) - keep)
        self.history = self.history[drop:]
        self.history_start += drop

    def _slice(self, start: int, end: int) -> np.ndarray:
        return self.history[start - self.history_start : end - self.history_start]

    def _collect(self, pieces: list[Piece], runs: list[Run]) -> None:
        while not self.vad.empty():
            segment = self.vad.front
            start, length = segment.start, len(segment.samples)
            if self.run_start != start:
                self.run_start, self.used, self.index = start, 0, 0
            self._cut(pieces, start + length, final=True)
            runs.append(Run(start, np.asarray(segment.samples, dtype=np.float32)))
            self.vad.pop()
            self.run_start = None
        if self.vad.is_speech_detected():
            start = self.vad.current_segment.start
            if start >= 0:
                if self.run_start != start:
                    self.run_start, self.used, self.index = start, 0, 0
                self._cut(pieces, self.fed, final=False)

    def _cut(self, pieces: list[Piece], end: int, final: bool) -> None:
        start = self.run_start
        while end - (start + self.used) >= self.window:
            a = start + self.used
            pieces.append(Piece(a, self._slice(a, a + self.window), start, self.index, True))
            self.used += self.window
            self.index += 1
        if final and end > start + self.used:
            a = start + self.used
            pieces.append(Piece(a, self._slice(a, end), start, self.index, False))

    def speaking(self) -> bool:
        return bool(self.vad.is_speech_detected())

    def flush(self) -> tuple[list[Piece], list[Run]]:
        """End of a frozen stream: whatever speech is in progress is finished."""
        pieces: list[Piece] = []
        runs: list[Run] = []
        self.vad.flush()
        self._collect(pieces, runs)
        return pieces, runs

    # -- the direction of a voice -------------------------------------------------
    def embed(self, samples: np.ndarray) -> np.ndarray:
        stream = self.encoder.create_stream()
        stream.accept_waveform(contract.RATE, samples)
        stream.input_finished()
        vec = np.asarray(self.encoder.compute(stream), dtype=np.float64)
        return vec / np.linalg.norm(vec)


class Transcriber:
    """Whisper on a finished speech run. Optional: absent model, absent text."""

    def __init__(self, models: Path):
        root = models / contract.WHISPER_DIR
        size = contract.WHISPER
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=str(root / f"{size}-encoder.int8.onnx"),
            decoder=str(root / f"{size}-decoder.int8.onnx"),
            tokens=str(root / f"{size}-tokens.txt"),
            language=contract.WHISPER_LANGUAGE,
            num_threads=contract.THREADS,
        )

    @staticmethod
    def available(models: Path) -> bool:
        return (models / contract.WHISPER_DIR / f"{contract.WHISPER}-encoder.int8.onnx").exists()

    def __call__(self, samples: np.ndarray) -> str:
        stream = self.recognizer.create_stream()
        stream.accept_waveform(contract.RATE, samples)
        self.recognizer.decode_stream(stream)
        return stream.result.text.strip()


def rms_dbfs(samples: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
    return 20.0 * float(np.log10(max(rms, 1e-12)))


def speech_probability(models: Path, samples: np.ndarray) -> np.ndarray:
    """Silero's own probability, one per 32 ms frame: what the VAD decision hides.

    The live path only sees Silero's yes/no at its published operating point. A
    probe needs the number underneath, to tell "not speech" from "speech the
    detector missed" (RETURNS.md, the 81-second hole).
    """
    import onnxruntime

    session = onnxruntime.InferenceSession(
        str(models / contract.VAD_MODEL), providers=["CPUExecutionProvider"]
    )
    shape = session.get_inputs()[1].shape
    h = np.zeros(shape, dtype=np.float32)
    c = np.zeros(shape, dtype=np.float32)
    block = session.get_inputs()[0].shape[1]
    out = []
    for start in range(0, len(samples) - block + 1, block):
        x = samples[start : start + block].astype(np.float32)[None, :]
        prob, h, c = session.run(None, {"x": x, "h": h, "c": c})
        out.append(float(prob[0][0]))
    return np.asarray(out)
