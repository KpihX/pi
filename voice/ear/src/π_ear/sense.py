"""The sensors: frozen, published, measured. They perceive; they never decide who.

* **Silero VAD** says where speech is, at its published operating point.
* **CAM++** turns ``ENCODER_WINDOW_S`` seconds of speech into a direction.
* **the spectrum** of a window (24 mel bands, its shape only): what the room and
  the loudspeaker do to a voice, for ``source.py``;
* **Spleeter** separates a voice from music, for the runs that mix both.

The recognisers of *what* was said are in ``asr.py``.

The cut is live: a window is emitted as soon as the speech run in progress has
grown by one encoder window, without waiting for the run to end. A second series
of windows, shifted by half a window (*shadows*), is decided too but never
learned from: two windows cover every second, so a change of speaker is placed
to the second, not to the window (Q4). What is left
when a run ends, shorter than a window, is emitted too, as a *glimpse*: the
model decides who it is, but never learns from it, because the encoder is not
reliable below its window (``contract.py``).
"""

from __future__ import annotations

import functools
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
    shadow: bool = False  # shifted by half a window: decided, never learned from


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
        self.min_speech_s = float(vad.silero_vad.min_speech_duration)  # published default
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
        self.shadow_used = 0
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
        if self.run_start is not None:
            # the run in progress is kept whole: the scribe reads it from its start (live,
            # studio runs of 70 s were sliced from a negative index past 60 s)
            keep = max(keep, self.history_start + len(self.history) - self.run_start)
        drop = max(0, len(self.history) - keep)
        self.history = self.history[drop:]
        self.history_start += drop

    def _slice(self, start: int, end: int) -> np.ndarray:
        start = max(start, self.history_start)
        return self.history[start - self.history_start : max(end - self.history_start, 0)]

    def _collect(self, pieces: list[Piece], runs: list[Run]) -> None:
        while not self.vad.empty():
            segment = self.vad.front
            start, length = segment.start, len(segment.samples)
            if self.run_start != start:
                self._begin(start)
            self._cut(pieces, start + length, final=True)
            runs.append(Run(start, np.asarray(segment.samples, dtype=np.float32)))
            self.vad.pop()
            self.run_start = None
        if self.vad.is_speech_detected():
            start = self.vad.current_segment.start
            if start >= 0:
                if self.run_start != start:
                    self._begin(start)
                self._cut(pieces, self.fed, final=False)

    def _begin(self, start: int) -> None:
        self.run_start, self.used, self.index = start, 0, 0
        self.shadow_used = self.window // 2

    def _cut(self, pieces: list[Piece], end: int, final: bool) -> None:
        start = self.run_start
        while end - (start + self.used) >= self.window:
            a = start + self.used
            pieces.append(Piece(a, self._slice(a, a + self.window), start, self.index, True))
            self.used += self.window
            self.index += 1
        while end - (start + self.shadow_used) >= self.window:
            a = start + self.shadow_used
            pieces.append(Piece(a, self._slice(a, a + self.window), start, -1, True, shadow=True))
            self.shadow_used += self.window
        if final and end > start + self.used:
            a = start + self.used
            pieces.append(Piece(a, self._slice(a, end), start, self.index, False))

    def run_audio(self) -> tuple[int, np.ndarray] | None:
        """The speech run in progress, from its start to now: what the fast reader hears."""
        if self.run_start is None or not self.vad.is_speech_detected():
            return None
        return self.run_start, self._slice(self.run_start, self.fed)

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


def spectrum(samples: np.ndarray) -> np.ndarray:
    """The shape of a window's long-term spectrum: 24 mel bands, in dB, level removed.

    Not a voice print: what a loudspeaker and a room do to any voice (a small
    driver loses the bass, a cone resonates, a room adds its colour). The level is
    removed so that near and far, loud and soft, have the same shape.
    """
    frame, hop = 400, 160  # 25 ms frames, 10 ms hop: the standard analysis grid
    if samples.size < frame:
        return np.zeros(BANDS)
    frames = np.lib.stride_tricks.sliding_window_view(samples, frame)[::hop] * np.hanning(frame)
    power = np.mean(np.abs(np.fft.rfft(frames, n=512)) ** 2, axis=0)
    bands = np.log10(_MEL @ power + 1e-12) * 10
    return bands - bands.mean()


BANDS = 24


def _mel_filters(bands: int = BANDS, n_fft: int = 512, rate: int = contract.RATE) -> np.ndarray:
    def mel(f):
        return 2595 * np.log10(1 + f / 700)

    edges = 700 * (10 ** (np.linspace(mel(60), mel(rate / 2 * 0.95), bands + 2) / 2595) - 1)
    bins = np.fft.rfftfreq(n_fft, 1 / rate)
    out = np.zeros((bands, bins.size))
    for b in range(bands):
        lo, mid, hi = edges[b : b + 3]
        out[b] = np.clip(np.minimum((bins - lo) / (mid - lo), (hi - bins) / (hi - mid)), 0, None)
    return out


_MEL = _mel_filters()


class Separator:
    """Spleeter, two stems: the voice out of the music (Q6). Optional."""

    def __init__(self, models: Path):
        root = models / contract.SEPARATOR
        self.model = sherpa_onnx.OfflineSourceSeparation(
            sherpa_onnx.OfflineSourceSeparationConfig(
                model=sherpa_onnx.OfflineSourceSeparationModelConfig(
                    spleeter=sherpa_onnx.OfflineSourceSeparationSpleeterModelConfig(
                        vocals=str(root / "vocals.int8.onnx"),
                        accompaniment=str(root / "accompaniment.int8.onnx"),
                    ),
                    num_threads=contract.THREADS,
                )
            )
        )

    @staticmethod
    def available(models: Path) -> bool:
        return (
            contract.SEPARATOR != "none"
            and (models / contract.SEPARATOR / "vocals.int8.onnx").exists()
        )

    def __call__(self, samples: np.ndarray) -> np.ndarray:
        from scipy.signal import resample_poly

        out = self.model.process(sample_rate=contract.RATE, samples=np.stack([samples, samples]))
        voice = np.asarray(out.stems[0].data, dtype=np.float32).mean(axis=0)
        back = resample_poly(voice, contract.RATE, out.sample_rate).astype(np.float32)
        return back[: len(samples)]


def rms_dbfs(samples: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
    return 20.0 * float(np.log10(max(rms, 1e-12)))


@functools.cache
def _silero(path: str):
    import onnxruntime

    return onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])


class Silero:
    """Silero's own probability, one per frame (512 samples, 32 ms), as a stream: the
    state is kept between calls, so a run is followed as it grows, at no extra cost."""

    def __init__(self, models: Path):
        self.session = _silero(str(models / contract.VAD_MODEL))
        self.block = self.session.get_inputs()[0].shape[1]
        shape = self.session.get_inputs()[1].shape
        self.h = np.zeros(shape, dtype=np.float32)
        self.c = np.zeros(shape, dtype=np.float32)
        self.pending = np.zeros(0, dtype=np.float32)
        self.probs: list[float] = []

    def feed(self, samples: np.ndarray) -> np.ndarray:
        """The probabilities of the frames these samples complete."""
        audio = np.concatenate([self.pending, np.asarray(samples, dtype=np.float32)])
        done = len(audio) - len(audio) % self.block
        out = []
        for start in range(0, done, self.block):
            x = audio[start : start + self.block][None, :]
            prob, self.h, self.c = self.session.run(None, {"x": x, "h": self.h, "c": self.c})
            out.append(float(prob[0][0]))
        self.pending = audio[done:]
        self.probs += out
        return np.asarray(out)


def speech_probability(models: Path, samples: np.ndarray) -> np.ndarray:
    """Silero's own probability, one per 32 ms frame: what the VAD decision hides.

    The live path only sees Silero's yes/no at its published operating point. A
    probe needs the number underneath, to tell "not speech" from "speech the
    detector missed" (RETURNS.md, the 81-second hole).
    """
    return Silero(models).feed(samples)
