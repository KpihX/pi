"""The physics of the voice, computed with numpy and nothing else.

No model, no download, no dependency: pitch, level, brightness and noisiness are
cheap enough to be computed on every window. They are the numbers that give the
event record its meaning, and they are also the placeholder identity vector until
a real speaker encoder is wired in (v1, CAM++).

Everything here returns plain numbers. No label, no verdict, no category: the
record carries measurements, and any interpretation of them happens downstream.
"""

from __future__ import annotations

import numpy as np


def load_raw(path, offset: int = 0) -> np.ndarray:
    """Read interleaved little-endian int16 samples from a raw PCM file.

    Returns floats in [-1, 1]. ``offset`` is in bytes, so the caller can follow a
    file that is still being written without re-reading it.
    """
    with open(path, "rb") as handle:
        handle.seek(offset)
        raw = handle.read()
    if not raw:
        return np.empty(0, dtype=np.float32)
    usable = len(raw) - (len(raw) % 2)
    samples = np.frombuffer(raw[:usable], dtype="<i2")
    return samples.astype(np.float32) / 32768.0


def _frames(x: np.ndarray, size: int, hop: int) -> np.ndarray:
    if x.size < size:
        return np.empty((0, size), dtype=np.float32)
    count = 1 + (x.size - size) // hop
    idx = np.arange(size)[None, :] + hop * np.arange(count)[:, None]
    return x[idx]


def rms_dbfs(x: np.ndarray) -> float:
    """Level in dBFS, the only honest absolute loudness number we can give."""
    if x.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(x))))
    return float(20.0 * np.log10(max(rms, 1e-12)))


def zero_crossing_rate(x: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(np.sign(x))) > 0))


def spectral_centroid(x: np.ndarray, rate: int) -> float:
    """Brightness in Hz: one of the cheapest cues that a voice is not another."""
    if x.size < 8:
        return 0.0
    windowed = x * np.hanning(x.size)
    spectrum = np.abs(np.fft.rfft(windowed))
    total = float(spectrum.sum())
    if total <= 0.0:
        return 0.0
    freqs = np.fft.rfftfreq(x.size, 1.0 / rate)
    return float((spectrum * freqs).sum() / total)


def f0_autocorrelation(x: np.ndarray, rate: int, fmin: float, fmax: float) -> tuple[float, float]:
    """Fundamental frequency of one frame, plus the periodicity peak.

    Plain autocorrelation by FFT. Returns ``(0.0, 0.0)`` when the frame carries no
    period at all, so the caller can decide what "voiced" means instead of having
    that decision hidden here.
    """
    if x.size < 16:
        return 0.0, 0.0
    centered = x - float(x.mean())
    if not np.any(centered):
        return 0.0, 0.0
    size = centered.size
    spectrum = np.fft.rfft(centered, 2 * size)
    autocorr = np.fft.irfft(spectrum * np.conj(spectrum))[:size]
    if autocorr[0] <= 0.0:
        return 0.0, 0.0
    autocorr = autocorr / autocorr[0]
    lo = max(1, int(rate / fmax))
    hi = min(size - 1, int(rate / fmin))
    if hi <= lo:
        return 0.0, 0.0
    segment = autocorr[lo:hi]
    offset = int(np.argmax(segment))
    lag = lo + offset
    # A maximum sitting on the FIRST lag of the search range is not a pitch, it is
    # the absence of one: with nothing periodic the autocorrelation decays
    # monotonically and its maximum lands on the shortest lag allowed, which would
    # report the highest frequency the search range can express (457.1 Hz at
    # 16 kHz with f0_max 450). Measured on 2026-10-18 on this machine: that
    # artefact, on room and keyboard noise, was enough to register a phantom
    # speaker. A genuine period always peaks strictly inside the range.
    if offset == 0:
        return 0.0, 0.0
    return float(rate / lag), float(autocorr[lag])


def analyse_window(x: np.ndarray, cfg) -> dict:
    """Everything we know about one analysis window, as numbers.

    The window is what the inventory sees and what the event record keeps. The
    aggregate is deliberately robust: median pitch over voiced frames, median
    level, mean brightness.
    """
    frame = int(cfg.rate * cfg.frame_ms / 1000)
    hop = max(1, int(cfg.rate * cfg.frame_hop_ms / 1000))
    frames = _frames(x, frame, hop)

    levels: list[float] = []
    pitches: list[float] = []
    brights: list[float] = []
    crossings: list[float] = []
    voiced = 0

    for one in frames:
        level = rms_dbfs(one)
        levels.append(level)
        brights.append(spectral_centroid(one, cfg.rate))
        crossings.append(zero_crossing_rate(one))
        pitch, peak = f0_autocorrelation(one, cfg.rate, cfg.f0_min, cfg.f0_max)
        if level > cfg.level_floor_db and peak >= cfg.voiced_min_peak:
            voiced += 1
            if pitch > 0.0:
                pitches.append(pitch)

    count = max(1, frames.shape[0])
    return {
        "rms_dbfs": round(float(np.median(levels)) if levels else -120.0, 2),
        "f0_hz": round(float(np.median(pitches)), 1) if pitches else 0.0,
        "centroid_hz": round(float(np.mean(brights)) if brights else 0.0, 1),
        "zcr": round(float(np.mean(crossings)) if crossings else 0.0, 4),
        "voiced_ratio": round(voiced / count, 3),
        "frames": int(frames.shape[0]),
    }


def identity_vector(features: dict) -> np.ndarray:
    """The provisional identity vector (v0 only, replaced by CAM++ in v1).

    Four normalised numbers: pitch, brightness, level, noisiness. Honest about
    what it is: this is NOT biometric identity, a voice on a film will collide
    with a voice in the room. It exists so that the *mechanism* (candidates,
    evidence, registration, moving threshold) can be watched working tonight,
    with no download and no model.
    """
    vec = np.array(
        [
            features.get("f0_hz", 0.0) / 200.0,
            features.get("centroid_hz", 0.0) / 2000.0,
            features.get("rms_dbfs", -60.0) / 60.0,
            features.get("zcr", 0.0),
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        return vec
    return vec / norm


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.dot(a, b))


def percentile(value: float, history: list[float] | np.ndarray) -> float:
    """Rank of ``value`` inside a history, in percent. 0 means lowest ever heard.

    This is the whole answer to "was that voice high or low": the percentile is
    relative to that speaker's own observed distribution, never to an absolute
    frequency grid. A bass at 120 Hz can sit in the top decile for him.
    """
    values = np.asarray(history, dtype=np.float64)
    if values.size == 0:
        return 50.0
    return float(100.0 * np.mean(values <= value))
