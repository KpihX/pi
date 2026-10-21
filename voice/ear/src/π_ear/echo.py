"""The machine's own sound, taken out of the microphone.

When the output is captured (``contract.OUTPUT``), the ear has two streams: the
microphone (the room) and what the machine itself plays, exactly, before it reaches
the air. What the machine plays is then not something to guess through the
microphone: it is subtracted.

Measured 2026-10-21 (``IDEAS.md``, v2.2) on a simulated laptop (its own speakers, a
small driver, the room, its own microphone), French sentences said over LibriSpeech
played by the machine:

* WebRTC AEC3 removes the most echo (20-25 dB) and destroys the person speaking over
  it: its residual suppressor, built for telephony, gates the near end whenever the
  echo dominates. Parakeet read 0.0 of the person's words where the raw microphone
  gave a few. An ambient ear must hear people *while* the machine plays: refused;
* Speex's MDF echo canceller, linear only (its residual suppressor off), removes
  about 16 dB and keeps the person: three quarters of the words where the person
  speaks about as loud as the machine plays, from almost none. Its step size is not
  set: MDF estimates it from the stream, double talk included (Valin, "On adjusting
  the learning rate in frequency domain echo cancellation with double-talk", 2007).
  Where the machine plays 10 dB over the person, the loudspeaker's nonlinearity
  leaves too much: that limit is measured live (``share``), never hidden.

The two recorders start at different moments, so the streams are offset by up to a
chunk, plus the output's own latency: more than a filter's tail covers. The offset
is estimated from the stream (GCC-PHAT, the lag at which the reference best
explains the microphone), every time a stretch of the reference was played, and the
most frequent estimate since the last discontinuity is used. Until there is one,
nothing is subtracted.

Per frame, ``share`` is the part of the microphone's energy the canceller attributed
to the machine: the evidence that what the microphone heard there was the machine
(the scene model reads it). Per chunk, ``left`` is what the machine still explains of
what remains (``left_of``): the canceller's own doubt, high where it failed.
"""

from __future__ import annotations

import ctypes
from collections import Counter

import numpy as np

from π_ear import contract

P16 = ctypes.POINTER(ctypes.c_int16)


def _speex():
    from pyaec import lib

    return lib


def available() -> bool:
    try:
        return _speex() is not None
    except ImportError:
        return False


def lag_of(mic: np.ndarray, ref: np.ndarray, most: int) -> int:
    """The lag in samples, in [-most, most], at which ``ref`` best explains ``mic``
    (GCC-PHAT): positive when the microphone hears the reference later."""
    n = 1 << int(np.ceil(np.log2(len(mic) + len(ref) + 2 * most)))
    cross = np.fft.rfft(mic, n) * np.conj(np.fft.rfft(ref, n))
    c = np.fft.irfft(cross / (np.abs(cross) + np.finfo(np.float32).tiny), n)
    lags = np.concatenate([c[: most + 1], c[n - most :]])  # 0..most, then -most..-1
    best = int(np.argmax(lags))
    return best if best <= most else best - len(lags)


def left_of(clean: np.ndarray, ref: np.ndarray) -> float:
    """How much of what is left the machine still explains: the magnitude-squared
    coherence between the cancelled microphone and the reference, averaged over the
    band speech lives in. Near 0 when the machine is gone (or played nothing), near 1
    when the canceller could not take it out (a driver pushed into its nonlinearity)."""
    from scipy.signal import coherence

    if not np.any(ref) or not np.any(clean):
        return 0.0
    freqs, c = coherence(clean, ref, fs=contract.RATE, nperseg=contract.ECHO_FRAME * 4)
    band = (freqs >= contract.SPEECH_BAND_HZ[0]) & (freqs <= contract.SPEECH_BAND_HZ[1])
    return float(np.mean(c[band]))


class Echo:
    """Cancels the reference out of the microphone, sample-aligned by stream index.

    Both streams are pushed with their index in samples on one clock (the chunk
    names); ``cancel`` gives the microphone back with the machine subtracted, and the
    machine's share per frame."""

    def __init__(self):
        self.lib = _speex()
        self.frame = contract.ECHO_FRAME
        self.tail = int(contract.ECHO_TAIL_S * contract.RATE)
        self.most = int(contract.ECHO_MOST_LAG_S * contract.RATE)
        self.window = int(contract.ECHO_ESTIMATE_S * contract.RATE)
        self.handle = None
        self.ref: dict[int, np.ndarray] = {}  # chunk index -> samples
        self.votes: Counter = Counter()
        self.lag: int | None = None
        self.pending_mic: list[np.ndarray] = []
        self.pending_ref: list[np.ndarray] = []
        self.last_mic: int | None = None
        self.estimates = 0
        self.left = 0.0  # of the last chunk cancelled (``left_of``)

    # -- the reference --------------------------------------------------------------
    def push_ref(self, index: int, samples: np.ndarray) -> None:
        self.ref[index] = np.asarray(samples, dtype=np.float32)
        for old in [k for k in self.ref if k < index - contract.ECHO_KEEP_CHUNKS]:
            del self.ref[old]

    def covers(self, index: int) -> bool:
        """Is the reference there for everything the microphone chunk ``index`` may
        hear (its own second and the lags around it)?"""
        reach = int(np.ceil(self.most / (contract.RATE * contract.CHUNK_S)))
        return all(k in self.ref for k in range(index - reach, index + reach + 1))

    def _reference(self, index: int, n: int, lag: int) -> np.ndarray:
        """The reference as the microphone chunk ``index`` hears it: shifted by the lag,
        zeros where nothing was captured (the machine played nothing, or no recorder)."""
        step = contract.RATE * contract.CHUNK_S
        start = index * step - lag
        out = np.zeros(n, dtype=np.float32)
        first, last = start // step, (start + n - 1) // step
        for k in range(first, last + 1):
            chunk = self.ref.get(k)
            if chunk is None:
                continue
            lo, hi = max(start, k * step), min(start + n, k * step + len(chunk))
            if hi > lo:
                out[lo - start : hi - start] = chunk[lo - k * step : hi - k * step]
        return out

    # -- the delay ------------------------------------------------------------------
    def _estimate(self, index: int, mic: np.ndarray) -> None:
        if self.last_mic is not None and index != self.last_mic + 1:
            self.votes.clear()  # a hole in the stream: the recorders may have restarted
            self.pending_mic, self.pending_ref = [], []
        self.last_mic = index
        ref = self._reference(index, len(mic), 0)
        self.pending_mic.append(mic)
        self.pending_ref.append(ref)
        if sum(len(x) for x in self.pending_mic) < self.window:
            return
        m, r = np.concatenate(self.pending_mic), np.concatenate(self.pending_ref)
        self.pending_mic, self.pending_ref = [], []
        if not np.any(r):
            return  # the machine played nothing: nothing to learn the delay from
        self.votes[lag_of(m, r, self.most)] += 1
        self.estimates += 1
        lag = self.votes.most_common(1)[0][0]
        if lag != self.lag:
            self.lag = lag  # a new alignment is a new filter
            self._drop()

    # -- cancelling -------------------------------------------------------------------
    def cancel(self, index: int, mic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(the microphone without the machine, the machine's share per frame)."""
        mic = np.asarray(mic, dtype=np.float32)
        self._estimate(index, mic)
        frames = len(mic) // self.frame
        if self.lag is None:
            return mic, np.zeros(frames)
        ref = self._reference(index, len(mic), self.lag)
        if not np.any(ref):
            return mic, np.zeros(frames)
        if self.handle is None:
            self.handle = self.lib.AecNew(self.frame, self.tail, contract.RATE, False)
        m = np.clip(np.round(mic * 32768), -32768, 32767).astype(np.int16)
        r = np.clip(np.round(ref * 32768), -32768, 32767).astype(np.int16)
        out = m.copy()
        for f in range(frames):
            a = np.ascontiguousarray(m[f * self.frame : (f + 1) * self.frame])
            b = np.ascontiguousarray(r[f * self.frame : (f + 1) * self.frame])
            o = np.zeros(self.frame, dtype=np.int16)
            self.lib.AecCancelEcho(
                self.handle,
                a.ctypes.data_as(P16),
                b.ctypes.data_as(P16),
                o.ctypes.data_as(P16),
                self.frame,
            )
            out[f * self.frame : (f + 1) * self.frame] = o
        clean = out.astype(np.float32) / 32768
        kept = clean[: frames * self.frame].reshape(frames, self.frame)
        heard = mic[: frames * self.frame].reshape(frames, self.frame)
        power = np.sum(heard * heard, axis=1)
        # the part of the microphone's energy that went with the machine (not the energy
        # of mic - clean: Speex's own DC notch and pre-emphasis make that difference
        # larger than the microphone itself, measured)
        share = 1 - np.divide(
            np.sum(kept * kept, axis=1), power, out=np.ones(frames), where=power > 0
        )
        self.left = left_of(clean, ref)
        return clean, np.clip(share, 0.0, 1.0)

    def stats(self) -> dict:
        return {
            "lag_s": None if self.lag is None else round(self.lag / contract.RATE, 4),
            "estimates": self.estimates,
            "agree": self.votes.most_common(1)[0][1] if self.votes else 0,
            "left": round(self.left, 3),
        }

    def _drop(self) -> None:
        if getattr(self, "handle", None) is not None:
            self.lib.AecDestroy(self.handle)
        self.handle = None

    def __del__(self):
        self._drop()
