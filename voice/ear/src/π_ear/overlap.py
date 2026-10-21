"""Two voices at once: a sensor that hears overlap, so that a mix is never learned as a voice.

CAM++ turns a window into the direction of *a* voice. Two people talking over each
other give a direction between theirs: decided, it is the louder one at best; learned,
it drags a voice towards the other, or makes a third voice that is nobody (step 4 of the
ambient brief: at a table, in a meeting, people talk over each other all the time).

**The sensor** is pyannote's segmentation-3.0 (Plaquet & Bredin 2023), as exported by
sherpa-onnx: ten seconds of audio in, for each 17 ms frame the posterior of seven
classes, the powerset of up to three local speakers taken one or two at a time:
nobody, {1}, {2}, {3}, {1,2}, {1,3}, {2,3}. P(overlap) of a frame is the mass of the
last three. It is a published, frozen sensor; it says nothing of who.

**Its numbers are its own.** Its posteriors carry the prior of the real meetings it was
trained on (AMI, DIHARD and others), which is closer to a table of people than anything
the ear could draw. A calibration on the base was tried and refused by its own
measurement (``public.measure_overlap``, IDEAS step 4): on mixes of the base's voices
the score separates a window of two voices from a window of one with an AUC of about
0.7 only, too weak for the stream to learn how often its windows are mixes (that
estimate ran away to "every window is a mix" on two scenes of three, and nothing was
learned). Weak evidence is used as what it is: the segmenter's own word.

**What it decides, nothing more**: a window is a mix when more of its speech is two voices
at once than one voice alone, by the segmenter's expected frames of each. A mix is decided
(who is the louder?) but never learned from, like a shadow; its decision says ``overlap``,
that share. A first rule, the window's most probably overlapped frame alone, learned fewer
mixes on scenes where people cut in (purity +2 to +5 points), and split a voice in three
in a room where nobody did: the segmenter hears FSDD's telephone-band voices as two voices
on single frames, and a call's voice is telephone-band too. The rule kept is the safe one:
it never learns a window that is more a mix than a voice, and lets short overlaps (a third
of a window, the most the real sample has) be learned as the louder voice.

The cost, measured: one ten-second segmentation in about 90 ms on one thread (int8).
It runs once per second of speech per channel at most: the frames already computed are
kept, a window needs only the frames it covers.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

from π_ear import contract

ONE = slice(1, 4)  # the powerset classes {1}, {2}, {3}
TWO = slice(4, 7)  # the powerset classes {1,2}, {1,3}, {2,3}


@functools.cache
def _session(path: str):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = contract.THREADS
    options.inter_op_num_threads = 1
    return ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])


def available(models: Path) -> bool:
    if contract.SEGMENTATION == "none":
        return False
    return (models / contract.SEGMENTATION / "model.int8.onnx").exists()


class Segmenter:
    """pyannote segmentation-3.0: audio -> P(two voices), P(one voice) per frame, raw."""

    def __init__(self, models: Path):
        self.session = _session(str(models / contract.SEGMENTATION / "model.int8.onnx"))
        meta = self.session.get_modelmeta().custom_metadata_map
        self.size = int(meta["window_size"])  # samples in, ten seconds
        self.shift = int(meta["receptive_field_shift"])  # samples between frames
        self.field = int(meta["receptive_field_size"])  # samples one frame sees

    def __call__(self, audio: np.ndarray, one: bool = False):
        """(centre sample of each frame, raw P(two voices at once) of each frame) for
        ``audio``, at most ``size`` samples (shorter is padded with silence after it);
        with ``one``, P(one voice alone) of each frame too."""
        x = np.zeros(self.size, dtype=np.float32)
        x[: len(audio)] = audio[: self.size]
        y = np.exp(self.session.run(None, {"x": x[None, None]})[0][0])
        p = y[:, TWO].sum(axis=1)
        centres = np.arange(len(p)) * self.shift + self.field // (1 + 1)
        keep = centres < len(audio)
        if one:
            return centres[keep], p[keep], y[:, ONE].sum(axis=1)[keep]
        return centres[keep], p[keep]


def log_odds(p: np.ndarray) -> np.ndarray:
    eps = np.finfo(np.float32).eps
    q = np.clip(p, eps, 1 - eps)
    return np.log(q) - np.log1p(-q)


class Overlap:
    """One channel's overlap, frame by frame, computed as the speech comes."""

    def __init__(self, models: Path):
        self.segment = Segmenter(models)
        # frame (absolute) -> [P(two voices), P(one voice)], per pass
        self.frames: dict[int, list[tuple[float, float]]] = {}
        self.done = None  # the last sample a pass ended at

    def _pass(self, audio_of, end: int, since: int) -> None:
        start = max(end - self.segment.size, since, 0)
        audio = audio_of(start, end)
        if not len(audio):
            return
        centres, two, one = self.segment(audio, one=True)
        for c, p, q in zip(centres, two, one, strict=True):
            frame = int((start + c) // self.segment.shift)
            self.frames.setdefault(frame, []).append((float(p), float(q)))
        self.done = end

    def of(self, audio_of, start: int, end: int, since: int = 0) -> float | None:
        """How much of the window's speech is two voices at once: the segmenter's expected
        frames of two voices over its expected frames of speech, one voice or two (a frame
        seen by several passes: their mean). More than half: the window is more a mix
        than one voice. The first build took the most probably overlapped frame alone: in
        a room where nobody spoke over anybody, twelve windows of FSDD's telephone-band
        voices came out mixes, one voice was split in three (``tests/test_real.py``).
        ``audio_of(a, b)``: the channel's samples; ``since``: where the speech run
        it is in began, the context the segmenter is given (measured: ten seconds of
        context reaching back over a stretch the echo canceller had worked on made one
        clean voice after it a mix, 0.62 to 0.90, where the raw microphone gave 0.03).
        None: no frame of it heard."""
        step = self.segment.shift
        if self.done is None or self.done < end:  # its last frames are not heard yet
            self._pass(audio_of, end, since)
        mine = [
            np.mean(self.frames[f], axis=0)
            for f in range(start // step, end // step + 1)
            if f in self.frames
        ]
        oldest = end // step - (self.segment.size // step) * (1 + 1)
        for f in [f for f in self.frames if f < oldest]:  # a resource bound, not a judgement
            del self.frames[f]
        if not mine:
            return None
        two, one = (float(v) for v in np.sum(mine, axis=0))
        return two / (two + one) if two + one else None
