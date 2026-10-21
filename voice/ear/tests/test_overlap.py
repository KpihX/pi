"""Two voices at once (``overlap.py``): heard by the segmenter, never learned as one voice."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_data

from π_ear import contract, overlap
from π_ear.public import load

pytestmark = [
    needs_data,
    pytest.mark.skipif(not overlap.available(contract.MODELS), reason="no segmenter"),
]
R = contract.RATE


def _sample():
    """The pyannote sample: two people, thirty seconds, who spoke when (RTTM)."""
    x = load(contract.DATA / "sample.wav")
    turns = []
    for line in (contract.DATA / "sample.rttm").read_text().splitlines():
        f = line.split()
        turns.append((float(f[3]), float(f[3]) + float(f[4]), f[7]))
    return x, turns


def _overlapped(turns, a: float, b: float) -> float:
    """Seconds of [a, b) in which two people of the RTTM speak at once."""
    total = 0.0
    for i, (a1, b1, w1) in enumerate(turns):
        for a2, b2, w2 in turns[i + 1 :]:
            if w1 != w2:
                total += max(0.0, min(b1, b2, b) - max(a1, a2, a))
    return total


def test_on_a_real_conversation_the_share_of_two_voices_follows_the_overlap():
    """Real speech (the pyannote sample), the windows the ear would cut, the segmenter's
    own word: the share of a window's speech it hears as two voices is several times
    larger where two people do speak at once (measured: 0.14 against 0.03 on average),
    and no clean window is more a mix than one voice. Its overlaps are short (at most a
    third of a window): none of them is either."""
    x, turns = _sample()
    sensor = overlap.Overlap(contract.MODELS)
    window = int(contract.ENCODER_WINDOW_S * R)
    shares, mixed = [], []
    for start in range(0, len(x) - window + 1, R):
        shares.append(sensor.of(lambda a, b: x[a:b], start, start + window))
        mixed.append(_overlapped(turns, start / R, (start + window) / R) > 0)
    shares, mixed = np.array(shares), np.array(mixed)
    assert shares[mixed].mean() > 3 * shares[~mixed].mean()
    assert not (shares[~mixed] > 1 - shares[~mixed]).any()


def test_a_window_is_scored_with_the_frames_already_heard():
    """Frames are kept between windows: a window inside what a pass heard needs no pass."""
    x, _ = _sample()
    sensor = overlap.Overlap(contract.MODELS)
    calls = []
    segment = sensor.segment

    def counting(audio, one=False):
        calls.append(len(audio))
        return segment(audio, one=one)

    sensor.segment = type("S", (), {"__call__": staticmethod(counting)})()
    sensor.segment.size, sensor.segment.shift = segment.size, segment.shift
    sensor.of(lambda a, b: x[a:b], 8 * R, 10 * R)
    sensor.of(lambda a, b: x[a:b], 7 * R, 9 * R)  # inside: no new pass
    assert len(calls) == 1
    sensor.of(lambda a, b: x[a:b], 9 * R, 11 * R)
    assert len(calls) == 1 + 1


def test_the_segmenter_s_word_is_weak_on_the_base_mixes_and_says_so():
    """Measured, not assumed: on mixes of the base's voices through simulated rooms, a
    window of two scores above a window of one more often than chance, and far from
    always (the reason the stream does not learn a prior on it, IDEAS step 4)."""
    from π_ear.public import measure_overlap

    measured = measure_overlap(contract.DATA, contract.MODELS, np.random.default_rng(5))
    assert 1 / 2 < measured["auc"] < 1
    assert measured["two"]["mean"] > measured["one"]["mean"]


def test_the_ear_decides_a_mix_and_does_not_learn_from_it(tmp_path):
    """Two people at once for a while, each alone before and after, through the whole
    ear: every window says its ``overlap``; a window more a mix than one voice is
    decided (who is the louder?) but not learned, and no window of one voice is one."""
    from π_ear.ear import Ear, Places
    from π_ear.io import read_jsonl
    from π_ear.public import chunk_name, synth

    fr = np.concatenate(list(synth(contract.MODELS, [0], FR, language="fr").values())[0])
    en = np.concatenate(list(synth(contract.MODELS, [40], EN).values())[0])
    both = min(len(fr), len(en)) // (1 + 1)
    gap = np.zeros(R, np.float32)
    x = np.concatenate([gap, fr[both:], gap, fr[:both] + en[:both], gap, en[both:], gap])
    x = (x / np.max(np.abs(x)) * 0.5).astype(np.float32)
    a = (R + len(fr) - both + R) / R  # the stretch of two voices
    span = (a, a + both / R)
    places = Places(root=tmp_path / "live", memory=tmp_path / "memory", audio=tmp_path / "a")
    places.audio_dir.mkdir(parents=True)
    pcm = (x * 32767).astype("<i2")
    for k, start in enumerate(range(0, len(pcm), R)):
        (places.audio_dir / chunk_name(k)).write_bytes(pcm[start : start + R].tobytes())
    ear = Ear(places, from_start=True, live=False, asr=False)
    ear.run(drain=True)
    decisions = [e for e in read_jsonl(places.events)[0] if "row" in e]
    assert decisions and all(e["overlap"] is not None for e in decisions)
    inside = [e for e in decisions if span[0] <= e["t0"] and e["t1"] <= span[1]]
    alone = [e for e in decisions if e["t1"] <= span[0] or span[1] <= e["t0"]]
    mixes = [e for e in decisions if e["overlap"] > 1 - e["overlap"]]
    assert inside and sum(e in mixes for e in inside) > len(inside) / (1 + 1)
    assert not any(e in mixes for e in alone)
    assert not any(e["learned"] for e in mixes)
    assert ear.counters["overlapped"] == len(mixes)


FR = [
    "je pense qu'on devrait commencer par le deuxième chapitre",
    "attends, tu as vu le message de Paul ce matin",
    "alors on garde le plan d'hier, avec les trois parties",
]
EN = [
    "we should start with the second chapter today",
    "wait, did you see the message from Paul this morning",
    "so we keep yesterday's plan with the three parts",
]
