"""The machine's own sound, taken out of the microphone (``echo.py``).

A simulated laptop: what the machine plays (LibriSpeech, clean), heard through its
own small speakers and the room by its own microphone, with a person speaking French
over it. The two recorders start at different moments, so the streams are offset by
a fraction of a chunk either way, plus the output's latency.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_data

from π_ear import asr, contract, echo
from π_ear.lexicon import align, tokens
from π_ear.public import load, loudspeaker, place, synth

R = contract.RATE
SAID = [
    "je voudrais qu'on reprenne le chapitre trois demain matin",
    "envoie le compte rendu à toute l'équipe avant midi",
    "on se retrouve devant la bibliothèque à quatorze heures",
]

pytestmark = pytest.mark.skipif(not echo.available(), reason="no Speex echo canceller")


def test_the_lag_is_found_either_way():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(4 * R).astype(np.float32)
    for lag in (0, 1234, -3210, R // 2, -R // 3):
        mic = np.roll(ref, lag) * 0.3 + rng.standard_normal(len(ref)).astype(np.float32) * 0.1
        assert echo.lag_of(mic, ref, R) == lag


def _scene(seed: int, offset_s: float):
    rng = np.random.default_rng(seed)
    tests = contract.MODELS / "sherpa-onnx-whisper-tiny" / "test_wavs"
    far = np.concatenate([load(tests / f) for f in ("0.wav", "1.wav", "8k.wav")] * 2)
    clips = list(synth(contract.MODELS, [1], SAID, language="fr").values())[0]
    n = len(far)
    ref = (far / np.sqrt(np.mean(far**2)) * 10 ** (-22 / 20)).astype(np.float32)
    heard = place(rng)(loudspeaker(ref, rng)).astype(np.float32)
    shift = int(offset_s * R)  # the microphone hears the machine `shift` samples later
    echo_mic = np.roll(heard, shift)
    if shift > 0:
        echo_mic[:shift] = 0
    person, spans, room = np.zeros(n, np.float32), [], place(rng)
    for i, clip in enumerate(clips):
        at = int((i + 0.6) * n / (len(clips) + 0.6))
        y = room(clip).astype(np.float32)[: n - at]
        person[at : at + len(y)] += y
        spans.append((at, at + len(y), SAID[i]))
    speech = np.zeros(n, bool)
    for a, b, _ in spans:
        speech[a:b] = True
    echo_mic *= 10 ** (-30 / 20) / np.sqrt(np.mean(echo_mic[speech] ** 2))
    person *= 10 ** (-30 / 20) / np.sqrt(np.mean(person[speech] ** 2))
    mic = echo_mic + person + (rng.standard_normal(n) * 10 ** (-65 / 20)).astype(np.float32)
    return mic, ref, person, spans, speech


def _run(mic, ref):
    """The two streams arrive together, chunk by chunk, the reference a little ahead
    (the ear waits for it, ``Echo.covers``)."""
    canceller, step = echo.Echo(), R * contract.CHUNK_S
    ahead = int(np.ceil(contract.ECHO_MOST_LAG_S / contract.CHUNK_S))
    chunks = [ref[start : start + step] for start in range(0, len(ref), step)]
    out, share = [], []
    for k, start in enumerate(range(0, len(mic), step)):
        for j in range(k, min(k + ahead + 1, len(chunks))):
            if j not in canceller.ref:
                canceller.push_ref(j, chunks[j])
        clean, s = canceller.cancel(k, mic[start : start + step])
        out.append(clean)
        share.append(s)
    return canceller, np.concatenate(out), np.concatenate(share)


def _recall(fast, x, spans):
    def one(truth, written):
        t = tokens(truth)
        return sum(op == "=" for op, _, _ in align(t, tokens(written))) / len(t)

    return float(np.mean([one(s, asr.read(fast, x[a:b]).text) for a, b, s in spans]))


@needs_data
@pytest.mark.parametrize("offset_s", [0.37, -0.42])
def test_a_person_is_heard_over_what_the_machine_plays(offset_s):
    """The offset is learned from the stream, the machine is taken out, the person
    stays: Parakeet reads the person's words over the machine's sound (a driver
    inside its linear range: seed 11)."""
    fast = asr.build(contract.FAST, contract.MODELS)
    mic, ref, person, spans, speech = _scene(11, offset_s)
    canceller, clean, share = _run(mic, ref)
    # the recorders' offset, plus the speaker's and the room's own delay (a few ms):
    # inside what the filter's tail covers
    assert 0 <= canceller.lag - int(offset_s * R) < contract.ECHO_TAIL_S * R
    quiet = ~speech
    quiet[: int(contract.ECHO_ESTIMATE_S + 4) * R] = (
        False  # the offset learned, the filter converged
    )
    removed_db = 10 * np.log10(np.mean(mic[quiet] ** 2) / np.mean(clean[quiet] ** 2))
    # bounded by the speaker's own nonlinearity, drawn per scene (a linear canceller
    # cannot take out what no linear filter models): 9-11 dB here, 16 dB on a gentler
    # driver (lab aec_compare)
    assert removed_db > 6, removed_db
    late = [(a, b, s) for a, b, s in spans if a > (contract.ECHO_ESTIMATE_S + 4) * R]
    assert late
    assert _recall(fast, clean, late) > _recall(fast, mic, late) + 0.3
    # the share says where the machine was: high where it alone played
    frames = np.arange(len(share)) * contract.ECHO_FRAME
    alone = quiet[frames]
    assert np.median(share[alone]) > np.median(share[~alone & (frames > 12 * R)])


def test_nothing_is_subtracted_before_the_offset_is_known_or_when_the_machine_is_silent():
    canceller = echo.Echo()
    mic = np.random.default_rng(1).standard_normal(R).astype(np.float32) * 0.1
    clean, share = canceller.cancel(0, mic)
    assert np.array_equal(clean, mic) and not share.any()
    for k in range(int(contract.ECHO_ESTIMATE_S) + 2):
        canceller.push_ref(k, np.zeros(R, np.float32))
        clean, share = canceller.cancel(k, mic)
    assert canceller.lag is None and np.array_equal(clean, mic)


@needs_data
def test_where_the_canceller_fails_it_says_so():
    """A small driver pushed into its nonlinearity (seed 3): a linear canceller cannot
    take the machine out (the person's words stay unread, measured in the lab), and
    what is left still follows the machine: ``left`` is higher than where it worked."""
    worked = _run(*_scene(11, 0.37)[:2])[0]
    failed = _run(*_scene(3, 0.37)[:2])[0]
    lefts = {}
    for name, seed in (("worked", 11), ("failed", 3)):
        mic, ref = _scene(seed, 0.37)[:2]
        canceller, step, values = echo.Echo(), R * contract.CHUNK_S, []
        for k, start in enumerate(range(0, len(mic), step)):
            for j in range(k, k + 3):
                if j * step < len(ref) and j not in canceller.ref:
                    canceller.push_ref(j, ref[j * step : (j + 1) * step])
            canceller.cancel(k, mic[start : start + step])
            if k > contract.ECHO_ESTIMATE_S + 4:
                values.append(canceller.left)
        lefts[name] = float(np.mean(values))
    assert worked.lag is not None and failed.lag is not None
    assert lefts["failed"] > lefts["worked"], lefts
