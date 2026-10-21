"""The ear on a real session: segments of a live recording, replayed through the ear.

The recording (``tests/live/<name>.json`` is its manifest; the audio stays on the
operator's machine, ``π_ear.live``) is replayed segment by segment through the real
ear, from the memory the ear had at the end of that session, with the real
recognisers. What is asserted is what each segment is for (its ``expect``), and that
the ear does at least as well as the ear that heard it live (v2.0, the same base
models): a measurement against several references, never one taken as the truth.

Slow (minutes per segment): ``make test-live``. Skipped without the recording.
"""

from __future__ import annotations

import collections

import numpy as np
import pytest

from π_ear import asr, contract
from π_ear.ear import Ear, Places
from π_ear.io import read_jsonl
from π_ear.lexicon import tokens
from π_ear.live import Recording, recall, words_between

NAME = "2026-10-20-psg-barca"
REC = Recording.open(NAME)
RECOGNISERS = asr.available(contract.FAST, contract.MODELS) and asr.available(
    contract.SLOW, contract.MODELS
)
pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(REC is None, reason=f"the live recording {NAME} is not on this machine"),
    pytest.mark.skipif(not RECOGNISERS, reason="the default recognisers are not downloaded"),
]

# What v2.0 wrote live on each segment, recall of the words of two other systems'
# readings (Whisper turbo told French on 30 s blocks; Parakeet on 20 s blocks),
# measured 2026-10-21 from the session's own record.
LIVE_V20 = {
    "match": {"turbo-fr": 0.426, "parakeet": 0.593},
    "kpihx": {"turbo-fr": 0.235, "parakeet": 0.436},
    "studio": {"turbo-fr": 0.392, "parakeet": 0.506},
    "player": {"turbo-fr": 0.013, "parakeet": 0.016},
}


def replay(tmp_path, name: str):
    segment = REC.segments[name]
    memory = REC.memory(tmp_path / "memory")
    places = Places(root=tmp_path / "live", memory=memory, audio=tmp_path / "chunks", corpus=None)
    REC.write(segment, places.audio_dir)
    ear = Ear(places, from_start=True, live=False)
    ear.run(drain=True)
    events = [e for e in read_jsonl(places.events)[0] if e.get("session") == ear.session]
    final = {}
    for e in events:
        if e.get("kind") == "text":
            final[e["piece"]] = e
    return segment, ear, events, sorted(final.values(), key=lambda e: e["t0"])


def written_recall(segment, texts) -> dict[str, float]:
    written = [w for e in texts for w in tokens(e["text"])]
    return {
        system: recall(words_between(REC.listening(system), segment.start, segment.end), written)
        for system in ("turbo-fr", "parakeet")
        if REC.listening(system)
    }


def precision(segment, texts) -> float:
    """The share of the words written that either other system also heard, in order."""
    written = [w for e in texts for w in tokens(e["text"])]
    if not written:
        return 1.0
    heard = [
        recall(written, words_between(REC.listening(system), segment.start, segment.end))
        for system in ("turbo-fr", "parakeet")
    ]
    return max(heard)


def check(segment, ear, events, texts) -> None:
    expect = segment.expect
    assert not [e for e in events if e.get("kind") == "error"], "a job of the scribe failed"
    voices = {e["speaker"] for e in events if e.get("kind") in ("known", "new") and e["learned"]}
    sources = {v: ear.source_of(v)[0] for v in voices}
    if "person" in expect:
        assert ("person" in sources.values()) == expect["person"], sources
    if "language" in expect:
        said = collections.Counter(e["language"] for e in texts if e["text"])
        assert said.most_common(1)[0][0] == expect["language"], said
    window = asr.window_s(asr.build(contract.SLOW, contract.MODELS))
    assert all(e["t1"] - e["t0"] < window for e in texts), "a piece longer than Whisper reads"
    if expect.get("invent") is False:
        assert precision(segment, texts) >= 0.5, texts


@pytest.mark.parametrize("name", ["match", "kpihx", "studio", "player"])
def test_a_live_segment_is_read_at_least_as_well_as_live(tmp_path, name):
    segment, ear, events, texts = replay(tmp_path, name)
    check(segment, ear, events, texts)
    now = written_recall(segment, texts)
    for system, then in LIVE_V20[name].items():
        # one word in a hundred: the replay's own noise (the match segment, at the oracle
        # of its two readings, measured 0.421 against v2.0's 0.426)
        assert now[system] >= then - 0.01, (system, now, LIVE_V20[name])


def test_kpihx_in_the_room_is_a_person_and_the_television_devices(tmp_path):
    """The voice the ear learned on 2026-10-19 (``b54d13``) is found again, as a person;
    every voice of the broadcast is a device (live, every source decision of this
    session read another window's spectrum, and KπX came out a device)."""
    segment, ear, events, texts = replay(tmp_path, "kpihx")
    voices = collections.Counter(
        e["speaker"] for e in events if e.get("kind") in ("known", "new") and e["learned"]
    )
    kpihx = next(v for v in voices if v.startswith("b54d13"))
    assert ear.source_of(kpihx)[0] == "person"
    others = [v for v in voices if v != kpihx and voices[v] >= 3]
    assert others and all(ear.source_of(v)[0] == "device" for v in others)


def test_an_interview_question_and_its_answer_are_two_voices_two_pieces(tmp_path):
    """Live, the journalist's question and the player's answer fell in one run and one
    text. The scribe cuts where the voice changes: each piece is one voice's."""
    segment, ear, events, texts = replay(tmp_path, "player")
    owners = collections.Counter(e["speaker"] for e in texts if e["text"])
    assert len([v for v, n in owners.items() if n >= 3]) >= 2, owners
    mixed = [e for e in texts if e["speakers"] and max(e["speakers"].values()) < 0.6]
    assert len(mixed) <= len(texts) // 4, mixed


def test_nothing_is_written_where_almost_nothing_is_heard(tmp_path):
    segment, ear, events, texts = replay(tmp_path, "quiet")
    check(segment, ear, events, texts)


def test_a_studio_run_longer_than_a_minute_is_handed_over_whole():
    from π_ear.sense import Sensor

    segment = REC.segments["studio"]
    x = REC.audio(segment.start, segment.end)
    sensor, longest = Sensor(contract.MODELS), 0.0
    for start in range(0, len(x), contract.RATE):
        sensor.feed(x[start : start + contract.RATE])
        current = sensor.run_audio()
        if current is not None:
            begin, samples = current
            assert np.array_equal(samples, x[begin : begin + len(samples)])
            longest = max(longest, len(samples) / contract.RATE)
    assert longest > 60, f"no run longer than a minute here ({longest:.0f} s)"
