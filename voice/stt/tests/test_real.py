"""The ear in simulated rooms built from public recordings: the whole system, end to end.

Every scene goes through the real path: one-second raw chunks in a directory,
the tailer, Silero, CAM++, the model, the record, the state. The truth is known,
so the numbers are not opinions. What each scene is for:

* ``room``: six real voices taking turns, with coughs, doors, keyboard, laughter
  and a room noise floor in between. The everyday case;
* ``sounds``: 160 clips of 20 indoor sound classes (ESC-50 fold 5, never heard
  by the base) and no voice at all. The v0 failure (a person invented out of a
  keyboard) must not happen;
* ``silence``: two minutes of room floor. Nothing may be observed;
* ``conversation``: the pyannote sample, two people with fast turns;
* ``four``: the sherpa-onnx Mandarin file, four speakers, another language;
* ``twins``: synthetic voices the sensor cannot tell apart, fused, then
  corrected once. The paradigm: the error is obeyed now and does not come back.
"""

from __future__ import annotations

import collections
import random
import time

import pytest
from conftest import needs_data

from π_ear import contract
from π_ear.ear import Ear, Places, correction
from π_ear.io import read_jsonl
from π_ear.public import Scene, esc50, load, score, synth
from π_ear.public import room as public_room

pytestmark = needs_data
D = contract.DATA


def replay(tmp_path, scene: Scene, name: str, memory=None, asr=False):
    places = Places(
        root=tmp_path / "live",
        memory=memory or tmp_path / f"{name}-memory",
        audio=tmp_path / name,
        corpus=None,
    )
    scene.write(places.audio_dir)
    started = time.monotonic()
    ear = Ear(places, from_start=True, live=False, asr=asr)
    ear.run(drain=True)
    elapsed = time.monotonic() - started
    events = [e for e in read_jsonl(places.events)[0] if e.get("session") == ear.session]
    return ear, events, elapsed


@pytest.fixture(scope="module")
def sounds():
    return esc50(D)


@pytest.fixture(scope="module")
def room(tmp_path_factory, sounds):
    scene, voices = public_room(D, seed=3, sounds=sounds, varied=False)
    ear, events, elapsed = replay(tmp_path_factory.mktemp("room"), scene, "room")
    return scene, voices, ear, events, elapsed


@pytest.mark.parametrize("seed", range(1, 13))
def test_twelve_rooms_never_mix_two_voices_and_never_invent_one(tmp_path, sounds, seed):
    """Random gains, floors, takes and orders. Measured: purity 1.000 and no phantom
    in 12 rooms out of 12; one voice split in two in 4 of them (the safe error,
    see the next test)."""
    scene, voices = public_room(D, seed=seed, sounds=sounds)
    _, events, _ = replay(tmp_path, scene, "room")
    result = score(events, scene, voices)
    assert result["purity"] == 1.0 and result["phantoms"] == []
    assert all(1 <= n <= 2 for n in result["clusters_per_voice"].values())


def test_a_split_voice_said_once_to_be_one_stays_one(tmp_path, sounds):
    """Room 2 splits two voices (one FSDD speaker across takes, one LibriSpeech
    speaker across gains). One `same` per split, and in the next room, with new
    takes of the same people, each of them is one speaker again."""
    memory = tmp_path / "memory"
    scene, voices = public_room(D, seed=2, sounds=sounds)
    ear, events, _ = replay(tmp_path, scene, "day1", memory=memory)
    split = {v: k for v, k in score(events, scene, voices)["clusters_per_voice"].items() if k > 1}
    assert split, "room 2 no longer splits anyone: pick another room"
    for voice in split:
        heads = {}
        for e in events:
            if e.get("kind") in ("known", "new") and scene.label_at(e["t0"], e["t1"]) == voice:
                heads.setdefault(e["speaker"], e["row"])
        correction(ear.places, "same", list(heads.values()))
    scene2, _ = public_room(D, seed=8, sounds=sounds)
    _, events2, _ = replay(tmp_path, scene2, "day2", memory=memory)
    after = score(events2, scene2, voices)
    assert after["purity"] == 1.0
    for voice in split:
        assert after["clusters_per_voice"][voice] == 1, after["by_truth"][voice]


def test_room_counts_six_voices_and_never_mixes_two(room):
    scene, voices, ear, events, _ = room
    result = score(events, scene, voices)
    assert result["phantoms"] == []
    assert result["clusters_per_voice"] == {v: 1 for v in voices}
    assert result["purity"] == 1.0
    assert result["coverage"] >= 0.85


def test_room_sounds_are_not_people(room):
    """Coughs, laughter and breathing pass the VAD sometimes; they stay unknown,
    or are recognised as noise by what the base heard of ESC-50 folds 1-3."""
    scene, _, _, events, _ = room
    for e in events:
        if "row" in e and scene.label_at(e["t0"], e["t1"]).startswith("noise:"):
            assert e["kind"] in ("unknown", "noise"), e


def test_room_runs_far_faster_than_real_time(room):
    scene, _, _, _, elapsed = room
    assert elapsed < scene.length / contract.RATE / 10


def test_a_hundred_and_sixty_indoor_sounds_make_nobody(tmp_path, sounds):
    scene = Scene(seed=4).silence(1)
    for category, clips in sorted(sounds.items()):
        for x in clips:
            scene.add(f"noise:{category}", x, gain_db=-3).silence(0.3)
    ear, events, _ = replay(tmp_path, scene, "sounds")
    people = collections.Counter(
        scene.label_at(e["t0"], e["t1"]) for e in events if e.get("kind") in ("new", "known")
    )
    # Laughter is the one class that is a human voice: CAM++ hears the person who
    # laughs, and a long laugh may register as that person. Nothing else may.
    assert set(people) <= {"noise:laughing"}, people
    assert ear.model.summary()["speakers"] <= 1, people


def test_two_minutes_of_room_floor_observe_nothing(tmp_path):
    ear, events, _ = replay(tmp_path, Scene(seed=5).silence(120), "silence")
    assert events == [] and len(ear.store) == 0


def test_a_fast_two_person_conversation(tmp_path):
    x = load(D / "sample.wav")
    scene = Scene(seed=0)
    scene.audio, scene.length = [x], len(x)
    for line in (D / "sample.rttm").read_text().splitlines():
        parts = line.split()
        scene.truth.append((float(parts[3]), float(parts[3]) + float(parts[4]), parts[7]))
    _, events, _ = replay(tmp_path, scene, "conversation")
    result = score(events, scene, {"speaker90", "speaker91"})
    assert result["speakers"] == 2 and result["phantoms"] == []
    assert result["purity"] >= 0.7  # 2 s windows straddle turns of 0.4 to 4 s


def test_four_speakers_in_another_language(tmp_path):
    x = load(D / "zh4.wav")
    scene = Scene(seed=0)
    scene.audio, scene.length = [x], len(x)
    ear, _, _ = replay(tmp_path, scene, "four")
    assert ear.model.summary()["speakers"] == 4


TWINS = [480, 860, 12]
FIRST = [
    "The meeting was moved to Thursday because nobody could find the projector.",
    "I think we should take the train, it is faster and much less stressful.",
    "Could you send me the report before the end of the afternoon please?",
    "She planted tomatoes, basil and a small lemon tree on the balcony.",
]
SECOND = [
    "Honestly the second half of the film was better than the first one.",
    "We lost the signal somewhere between the bridge and the old factory.",
    "Remember to turn off the oven when the bread starts to smell good.",
    "The children ran across the field, laughing and shouting at the dogs.",
]


def _tts_scene(sentences, seed):
    voices = synth(contract.MODELS, TWINS, sentences)
    turns = [(k, u) for k, us in voices.items() for u in us]
    random.Random(seed).shuffle(turns)
    scene = Scene(seed=seed).silence(1)
    for key, u in turns:
        scene.add(key, u).silence(0.8)
    return scene, set(voices)


def _owners(events, scene):
    owner = collections.defaultdict(collections.Counter)
    for e in events:
        if e.get("kind") in ("known", "new") and e["learned"]:
            owner[scene.label_at(e["t0"], e["t1"])][e["speaker"]] += 1
    return owner


def test_twins_corrected_once_stay_apart(tmp_path):
    """Synthetic voices sit closer than real ones (EER 13 %, measured): the sensor
    fuses some of them. One `different` per fused pair, said once, and the next
    session keeps those voices apart, with nothing else touched."""
    memory = tmp_path / "memory"
    scene, voices = _tts_scene(FIRST, 1)
    ear, events, _ = replay(tmp_path, scene, "day1", memory=memory)
    owner = _owners(events, scene)
    fused = collections.defaultdict(list)
    for voice in sorted(voices):
        key = owner[voice].most_common(1)[0][0]
        row = next(
            e["row"]
            for e in reversed(events)
            if e.get("speaker") == key
            and scene.label_at(e["t0"], e["t1"]) == voice
            and e["learned"]
        )
        fused[key].append(row)
    corrected = [rows for rows in fused.values() if len(rows) > 1]
    for rows in corrected:
        correction(ear.places, "different", rows)
    scene2, _ = _tts_scene(SECOND, 2)
    ear2, events2, _ = replay(tmp_path, scene2, "day2", memory=memory)
    owner2 = _owners(events2, scene2)
    majority = {v: owner2[v].most_common(1)[0][0] for v in voices}
    assert len(set(majority.values())) == len(voices), (corrected, dict(owner2))
    assert ear2.counters["corrections"] == len(corrected)


def test_a_sound_said_once_not_to_be_a_voice_does_not_come_back(tmp_path, sounds):
    """Laughs, heard three times over, can look like a person. `not_voice` once on
    that "person", and the same laughs go to noise afterwards, not to a speaker."""
    scene = Scene(seed=6).silence(1)
    for _ in range(3):
        for clip in sounds["laughing"]:
            scene.add("noise:laughing", clip, gain_db=-3).silence(0.5)
    memory = tmp_path / "memory"
    ear, events, _ = replay(tmp_path, scene, "laugh1", memory=memory)
    phantoms = [c for c in ear.model.clusters.values() if c.kind == "speaker"]
    if not phantoms:
        pytest.skip("this clip never became a phantom: nothing to correct")
    for c in phantoms:
        correction(ear.places, "not_voice", list(c.members))
    ear2, events2, _ = replay(tmp_path, scene, "laugh2", memory=memory)
    assert ear2.model.summary()["speakers"] == 0
    assert any(e["kind"] == "noise" for e in events2 if "row" in e)


def test_what_was_said(tmp_path):
    """Whisper on a finished run: the transcript joins the record, linked by row."""
    tests = contract.MODELS / contract.WHISPER_DIR / "test_wavs"
    scene = Scene(seed=0).silence(1).add("libri1089", load(tests / "0.wav")).silence(2)
    _, events, _ = replay(tmp_path, scene, "asr", asr=True)
    texts = [e for e in events if e.get("kind") == "text"]
    assert texts and texts[0]["rows"]
    assert "lamps" in texts[0]["text"].lower() or "nightfall" in texts[0]["text"].lower()
