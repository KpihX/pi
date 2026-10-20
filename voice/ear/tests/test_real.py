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

import numpy as np
import pytest
from conftest import needs_data

from π_ear import contract
from π_ear.ear import Ear, Places, correction
from π_ear.io import read_jsonl
from π_ear.public import Scene, esc50, libri, load, loudspeaker, place, score, synth
from π_ear.public import room as public_room

pytestmark = needs_data
D = contract.DATA


def asr_models() -> bool:
    from π_ear import asr

    return asr.available(contract.FAST, contract.MODELS) and asr.available(
        contract.SLOW, contract.MODELS
    )


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
    assert "NaN" not in places.events.read_text(), "a NaN reached the record"
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
    """Random gains, floors, takes and orders. Measured (v2): purity 1.000 in 12
    rooms out of 12, no voice split (v1: 4 rooms out of 12 split one), and one
    laugh heard as the person who laughs in room 12 (a human voice, not a sound:
    see the 160 sounds below)."""
    scene, voices = public_room(D, seed=seed, sounds=sounds)
    _, events, _ = replay(tmp_path, scene, "room")
    result = score(events, scene, voices)
    assert result["purity"] == 1.0
    assert {result["owners"][s] for s in result["phantoms"]} <= {"noise:laughing"}
    assert all(n == 1 for n in result["clusters_per_voice"].values())
    assert result["turns"] >= 0.95


def _elsewhere(scene: Scene, seed: int) -> Scene:
    """The whole scene heard in another place: one room, one microphone, one floor."""
    scene.audio = [place(np.random.default_rng(seed))(np.concatenate(scene.audio))]
    return scene


def _keys(events, scene, voices) -> dict[str, collections.Counter]:
    keys = collections.defaultdict(collections.Counter)
    for e in events:
        if e.get("kind") in ("known", "new") and e["learned"]:
            label = scene.label_at(e["t0"], e["t1"])
            if label in voices:
                keys[label][e["speaker"]] += 1
    return keys


def test_a_voice_heard_elsewhere_said_once_to_be_one_stays_one(tmp_path, sounds):
    """Day 1, six voices in one room. Day 2, the same people in another place
    (another room, another microphone): measured, every voice but one comes back
    under a new key, some under two (the place moves every voice further than the
    base's places did). One `same` per voice, joining a day-1 row and its day-2
    rows; day 3, new takes in that second place, and each voice is its day-1
    voice again (measured: whole, but for one window of one voice)."""
    memory = tmp_path / "memory"
    day1, voices = public_room(D, seed=2, sounds=sounds)
    ear, events1, _ = replay(tmp_path, day1, "day1", memory=memory)
    first = {v: k.most_common(1)[0][0] for v, k in _keys(events1, day1, voices).items()}
    day2 = _elsewhere(public_room(D, seed=8, sounds=sounds)[0], 7)
    _, events2, _ = replay(tmp_path, day2, "day2", memory=memory)
    keys2 = _keys(events2, day2, voices)
    moved = {v for v in voices if set(keys2[v]) != {first[v]}}
    assert len(moved) >= 3, keys2  # if not, the second place is too mild to test this
    row = {}
    for e in events1 + events2:
        if e.get("kind") in ("known", "new") and e["learned"]:
            row.setdefault(e["speaker"], e["row"])
    for voice in sorted(moved):
        correction(ear.places, "same", [row[first[voice]]] + [row[k] for k in keys2[voice]])
    day3 = _elsewhere(public_room(D, seed=9, sounds=sounds)[0], 7)
    _, events3, _ = replay(tmp_path, day3, "day3", memory=memory)
    keys3 = _keys(events3, day3, voices)
    for voice in voices:
        key, n = keys3[voice].most_common(1)[0]
        assert key == first[voice] and n >= 0.9 * sum(keys3[voice].values()), keys3[voice]


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
    assert [e["kind"] for e in events] == ["session"] and len(ear.store) == 0
    assert ear.state()["audio"]["floor_dbfs"] == pytest.approx(-60, abs=1.5)


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
    # 2 s windows straddle turns of 0.4 to 4 s: measured, 0.625 of the learned
    # windows, and 0.75 of the time once the half windows take the voice of the
    # more confident window covering them (the ``turns`` line, Q4). Before the
    # place prior, the two voices of this room were heard as one.
    assert result["purity"] >= 0.6 and result["turns"] >= 0.7


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
    phantoms = list(ear.model.voices.values())
    if not phantoms:
        pytest.skip("this clip never became a phantom: nothing to correct")
    for voice in phantoms:
        rows = [r for c in voice.conditions for r in ear.model.clusters[c].members]
        correction(ear.places, "not_voice", rows)
    ear2, events2, _ = replay(tmp_path, scene, "laugh2", memory=memory)
    assert ear2.model.summary()["speakers"] == 0
    assert any(e["kind"] == "noise" for e in events2 if "row" in e)


def test_what_was_said(tmp_path):
    """Whisper on a finished run: the transcript joins the record, linked by row."""
    tests = contract.MODELS / "sherpa-onnx-whisper-tiny" / "test_wavs"
    scene = Scene(seed=0).silence(1).add("libri1089", load(tests / "0.wav")).silence(2)
    _, events, _ = replay(tmp_path, scene, "asr", asr=True)
    texts = [e for e in events if e.get("kind") == "text"]
    assert texts and texts[0]["rows"]
    assert "lamps" in texts[0]["text"].lower() or "nightfall" in texts[0]["text"].lower()


def test_a_voice_from_a_loudspeaker_is_a_device_and_a_device_said_once_stays_one(tmp_path):
    """Q3: a person in the room, another voice played by a loudspeaker. Measured
    on three pairs: 5 of the 6 voices placed right from the spectral shape alone,
    the miss (a device heard as a person) at p 0.77, not sure of itself. One
    ``device`` or ``person`` per miss, and from then on that voice is what it
    was told, in this session and the next."""
    voices = libri(D)
    rng = np.random.default_rng(0)
    right, total = 0, 0
    for n, (live, played) in enumerate(
        (("libri5703", "libri198"), ("libri198", "libri3436"), ("libri3436", "libri5703"))
    ):
        a, b = voices[live], loudspeaker(voices[played], rng)
        scene = Scene(seed=0).silence(1)
        for label, x in (
            ("person", a[: len(a) // 2]),
            ("device", b[: len(b) // 2]),
            ("person", a[len(a) // 2 :]),
            ("device", b[len(b) // 2 :]),
        ):
            scene.add(label, x).silence(1)
        memory = tmp_path / f"memory{n}"
        ear, events, _ = replay(tmp_path, scene, f"pair{n}", memory=memory)
        result = score(events, scene, {"person", "device"})
        assert result["speakers"] == 2 and result["purity"] == 1.0
        for key, truth in result["owners"].items():
            total += 1
            decided, _ = ear.source_of(key)
            right += decided == truth
            if decided != truth:
                correction(ear.places, truth, ear.voice_rows(key)[:1])
                ear.run(drain=True)
                assert ear.source_of(key) == (truth, 1.0)
                ear2, events2, _ = replay(tmp_path, scene, f"pair{n}-again", memory=memory)
                heard = {e["source"] for e in events2 if "row" in e and e["speaker"] == key}
                assert heard == {truth}, heard
    assert total == 6 and right >= 5


def _bed(x: np.ndarray, music: np.ndarray, snr_db: float, rng) -> np.ndarray:
    start = int(rng.integers(len(music) - len(x)))
    m = music[start : start + len(x)]
    m = m * np.sqrt(np.mean(x * x) / np.mean(m * m) / 10 ** (snr_db / 10))
    return (x + m).astype(np.float32)


def test_music_in_the_room_makes_nobody_and_mixes_no_one(tmp_path, sounds):
    """Q6, for who: the six-voice room with music playing all along, 10 dB under
    the speech. Measured: purity 1.000, each voice one speaker, and the music
    alone never a person (at 5 dB, two voices split: the safe error)."""
    music = [load(p) for p in sorted((D / "music").glob("*.ogg"))]
    scene, voices = public_room(D, seed=3, sounds=sounds, varied=False)
    x = np.concatenate(scene.audio)
    bed = np.concatenate(music * (1 + len(x) // sum(map(len, music))))[: len(x)]
    speech = np.mean(np.square(x[np.abs(x) > 0.01]))
    scene.audio = [(x + bed * np.sqrt(speech / np.mean(bed * bed) / 10)).astype(np.float32)]
    _, events, _ = replay(tmp_path, scene, "music")
    result = score(events, scene, voices)
    assert result["purity"] == 1.0
    assert all(n == 1 for n in result["clusters_per_voice"].values())
    assert "silence" not in {result["owners"][s] for s in result["phantoms"]}


FR_TEST = [
    "Le marché du samedi ferme plus tôt pendant les vacances d'été.",
    "Il a laissé son vélo devant la bibliothèque sans mettre le cadenas.",
]


def _read(tmp_path, name, turns):
    scene = Scene(seed=0).silence(1)
    for label, x in turns:
        scene.add(label, x).silence(2)
    _, events, _ = replay(tmp_path, scene, name, asr=True)
    final = {}
    for e in events:
        if e.get("kind") == "text":
            final[scene.label_at(e["t0"], e["t1"])] = e  # the last revision wins
    return final


def test_english_and_french_are_read_clean_and_through_music(tmp_path):
    """W1-W3 on the real recognisers (Parakeet v3 fast, Whisper turbo slow): two
    languages, clean and under music at 0 dB, in the ear's own loop; each final
    text is a revision of the fast one."""
    from π_ear.lexicon import errors, tokens

    if not asr_models():
        pytest.skip("the default recognisers are not downloaded (make fetch)")
    music = [load(p) for p in sorted((D / "music").glob("*.ogg"))]
    rng = np.random.default_rng(0)
    en = next(iter(synth(contract.MODELS, [TWINS[0]], FIRST[:2]).values()))
    fr = next(iter(synth(contract.MODELS, [1], FR_TEST, language="fr").values()))
    truths = {"en0": FIRST[0], "fr0": FR_TEST[0], "en1": FIRST[1], "fr1": FR_TEST[1]}
    clean = [("en0", en[0]), ("fr0", fr[0])]
    mixed = [("en1", _bed(en[1], music[0], 0, rng)), ("fr1", _bed(fr[1], music[2], 0, rng))]
    final = _read(tmp_path, "langs", clean + mixed)
    assert set(final) == set(truths), list(final)
    wrong = said = 0
    for label, e in final.items():
        truth = tokens(truths[label])
        wrong += errors(truth, tokens(e["text"]))
        said += len(truth)
        assert e["revision"] >= 1 and e["words"] and e["doubt"] != "not_understood"
    # measured: 3 errors in 49 words (6 %), two of them one homophone ("plus tôt"
    # heard "plutôt"), the kind of error one `text` correction is for
    assert wrong / said <= 0.1, (wrong, said, {k: e["text"] for k, e in final.items()})
