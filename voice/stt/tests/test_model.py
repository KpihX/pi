"""The learned ear, on voices drawn with the geometry CAM++ was measured to have.

No model, no download: each claim the paradigm rests on, checked on its own.
"""

from __future__ import annotations

import collections

import numpy as np
from conftest import World, vmf

from π_ear.model import Model


def _purity(model: Model, truth: dict[int, int]) -> tuple[int, float]:
    owner = collections.defaultdict(collections.Counter)
    for row, who in truth.items():
        if row in model.where and model.clusters[model.where[row]].kind == "speaker":
            owner[model.where[row]][who] += 1
    right = sum(c.most_common(1)[0][1] for c in owner.values())
    return len(owner), right / max(sum(sum(c.values()) for c in owner.values()), 1)


def test_cold_start_unknown_then_new_then_known(world):
    """At t=0 the state is empty, and the ear is still honest: it says unknown."""
    model, store = world.model()
    voice = world.voices(1)[0]
    kinds = [world.feed(model, store, x)["kind"] for x in world.windows(voice, 4)]
    assert kinds == ["unknown", "new", "known", "known"]


def test_one_window_never_makes_a_person(world):
    """Random directions are noise: each stays a candidate, nobody is registered."""
    model, store = world.model()
    for x in world.voices(30):
        world.feed(model, store, x)
    assert model.summary()["speakers"] == 0


def test_interleaved_voices_are_counted_and_kept_apart(world):
    model, store = world.model()
    voices = world.voices(6)
    truth = {}
    order = [(i, x) for i, mu in enumerate(voices) for x in world.windows(mu, 25)]
    world.rng.shuffle(order)
    for who, x in order:
        world.feed(model, store, x)
        truth[len(store) - 1] = who
    speakers, purity = _purity(model, truth)
    assert speakers == 6
    assert purity == 1.0


def test_a_glimpse_is_decided_but_never_learned(world):
    model, store = world.model()
    voice = world.voices(1)[0]
    for x in world.windows(voice, 3):
        world.feed(model, store, x)
    before = model.to_dict()
    decision = world.feed(model, store, world.windows(voice, 1)[0], learn=False)
    assert decision["kind"] == "known" and decision["learned"] is False
    assert model.to_dict()["clusters"] == before["clusters"]


def test_one_voice_is_one_speaker_however_long_it_talks():
    """The v0 failure was one voice, many entries (56 identities for two voices)."""
    world = World(seed=3)
    model, store = world.model()
    voice = world.voices(1)[0]
    for x in world.windows(voice, 40):
        world.feed(model, store, x)
    assert model.summary()["speakers"] == 1


def test_same_correction_fuses_and_teaches_kappa():
    """One voice in two modes (two rooms, two days: cos 0.6 between the modes, the
    gap measured on FSDD), seen by the sensor as two voices. Said once to be one:
    fused at once, kappa absorbs all the cross pairs, and both modes land in it."""
    world = World(seed=5)
    model, store = world.model()
    a = world.voices(1)[0]
    b = vmf(a, 180.0, 1, world.rng)[0]
    for mu in (a, b):
        for x in world.windows(mu, 10):
            world.feed(model, store, x)
    assert model.summary()["speakers"] == 2
    kappa = model.kappa
    model.correct("same", [0, 10])
    assert model.summary()["speakers"] == 1
    assert model.kappa < kappa
    assert model.pairs["count"] == 100
    # and the next windows of either voice land in the fused speaker
    fused = model.where[0]
    for mu in (a, b):
        for x in world.windows(mu, 5):
            world.feed(model, store, x)
            assert model.where[len(store) - 1] == fused


def test_different_correction_splits_and_does_not_come_back():
    """Two voices the sensor cannot tell apart (an exaggerated TTS case), fused,
    then said once to be two: split now, never re-merged, and each new window
    goes to its own side."""
    world = World(seed=7)
    model, store = world.model()
    a = world.voices(1)[0]
    b = vmf(a, 3000.0, 1, world.rng)[0]  # a near-twin voice
    truth = {}
    for i, mu in enumerate((a, b)):
        for x in world.windows(mu, 15):
            world.feed(model, store, x)
            truth[len(store) - 1] = i
    model.correct("different", [0, 15])
    assert model.where[0] != model.where[15]
    assert (0, 15) in model.cannot
    for _ in range(3):
        for i, mu in enumerate((a, b)):
            for x in world.windows(mu, 5):
                world.feed(model, store, x)
                truth[len(store) - 1] = i
    assert model.where[0] != model.where[15], "the split was undone by a merge"


def test_not_voice_teaches_a_noise_cluster(world):
    """A repeated sound that became a 'speaker' is said once not to be a voice:
    the phantom is gone, and the next occurrence goes to noise."""
    model, store = world.model()
    sound = world.voices(1)[0]
    for x in world.windows(sound, 5):
        world.feed(model, store, x)
    phantom = model.where[0]
    assert model.clusters[phantom].kind == "speaker"
    model.correct("not_voice", list(model.clusters[phantom].members))
    assert model.summary()["speakers"] == 0
    assert model.summary()["noise"] >= 1
    decision = world.feed(model, store, world.windows(sound, 1)[0])
    assert decision["kind"] == "noise"


def test_a_name_survives_a_merge(world):
    model, store = world.model()
    voice = world.voices(1)[0]
    for x in world.windows(voice, 5):
        world.feed(model, store, x)
    model.correct("name", [0], label="KπX")
    for x in world.windows(voice, 5):
        assert world.feed(model, store, x)["label"] == "KπX"


def test_the_checkpoint_round_trips(world):
    model, store = world.model()
    for mu in world.voices(3):
        for x in world.windows(mu, 6):
            world.feed(model, store, x)
    clone = Model(model.base, store)
    clone.load(model.to_dict())
    probe = world.windows(world.voices(1)[0], 1)[0]
    assert clone.scores(probe, ("speaker",)) == model.scores(probe, ("speaker",))
    assert clone.kappa == model.kappa and clone.alpha == model.alpha


def test_keys_are_stable_and_merges_keep_the_oldest():
    world = World(seed=13)
    model, store = world.model()
    voice = world.voices(1)[0]
    for x in world.windows(voice, 3):
        world.feed(model, store, x)
    key = model.where[0]
    for x in world.windows(voice, 50):
        world.feed(model, store, x)
    assert model.where[0] == key
    assert np.isfinite(model.kappa)
