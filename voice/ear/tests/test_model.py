"""The learned ear, on voices drawn with the geometry CAM++ was measured to have.

No model, no download: each claim the paradigm rests on, checked on its own.
"""

from __future__ import annotations

import collections

import numpy as np
from conftest import World, vmf

from π_ear.model import Model
from π_ear.sphere import unit


def _purity(model: Model, truth: dict[int, int]) -> tuple[int, float]:
    owner = collections.defaultdict(collections.Counter)
    for row, who in truth.items():
        voice = model.voice_of(row)
        if voice is not None:
            owner[voice.key][who] += 1
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


def test_same_correction_joins_two_conditions_and_teaches_kappa_c():
    """One voice in two modes (two rooms, two days: cos 0.6 between the modes, the
    gap measured on FSDD), seen as two voices. Said once to be one: the two
    conditions join one voice, kappa_c (the drift between conditions) learns
    from every cross pair, and both modes keep landing in that voice."""
    world = World(seed=5)
    model, store = world.model()
    a = world.voices(1)[0]
    b = vmf(a, 180.0, 1, world.rng)[0]
    for mu in (a, b):
        for x in world.windows(mu, 10):
            world.feed(model, store, x)
    assert model.summary()["speakers"] == 2
    kappa_c = model.kappa_c
    model.correct("same", [0, 10])
    assert model.summary()["speakers"] == 1 and model.summary()["conditions"] == 2
    assert model.kappa_c < kappa_c
    assert model.links["count"] == 1
    fused = model.voice_of(0).key
    for mu in (a, b):
        for x in world.windows(mu, 5):
            world.feed(model, store, x)
            assert model.voice_of(len(store) - 1).key == fused


def test_conditions_of_one_voice_link_without_being_told():
    """The fragmentation fix (Q2): a voice heard in two conditions that drift as
    much as the base says voices drift is ONE voice with two conditions, and the
    other voice in the room is never folded into it. Over eight worlds (two
    conditions of one voice, one other voice), the conditions link in most, and
    two people are never one."""
    from conftest import Store

    from π_ear.model import fit_weights

    outcomes = []
    for seed in range(6, 14):
        world = World(seed=seed, kappa0=40.0)
        groups = {
            i: [list(world.windows(vmf(mu, 300.0, 1, world.rng)[0], 5)) for _ in range(4)]
            for i, mu in enumerate(world.voices(5))
        }
        store = Store()
        model = Model(fit_weights(groups, "drifting"), store)
        voice = world.voices(1)[0]
        for mode in vmf(voice, 300.0, 2, world.rng):
            for x in world.windows(mode, 12):
                world.feed(model, store, x)
        for x in world.windows(world.voices(1)[0], 12):
            world.feed(model, store, x)
        outcomes.append((model.summary()["speakers"], model.summary()["conditions"]))
    assert all(voices >= 2 for voices, _ in outcomes), "two people were made one"
    assert sum(o == (2, 3) for o in outcomes) >= 6, outcomes


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


# -- v1.1 ---------------------------------------------------------------------------


def test_the_margin_says_what_p_hides(world):
    """p saturates at 1.0 on 192 dimensions; the margin in nats keeps the doubt legible."""
    model, store = world.model()
    a, b = world.voices(2)
    for mu in (a, b):
        for x in world.windows(mu, 6):
            world.feed(model, store, x)
    clear = world.feed(model, store, world.windows(a, 1)[0])
    halfway = unit(a + b)
    torn = world.feed(model, store, halfway, learn=False)
    assert clear["p"] == torn["p"] == 1.0 or clear["margin"] > torn["margin"]
    assert clear["margin"] > torn["margin"] >= 0


def test_nothing_is_forgotten_before_anything_was_confirmed(world):
    model, store = world.model()
    for x in world.voices(5):
        world.feed(model, store, x)
    model.now += 10**6
    assert all(model.survival(c) == 1.0 for c in model.clusters.values())


def test_a_candidate_fades_at_the_rate_this_stream_confirms_voices():
    world = World(seed=21)
    model, store = world.model()
    voice = world.voices(1)[0]
    for x in world.windows(voice, 3):  # confirmed one second after its first window
        world.feed(model, store, x)
    lonely = world.voices(1)[0]
    world.feed(model, store, lonely)
    key = model.where[len(store) - 1]
    fresh = model.survival(model.clusters[key])
    model.now += 60
    model.refresh()
    assert model.survival(model.clusters[key]) < fresh
    model.now += 10**6  # long enough for the weight to be an exact zero
    model.refresh()
    model._forget()
    assert key not in model.clusters, "a weight of exactly zero is forgotten"


def test_candidates_that_never_come_back_are_explained_as_noise():
    """Survival with a cured fraction: windows heard once and never again lower
    the share of candidates that are real voices, instead of slowing everyone."""
    world = World(seed=22)
    model, store = world.model()
    for x in world.windows(world.voices(1)[0], 3):
        world.feed(model, store, x)
    alone = model.cure[0]
    for x in world.voices(10):
        world.feed(model, store, x)
    model.now += 100
    model.refresh()
    assert model.cure[0] < alone


def test_continuity_learns_only_from_labelled_transitions(world):
    model, store = world.model()
    voice = world.voices(1)[0]
    rows = []
    for x in world.windows(voice, 6):
        store.append(x)
        row = len(store) - 1
        model.observe(row, x, float(row), True, rows[-1] if rows else None)
        rows.append(row)
    assert model.stay() is None, "the model's own decisions never teach it continuity"
    model.correct("same", rows[:3])
    assert model.stay() is None, "one kind of label only: still nothing to estimate"
    model.correct("different", rows[3:5])
    assert model.stay() == 2 / 3


def test_continuity_breaks_a_tie_toward_the_voice_that_was_speaking():
    world = World(seed=23)
    model, store = world.model()
    a, b = world.voices(2)
    last = {}
    for mu, name in ((a, "a"), (b, "b")):
        for x in world.windows(mu, 8):
            world.feed(model, store, x)
            last[name] = len(store) - 1
    model.turns.update(stay=99.0, switch=1.0)
    tie = unit(a + b)
    after_a = model.explain(tie, model.where[last["a"]])[0]["id"]
    after_b = model.explain(tie, model.where[last["b"]])[0]["id"]
    assert after_a == model.where[last["a"]] and after_b == model.where[last["b"]]


def test_explain_ranks_every_explanation(world):
    model, store = world.model()
    voice = world.voices(1)[0]
    for x in world.windows(voice, 4):
        world.feed(model, store, x)
    options = model.explain(world.windows(voice, 1)[0])
    assert options[0]["id"] == model.where[0] and options[0]["behind_nats"] == 0
    assert any(o["id"] is None for o in options), "someone new is always an option"
    assert [o["behind_nats"] for o in options] == sorted(o["behind_nats"] for o in options)


def test_trust_is_none_until_both_outcomes_are_seen_then_grows_with_the_margin(world):
    """Q1: a margin in nats becomes P(right). The model does not pretend before it has
    seen itself both right and wrong; with the base's own replay it knows from day one."""
    model, _ = world.model()
    assert model.trust(10.0) is None
    rng = np.random.default_rng(0)
    model.calibration = [
        [m, float(rng.uniform() < 1 - np.exp(-m / 4))] for m in rng.uniform(0, 30, 300)
    ]
    low, mid, high = model.trust(0.5), model.trust(5.0), model.trust(25.0)
    assert low < mid < high and low < 0.5 < high


def test_a_wrong_decision_said_once_lowers_the_trust_of_its_margin(world):
    """A correction shows a confident decision was wrong: decisions with that
    margin are trusted less from then on."""
    model, store = world.model()
    a, b = world.voices(2)
    for x in np.vstack([world.windows(a, 6), world.windows(b, 6)]):
        world.feed(model, store, x)
    model.calibration += [[m, 1.0] for m in np.linspace(1, 40, 20)] + [[0.5, 0.0]]
    model.trust_weights = None
    margin = model.margin_of[11]
    before = model.trust(margin)
    model.correct("different", [0, 11])  # if they shared a voice, that decision was wrong
    model.calibration.append([margin, 0.0])
    model.trust_weights = None
    assert model.trust(margin) < before
