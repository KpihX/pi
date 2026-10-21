"""Second thoughts (``revise.py``): a decision said again when the model knows better."""

from __future__ import annotations

from types import SimpleNamespace

from conftest import vmf

from π_ear.revise import Second, resolve


def _event(decision: dict) -> dict:
    return {"kind": decision["kind"], "speaker": decision["speaker"]}


def test_a_first_window_said_unknown_is_said_again_once_its_voice_is_known(world):
    """The first window of a voice cannot be anyone (one window makes nobody); once
    the voice is a speaker, that window is said again as theirs, because it learned."""
    model, store = world.model()
    second = Second()
    voice = world.voices(1)[0]
    for x in world.windows(voice, 6):
        decision = world.feed(model, store, x)
        second.keep(len(store) - 1, _event(decision))
    assert second.said[0]["kind"] == "unknown"
    changes = second.check(model, store)
    assert [row for row, _, _ in changes] == [0]
    row, was, now = changes[0]
    assert was == {"kind": "unknown", "speaker": None}
    assert now["kind"] == "known" and now["why"] == "learned"
    assert now["speaker"] == model.voice_of(5).key


def test_nothing_is_said_again_when_nothing_changed(world):
    model, store = world.model()
    second = Second()
    for mu in world.voices(2):
        for x in world.windows(mu, 8):
            second.keep(len(store), _event(world.feed(model, store, x)))
    for row, _, now in second.check(model, store):
        second.said[row].update(kind=now["kind"], speaker=now["speaker"])
    assert second.check(model, store) == []  # said again once, then it holds


def test_a_glimpse_is_scored_again_without_being_learned(world):
    """A window never learned from (a glimpse) is decided again by the model as it is
    now, and the model learns nothing from it: the revision does not feed itself."""
    model, store = world.model()
    second = Second()
    voice = world.voices(1)[0]
    first = world.windows(voice, 1)[0]
    second.keep(0, _event(world.feed(model, store, first, learn=False)))
    for x in world.windows(voice, 6):
        world.feed(model, store, x)
    members = sum(c.n for c in model.clusters.values())
    (row, was, now), *rest = second.check(model, store)
    assert not rest and row == 0 and now["kind"] == "known" and now["why"] == "learned"
    assert sum(c.n for c in model.clusters.values()) == members
    # heard in another place than the model is in: not scored again, not known
    other = Second()
    other.keep(0, {"kind": "unknown", "speaker": None})
    assert other.check(model, store, here=lambda event: False) == []


def test_a_join_says_the_absorbed_voice_s_windows_again(world):
    """Two voices said to be one: the windows named by the correction are revised as
    corrected, the other windows of the absorbed voice as merged."""
    model, store = world.model()
    second = Second()
    a, b = world.voices(2)
    rows = {}
    for name, mu in (("a", a), ("b", b)):
        for x in world.windows(mu, 8):
            rows.setdefault(name, []).append(len(store))
            second.keep(len(store), _event(world.feed(model, store, x)))
    for row, _, now in second.check(model, store):
        second.said[row].update(kind=now["kind"], speaker=now["speaker"])
    told = [rows["a"][-1], rows["b"][-1]]
    model.correct("same", told)
    second.corrected |= set(told)
    changes = {row: now for row, _, now in second.check(model, store)}
    gone = {row for row in changes}
    assert gone & set(told) and all(changes[r]["why"] == "corrected" for r in gone & set(told))
    merged = gone - set(told)
    assert merged and all(changes[r]["why"] == "merged" for r in merged)
    assert len({now["speaker"] for now in changes.values()}) == 1
    assert second.corrected == set()


def test_where_a_voice_went_is_followed_through_every_fusion():
    assert resolve("a", {"a": "b", "b": "c"}) == "c"
    assert resolve("a", {"a": "b", "b": "a"}) in ("a", "b")  # a cycle ends
    assert resolve(None, {}) is None


def _stub(scores: dict[str, dict], sizes: dict[str, int]):
    """A model of voices of one condition each, with the link scores given."""
    clusters = {k: SimpleNamespace(n=n, members=list(range(n))) for k, n in sizes.items()}
    voices = {k: SimpleNamespace(key=k, conditions=[k]) for k in sizes}
    model = SimpleNamespace(voices=voices, clusters=clusters)
    model.voice_scores = lambda cluster: next(
        scores[k] for k, c in clusters.items() if c is cluster
    )
    return model


def test_a_question_is_asked_only_if_its_answer_should_change_a_decision():
    """value = min(P, 1 - P) * the windows that would change hands."""
    # B (6 windows) belongs to A with P = 1/2: three decisions expected to change
    model = _stub({"A": {None: 0.0}, "B": {"A": 0.0, None: 0.0}}, {"A": 40, "B": 6})
    best = Second().same_question(model, lambda k: True)
    assert best["speakers"] == ["A", "B"] and abs(best["value"] - 3) < 1e-9
    # the model is sure (P = e^-10): no answer is expected to change anything
    model = _stub({"A": {None: 0.0}, "B": {"A": -10.0, None: 0.0}}, {"A": 40, "B": 6})
    assert Second().same_question(model, lambda k: True) is None
    # a recording is nobody to ask about
    model = _stub({"A": {None: 0.0}, "B": {"A": 0.0, None: 0.0}}, {"A": 40, "B": 6})
    assert Second().same_question(model, lambda k: k != "B") is None


def test_the_most_valuable_question_first_and_never_twice():
    scores = {
        "A": {None: 0.0},
        "B": {"A": 0.0, None: 0.0},  # 6 windows at P = 1/2: value 3
        "C": {"A": -1.0, None: 0.0},  # 20 windows at P = 0.27: value 5.4
    }
    model = _stub(scores, {"A": 40, "B": 6, "C": 20})
    second = Second()
    best = second.same_question(model, lambda k: True)
    assert best["speakers"] == ["A", "C"]
    second.asked.add(frozenset(best["speakers"]))
    assert second.same_question(model, lambda k: True)["speakers"] == ["A", "B"]


def test_two_voices_the_model_is_sure_of_are_not_asked_about(world):
    model, store = world.model()
    for mu in world.voices(2):
        for x in world.windows(mu, 10):
            world.feed(model, store, x)
    assert len(model.voices) == 2
    assert Second().same_question(model, lambda k: True) is None


def test_a_voice_close_to_another_is_kept_apart_or_joined_by_the_model_not_asked(world):
    """Measured on this world's geometry: between "one voice" and "two", the model's own
    link evidence jumps from a fusion to tens of nats apart; no pair lands in doubt, so
    nothing is asked. The question is for where the model does not know."""
    model, store = world.model()
    a = world.voices(1)[0]
    for x in world.windows(a, 20):
        world.feed(model, store, x)
    for x in world.windows(vmf(a, 400.0, 1, world.rng)[0], 3):
        world.feed(model, store, x)
    assert Second().same_question(model, lambda k: True) is None
