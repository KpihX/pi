"""The *what*, inside the running ear: partials, revisions, questions, answers.

The real loop (Silero, CAM++, the model of who, the scribe's thread), with
recognisers plugged in by the registry, the way any other recogniser plugs in.
They say what they are told to say, so what is tested is the ear, not a model:
when the text is written, what it says it does not know, what it asks, and that
an answer, once given, is not needed again.
"""

from __future__ import annotations

import pytest
from conftest import needs_data

from π_ear import contract, port
from π_ear.asr import Hypothesis, Word
from π_ear.ear import Ear, Places
from π_ear.io import read_jsonl
from π_ear.lexicon import tokens
from π_ear.public import Scene, libri

pytestmark = needs_data


class Says:
    """A recogniser that hears ``words``, fast, with a log-probability per word."""

    def __init__(self, words: str, name: str, logp: float = -0.2):
        self.words, self.name, self.logp, self.calls = words, name, logp, 0

    def __call__(self, samples):
        self.calls += 1
        said = self.words.split()
        n = len(samples) / contract.RATE
        return Hypothesis(
            self.words,
            tuple(
                Word(w, i * n / len(said), (i + 1) * n / len(said), self.logp)
                for i, w in enumerate(said)
            ),
            "en",
            self.name,
        )


@pytest.fixture(scope="module")
def voice():
    return libri(contract.DATA)["libri5703"]


def _plug(monkeypatch, fast, slow=None):
    plugged = {contract.FAST: fast, contract.SLOW: slow}
    monkeypatch.setattr("π_ear.ear.asr.build", lambda name, models: plugged.get(name))


def _day(tmp_path, name: str, audio, memory=None) -> tuple[Ear, list[dict]]:
    places = Places(
        root=tmp_path / "live", memory=memory or tmp_path / "memory", audio=tmp_path / name
    )
    Scene(seed=0).silence(1).add("A", audio).silence(2).write(places.audio_dir)
    ear = Ear(places, from_start=True, live=False)
    ear.run(drain=True)
    events = [e for e in read_jsonl(places.events)[0] if e.get("session") == ear.session]
    return ear, events


def said(text: str, truth: str) -> bool:
    return tokens(text) == tokens(truth)


def _texts(events) -> list[dict]:
    return [e for e in events if e.get("kind") == "text"]


def test_the_text_appears_while_it_is_said_then_is_revised_in_the_silence(
    tmp_path, voice, monkeypatch
):
    """W1 and W2: partial lines while the run goes on, the fast reading at its
    end (revision 0), the slow one in the silence after (revision 1)."""
    words = "the quick brown fox jumps over the lazy dog"
    _plug(monkeypatch, Says(words, "fast"), Says(words, "slow"))
    _, events = _day(tmp_path, "day", voice)
    order = [(e["kind"], e.get("revision")) for e in events if e.get("kind") in ("partial", "text")]
    assert ("partial", None) in order and order.index(("partial", None)) < order.index(("text", 0))
    assert order.index(("text", 0)) < order.index(("text", 1))
    final = [e for e in _texts(events) if e["revision"] == 1]
    assert final and all(e["text"] == words and e["doubt"] in ("clear", "unseen") for e in final)
    assert all(w["agreed"] for e in final for w in e["words"])
    assert not [e for e in events if e.get("kind") == "question"]
    assert all(isinstance(port.parse(e), port.Text) for e in final)


def test_a_doubt_is_asked_answered_once_and_not_asked_again(tmp_path, voice, monkeypatch):
    """W3 and W4, the paradigm for words: the recognisers disagree on one name, the
    ear says it does not know and asks; KπX answers once; the next time the same
    voice says it, the ear writes what KπX said, and asks nothing."""
    _plug(
        monkeypatch,
        Says("send the notes to pi x before noon", "fast"),
        Says("send the notes to pie ex before noon", "slow"),
    )
    memory = tmp_path / "memory"
    ear, events = _day(tmp_path, "day1", voice, memory)
    questions = [e for e in events if e.get("kind") == "question" and e["about"] == "text"]
    assert questions, [(e["text"], e["doubt"]) for e in _texts(events)]
    asked = questions[0]
    assert asked["doubt"] in ("ambiguous", "not_understood") and len(asked["choices"]) >= 2
    texts = _texts(events)
    doubtful = [w for w in texts[-1]["words"] if not w["agreed"]]
    assert doubtful and all(w["trust"] < 0.9 for w in doubtful)

    assert [q.id for q in port.open_questions(ear.places.events)] == [asked["id"]]
    port.Inbox(memory).answer(asked["id"], text="send the notes to KπX before noon")
    ear.run(drain=True)  # the running ear obeys at its next tick
    after = [e for e in read_jsonl(ear.places.events)[0] if e.get("session") == ear.session]
    applied = [e for e in after if e.get("kind") == "correction"]
    assert applied and applied[-1]["what"] == "text" and applied[-1]["refused"] is None
    assert applied[-1]["answers"] == asked["id"]
    assert not port.open_questions(ear.places.events)
    revised = [e for e in _texts(after) if e["run"] == asked["run"]][-1]
    assert said(revised["text"], "send the notes to KπX before noon"), revised

    ear2, events2 = _day(tmp_path, "day2", voice, memory)
    again = _texts(events2)[-1]
    assert said(again["text"], "send the notes to KπX before noon"), again
    assert not [e for e in events2 if e.get("kind") == "question" and e["about"] == "text"], [
        (e.get("revision"), e.get("text"), e.get("doubt"), e["kind"])
        for e in events2
        if "row" not in e
    ]


def test_an_answer_undone_is_forgotten(tmp_path, voice, monkeypatch):
    _plug(
        monkeypatch,
        Says("call pi x now", "fast"),
        Says("call pie ex now", "slow"),
    )
    memory = tmp_path / "memory"
    ear, events = _day(tmp_path, "day1", voice, memory)
    run = _texts(events)[-1]
    inbox = port.Inbox(memory)
    inbox.correct("text", run["rows"], text="call KπX now")
    ear.run(drain=True)
    ear2, events2 = _day(tmp_path, "day2", voice, memory)
    assert said(_texts(events2)[-1]["text"], "call KπX now")
    lines, _ = read_jsonl(memory / "corrections.jsonl")
    inbox.undo(len(lines))
    ear2.run(drain=True)
    _, events3 = _day(tmp_path, "day3", voice, memory)
    assert not said(_texts(events3)[-1]["text"], "call KπX now")


def test_a_text_said_sessions_ago_can_still_be_corrected(tmp_path, voice, monkeypatch):
    """The scribe keeps every reading in ``runs.jsonl``: a correction aimed at a
    run of an earlier session finds it there."""
    _plug(monkeypatch, Says("meet me at the gare de lyon", "fast"))
    memory = tmp_path / "memory"
    _, events = _day(tmp_path, "day1", voice, memory)
    old = _texts(events)[-1]
    ear2, _ = _day(tmp_path, "day2", voice, memory)
    port.Inbox(memory).correct("text", old["rows"], text="meet me at the Gare de Lyon station")
    ear2.run(drain=True)
    after = [e for e in read_jsonl(ear2.places.events)[0] if e.get("session") == ear2.session]
    applied = [e for e in after if e.get("kind") == "correction"][-1]
    assert applied["refused"] is None
    revised = [e for e in _texts(after) if e["run"] == old["run"]]
    assert revised and said(revised[-1]["text"], "meet me at the Gare de Lyon station")


def test_a_voice_heard_again_unnamed_is_asked_for_its_name_once(tmp_path, voice, monkeypatch):
    """Q5: a voice of an earlier session, back and still nameless: one question.
    The answer names it, and a replay of the memory gives the same name."""
    _plug(monkeypatch, None)
    memory = tmp_path / "memory"
    _, events1 = _day(tmp_path, "day1", voice, memory)
    assert not [e for e in events1 if e.get("kind") == "question"]
    ear, events2 = _day(tmp_path, "day2", voice, memory)
    asked = [e for e in events2 if e.get("kind") == "question" and e["about"] == "name"]
    assert len(asked) == 1
    port.Inbox(memory).answer(asked[0]["id"], text="Alice")
    ear.run(drain=True)
    assert [v.label for v in ear.model.voices.values() if v.key == asked[0]["speaker"]] == ["Alice"]
    from π_ear.ear import rebuild

    replayed = rebuild(ear.places, ear.applied)
    assert {v.label for v in replayed.voices.values()} == {"Alice"}
    _, events3 = _day(tmp_path, "day3", voice, memory)
    assert not [e for e in events3 if e.get("kind") == "question"]
    assert {e["label"] for e in events3 if e.get("kind") == "known"} == {"Alice"}
