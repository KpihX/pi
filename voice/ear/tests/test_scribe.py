"""The *what*, inside the running ear: partials, revisions, questions, answers.

The real loop (Silero, CAM++, the model of who, the scribe's thread), with
recognisers plugged in by the registry, the way any other recogniser plugs in.
They say what they are told to say, so what is tested is the ear, not a model:
when the text is written, what it says it does not know, what it asks, and that
an answer, once given, is not needed again.
"""

from __future__ import annotations

import time

import numpy as np
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
    """W3 and W4, the paradigm for words: the recognisers disagree on one name in a
    sentence they otherwise share; the ear is sure of the sentence, not of the name,
    and asks about the name; KπX answers once; the next time the same voice says it,
    the ear writes what KπX said, and asks nothing."""
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
    assert len(asked["choices"]) >= 2 and set(asked["words"]) <= {"pi", "x", "pie", "ex"}
    assert asked["words"], asked
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
    revised = [e for e in _texts(after) if e["piece"] == asked["piece"]][-1]
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


def test_an_answer_that_says_nothing_is_refused_not_guessed(tmp_path, voice, monkeypatch):
    """An answer with neither a text nor a choice (a mouth that heard nothing back) is
    refused, and the question stays open; "" is an answer: nothing was said."""
    _plug(
        monkeypatch,
        Says("send the notes to pi x before noon", "fast"),
        Says("send the notes to pie ex before noon", "slow"),
    )
    memory = tmp_path / "memory"
    ear, events = _day(tmp_path, "day1", voice, memory)
    asked = next(e for e in events if e.get("kind") == "question" and e["about"] == "text")
    port.Inbox(memory).answer(asked["id"])
    ear.run(drain=True)
    applied = [e for e in read_jsonl(ear.places.events)[0] if e.get("kind") == "correction"]
    assert applied[-1]["refused"] == "an answer gives a text or a choice"
    assert [q.id for q in port.open_questions(ear.places.events)] == [asked["id"]]
    port.Inbox(memory).answer(asked["id"], text="")
    ear.run(drain=True)
    applied = [e for e in read_jsonl(ear.places.events)[0] if e.get("kind") == "correction"]
    assert applied[-1]["refused"] is None and not port.open_questions(ear.places.events)


def test_a_text_said_sessions_ago_can_still_be_corrected(tmp_path, voice, monkeypatch):
    """The scribe keeps every reading in ``pieces.jsonl``: a correction aimed at a
    piece of an earlier session finds it there."""
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
    revised = [e for e in _texts(after) if e["piece"] == old["piece"]]
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


# -- the scribe alone: pieces, groups, priorities ------------------------------------------


class Timed:
    """A recogniser that writes one word per half second of what it is given, named by
    the absolute second it was said at: where each word lands can be checked exactly."""

    def __init__(self, name: str, window_s: float | None = None, start: float = 0.0):
        self.name, self.window_s, self.calls = name, window_s, []

    def __call__(self, samples, language=None):
        n = len(samples) / contract.RATE
        self.calls.append((n, language))
        words = tuple(Word(f"w{i}", i * 0.5, i * 0.5 + 0.5, -0.1) for i in range(int(n / 0.5)))
        return Hypothesis(" ".join(w.text for w in words), words, "fr", self.name)


def _scribe(tmp_path, fast, slow, source=None):
    from π_ear.scribe import Scribe
    from π_ear.words import Reader

    written = []
    scribe = Scribe(
        Reader(),
        fast,
        slow,
        None,
        written.append,
        tmp_path,
        contract.MODELS,
        source_of=source or (lambda voice: ("device", 1.0)),
        floor_s=0.25,
    )
    return scribe, written


def _decisions(seconds: float, voices: list[str], start: float = 0.0) -> list[dict]:
    """Learned 2 s windows, the voice changing every ``seconds / len(voices)``."""
    out, row, t = [], 0, 0.0
    while t + 2 <= seconds:
        voice = voices[min(int(t / (seconds / len(voices))), len(voices) - 1)]
        out.append(
            {
                "row": row,
                "t0": start + t,
                "t1": start + t + 2,
                "kind": "known",
                "speaker": voice,
                "learned": True,
            }
        )
        row, t = row + 1, t + 2
    return out


def test_a_long_run_is_cut_under_the_window_and_where_the_voice_changes(tmp_path, voice):
    fast, slow = Timed("fast"), Timed("slow", window_s=12.0)
    scribe, written = _scribe(tmp_path, fast, slow)
    audio = np.concatenate([voice, voice])[: 24 * contract.RATE]
    scribe.finish("r", 0.0, audio, _decisions(24, ["a", "b"]))
    scribe.drain()
    texts = {}
    for e in written:
        if e.get("kind") == "text":
            texts[e["piece"]] = e
    assert texts and all(0.25 <= e["t1"] - e["t0"] < 12.0 for e in texts.values())
    assert any(abs(e["t1"] - 12.0) <= 1.0 for e in texts.values()), [
        (e["t0"], e["t1"]) for e in texts.values()
    ]
    assert {e["speaker"] for e in texts.values()} == {"a", "b"}
    # read slowly together, as much as the window holds, never more
    assert all(n < 12.0 for n, _ in slow.calls) and len(slow.calls) <= len(texts)


@pytest.mark.parametrize("trust, cut", [(0.0, False), (None, True), (0.9, True)])
def test_a_voice_change_is_a_cut_only_if_more_probably_real(tmp_path, voice, trust, cut):
    """The cut between two voices is gated on the model of who's calibrated trust; a
    trust of 0 is a trust (a first build read ``trust or 1`` as sure); a window the
    model could not calibrate (None) keeps its argmax."""
    fast, slow = Timed("fast"), Timed("slow", window_s=12.0)
    scribe, written = _scribe(tmp_path, fast, slow)
    decisions = [{**d, "trust": trust} for d in _decisions(8, ["a", "b"])]
    scribe.finish("r", 0.0, voice[: 8 * contract.RATE], decisions)
    scribe.drain()
    pieces = {e["piece"] for e in written if e.get("kind") == "text"}
    assert (len(pieces) > 1) == cut, pieces


def test_a_piece_lets_go_of_its_audio_once_read_by_both(tmp_path, voice):
    """Every piece kept its samples for the whole session (460 MB on two hours)."""
    fast, slow = Timed("fast"), Timed("slow", window_s=12.0)
    scribe, _ = _scribe(tmp_path, fast, slow)
    scribe.finish("r", 0.0, voice[: 8 * contract.RATE], _decisions(8, ["a", "b"]))
    scribe.drain()
    assert scribe.pieces and all(p.samples is None for p in scribe.pieces.values())
    alone, _ = _scribe(tmp_path / "alone", Timed("fast"), None)
    alone.finish("r", 0.0, voice[: 4 * contract.RATE], _decisions(4, ["a"]))
    alone.drain()
    assert alone.pieces and all(p.samples is None for p in alone.pieces.values())


@pytest.mark.parametrize("kind, asked", [("known", True), ("new", False)])
def test_a_voice_born_in_the_piece_is_not_asked_about(tmp_path, voice, kind, asked):
    fast, slow = Says("send the notes to pi x", "fast"), Says("send the notes to pie ex", "slow")
    scribe, written = _scribe(tmp_path, fast, slow, source=lambda v: ("person", 1.0))
    decisions = [{**d, "kind": kind} for d in _decisions(4, ["kpihx"])]
    scribe.finish("r", 0.0, voice[: 4 * contract.RATE], decisions)
    scribe.drain()
    assert bool([e for e in written if e.get("kind") == "question"]) == asked


def test_a_refit_that_fails_is_an_error_written_not_a_loop(tmp_path, voice):
    """A first build of the language model raised in its refit; the reader stayed
    dirty, the scribe refitted again at once, forever, and ``drain`` never returned."""
    fast, slow = Timed("fast"), Timed("slow", window_s=8.0)
    scribe, written = _scribe(tmp_path, fast, slow)

    def broken():
        raise ValueError("a refit that fails")

    scribe.reader.refit = broken
    scribe.reader.dirty = True
    scribe.finish("r", 0.0, voice[: 3 * contract.RATE], _decisions(3, ["a"]))
    scribe.drain()
    errors = [e for e in written if e.get("kind") == "error"]
    assert len(errors) == 1 and errors[0]["job"] == "refit"
    assert [e for e in written if e.get("kind") == "text"]


def test_short_pieces_are_read_slowly_together(tmp_path, voice):
    """Six voices' turns of 2 s: six pieces, read by the slow recogniser in groups as
    long as its window, then each given back its own words."""
    fast, slow = Timed("fast"), Timed("slow", window_s=12.0)
    scribe, written = _scribe(tmp_path, fast, slow)
    scribe.finish("r", 0.0, voice[: 12 * contract.RATE], _decisions(12, list("abcdef")))
    scribe.drain()
    pieces = {e["piece"] for e in written if e.get("kind") == "text"}
    assert len(pieces) >= 4 and len(slow.calls) < len(pieces), (len(pieces), slow.calls)
    assert all(n < 12.0 for n, _ in slow.calls)


def test_each_piece_gets_the_slow_words_said_during_it(tmp_path, voice):
    fast, slow = Timed("fast"), Timed("slow", window_s=8.0)
    scribe, written = _scribe(tmp_path, fast, slow)
    scribe.finish("r", 0.0, voice[: 7 * contract.RATE], _decisions(7, ["a", "b"]))
    scribe.drain()
    final = {}
    for e in written:
        if e.get("kind") == "text" and e["revision"] == 1:
            final[e["piece"]] = e
    pieces = sorted(final.values(), key=lambda e: e["t0"])
    assert len(pieces) >= 2
    # every word the slow recogniser wrote went to exactly one piece, in order
    slow_words = [
        w
        for e in pieces
        for r in e["readings"]
        if "slow" in r["origins"]
        for w in r["text"].split()
    ]
    assert slow_words == [f"w{i}" for i in range(len(slow_words))] or all(
        sum("slow" in r["origins"] for r in e["readings"]) <= 1 for e in pieces
    )


def test_people_are_read_first_and_never_while_a_person_speaks(tmp_path, voice):
    slow = Timed("slow", window_s=8.0)
    people = {"kpihx"}
    scribe, written = _scribe(
        tmp_path, Timed("fast"), slow, lambda v: ("person" if v in people else "device", 1.0)
    )
    scribe.set_speaking(True, person=True)  # KπX is speaking: the slow pass waits
    for n, who in enumerate(["tv", "tv", "kpihx"]):
        scribe.finish(f"r{n}", n * 10.0, voice[: 3 * contract.RATE], _decisions(3, [who], n * 10.0))
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not any(
        e.get("revision") == 0 for e in written if e.get("speaker") == "kpihx"
    ):
        time.sleep(0.05)
    assert not slow.calls, "the slow pass ran while a person spoke"
    scribe.set_speaking(False)
    scribe.drain()
    first = next(e for e in written if e.get("kind") == "text" and e["revision"] == 1)
    assert first["speaker"] == "kpihx"


def test_revisions_go_on_while_a_television_talks_without_a_pause(tmp_path, voice):
    """Live, a run grows every second; a first build always read the partial first and
    revised nothing for as long as the television spoke. Partials and revisions take
    turns now (never while a person speaks: the previous test)."""
    slow = Timed("slow", window_s=8.0)
    scribe, written = _scribe(tmp_path, Timed("fast"), slow)
    scribe.finish("before", 0.0, voice[: 3 * contract.RATE], _decisions(3, ["tv"]))
    long = np.concatenate([voice] * 4)[: 40 * contract.RATE]
    scribe.set_speaking(True, person=False)
    for second in range(4, 40):  # the run in progress, one second more each time
        scribe.hear(
            "long", 4.0, long[: (second - 3) * contract.RATE], _decisions(second - 3, ["tv"], 4.0)
        )
        time.sleep(0.05)
    revised = [e for e in written if e.get("kind") == "text" and e["revision"] == 1]
    assert slow.calls and revised, "nothing was revised while the television talked"
    assert any(e.get("kind") == "partial" for e in written)
    scribe.finish("long", 4.0, long, _decisions(40, ["tv"], 4.0))
    scribe.set_speaking(False)
    scribe.drain()
