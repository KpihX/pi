"""The reader of *what*, on readings whose truth is known. No model, no download."""

from __future__ import annotations

import random

import numpy as np
import pytest

from π_ear.asr import Hypothesis, Word
from π_ear.lexicon import Memory, align, edits, similarity, tokens
from π_ear.scribe import rebuild_reader
from π_ear.words import NONE, Reader, fit_reader

SENTENCES = [
    "the meeting moved to thursday because nobody found the projector",
    "could you send me the report before the end of the afternoon",
    "remember to turn off the oven when the bread smells good",
    "we lost the signal between the bridge and the old factory",
    "je vais chercher le cours de mathématiques sur mon téléphone",
    "la réunion a été déplacée parce que personne ne trouvait le projecteur",
    "il faut éteindre le four quand le pain commence à sentir bon",
    "nous avons perdu le signal entre le pont et la vieille usine",
]
JUNK = "uh ah mm the oh hm".split()


def hyp(text: str, logp: float, source: str) -> Hypothesis:
    words = tuple(Word(w, i * 0.3, i * 0.3 + 0.3, logp) for i, w in enumerate(text.split()))
    return Hypothesis(text, words, None, source)


def garble(text: str, rng: random.Random, rate: float) -> str:
    words = text.split()
    return " ".join(rng.choice(JUNK) if rng.random() < rate else w for w in words)


def public_runs(seed: int = 0) -> list:
    """Speech heard by two imperfect recognisers, and sounds where nothing was said."""
    rng = random.Random(seed)
    runs = []
    for _ in range(6):
        for truth in SENTENCES:
            fast = garble(truth, rng, 0.08)
            slow = garble(truth, rng, 0.05)
            runs.append(
                (
                    hyp(fast, -0.05 if fast == truth else -0.6, "fast"),
                    hyp(slow, None, "slow"),
                    truth,
                    rng.uniform(0.8, 0.95),
                )
            )
    for _ in range(12):
        noise = " ".join(rng.choice(JUNK) for _ in range(rng.randint(1, 3)))
        other = " ".join(rng.choice(JUNK) for _ in range(rng.randint(1, 3)))
        runs.append((hyp(noise, -1.5, "fast"), hyp(other, None, "slow"), "", rng.uniform(0.1, 0.8)))
    return runs


@pytest.fixture(scope="module")
def base():
    return fit_reader(public_runs())


def test_text_utilities():
    assert tokens("Élève, l’école !") == ["eleve", "l'ecole"]
    assert [op for op, _, _ in align(["a", "b", "c"], ["a", "x", "c", "d"])] == ["=", "~", "=", "+"]
    assert edits(tokens("call pi x now"), tokens("call kpihx now")) == [("pi x", "kpihx")]
    assert similarity(["a", "b"], ["a", "b"]) == 1 and similarity(["a"], ["z"]) == 0


def test_two_recognisers_agreeing_is_clear(base):
    reader = Reader(base)
    truth = SENTENCES[0]
    decided = reader.read(hyp(truth, -0.05, "fast"), hyp(truth, None, "slow"), None)
    assert decided["text"] == truth and decided["doubt"] in ("clear", "unseen")
    assert all(w["agreed"] for w in decided["words"])


def test_disagreement_is_named_and_both_readings_are_kept(base):
    reader = Reader(base)
    decided = reader.read(
        hyp("we lost the signal between the bridge and the old factory", -0.1, "fast"),
        hyp("we lost the signal between the ridge and the old factory", None, "slow"),
        None,
    )
    assert decided["doubt"] == "ambiguous"
    texts = [r["text"] for r in decided["readings"]]
    assert any("bridge" in t for t in texts) and any("ridge" in t for t in texts) and NONE in texts
    disputed = [w for w in decided["words"] if not w["agreed"]]
    agreed = [w for w in decided["words"] if w["agreed"]]
    assert disputed and max(w["trust"] for w in disputed) < min(w["trust"] for w in agreed)


def test_noise_is_not_understood(base):
    reader = Reader(base)
    decided = reader.read(hyp("hm oh", -1.6, "fast"), hyp("uh", None, "slow"), None)
    assert decided["doubt"] == "not_understood" and decided["text"] == ""


def test_an_error_corrected_once_does_not_come_back(base):
    """The second law, for words: KπX says once what was said, and the next time
    the recognisers make the same mistake, the reader writes the truth."""
    reader = Reader(base)
    heard = hyp("send the notes to pi x before noon", -0.3, "fast")
    again = hyp("send the notes to pi x before noon", None, "slow")
    truth = "send the notes to kpihx before noon"
    assert reader.read(heard, again, "v1")["text"] != truth
    reader.learn(heard, again, "v1", truth)
    reader.refit()
    assert reader.read(heard, again, "v1")["text"] == truth
    # and a new sentence with the same mistake inside it
    other = hyp("ask pi x about the meeting", -0.3, "fast")
    decided = reader.read(other, hyp("ask pi x about the meeting", None, "slow"), "v1")
    assert decided["text"] == "ask kpihx about the meeting"


def test_the_reader_round_trips_and_rebuilds_from_what_it_was_taught(base, tmp_path):
    from π_ear.io import append_jsonl

    reader = Reader(base)
    heard, again = hyp("call pi x now", -0.3, "fast"), hyp("call pi x now", None, "slow")
    reader.learn(heard, again, "v1", "call kpihx now")
    reader.refit()
    append_jsonl(
        tmp_path / "taught.jsonl",
        {
            "line": 1,
            "run": "r",
            "voice": "v1",
            "fast": heard.to_dict(),
            "slow": again.to_dict(),
            "truth": "call kpihx now",
        },
    )
    clone = Reader(base)
    clone.load(reader.to_dict())
    rebuilt = rebuild_reader(tmp_path, base, undone=set())
    for other in (clone, rebuilt):
        assert np.allclose(other.weights, reader.weights)
        assert other.read(heard, again, "v1")["text"] == "call kpihx now"
    forgotten = rebuild_reader(tmp_path, base, undone={1})
    assert forgotten.read(heard, again, "v1")["text"] != "call kpihx now"


def test_memory_recalls_only_the_same_voice():
    memory = Memory()
    memory.teach("v1", "le cours de gallois", "le cours de Galois", ["le cours de gallois"])
    assert memory.recall("v1", "le cours de gallois") and not memory.recall(
        "v2", "le cours de gallois"
    )
    assert memory.known("v1", ["galois"]) == 1 and memory.known("v2", ["galois"]) == 0
