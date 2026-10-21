"""The reader of *what*, on readings whose truth is known. No model, no download."""

from __future__ import annotations

import random

import numpy as np
import pytest

from π_ear.asr import Hypothesis, Word
from π_ear.language import Languages
from π_ear.lexicon import Memory, align, edits, similarity, tokens
from π_ear.scribe import rebuild_reader
from π_ear.words import FEATURES, NONE, Context, Heard, Reader, _fit, fit_reader

EN = [
    "the meeting moved to thursday because nobody found the projector",
    "could you send me the report before the end of the afternoon",
    "remember to turn off the oven when the bread smells good",
    "we lost the signal between the bridge and the old factory",
]
FR = [
    "la réunion a été déplacée parce que personne ne trouvait le projecteur",
    "tu peux m'envoyer le rapport avant la fin de l'après-midi",
    "pense à éteindre le four quand le pain sent bon",
    "nous avons perdu le signal entre le pont et la vieille usine",
]
JUNK = "uh ah mm the oh hm".split()
CLEAN, TV = 0.88, 0.5  # Silero's mean on speech: clean speech, a television heard by a laptop


def hyp(text: str, logp: float, source: str, language: str | None = None) -> Hypothesis:
    words = tuple(Word(w, i * 0.3, i * 0.3 + 0.3, logp) for i, w in enumerate(text.split()))
    return Hypothesis(text, words, language, source)


def garble(text: str, rng: random.Random, rate: float) -> str:
    return " ".join(rng.choice(JUNK) if rng.random() < rate else w for w in text.split())


def public(seed: int = 0) -> list[Heard]:
    """Speech in two languages and two regimes, heard by two imperfect recognisers, and
    sounds where nothing was said. Speech is quieter to Silero on television: the
    reader must learn the gap to the regime, not a level."""
    rng = random.Random(seed)
    out = []
    for regime in (CLEAN, TV):
        for language, sentences in (("en", EN), ("fr", FR)):
            for k in range(3):
                voice = f"{language}{k}"
                for truth in sentences:
                    f, s = garble(truth, rng, 0.08), garble(truth, rng, 0.05)
                    speech = regime + rng.uniform(-0.06, 0.06)
                    out.append(
                        Heard(
                            hyp(f, -0.05 if f == truth else -0.6, "fast"),
                            hyp(s, None, "slow", language),
                            truth,
                            Context(voice, speech, regime),
                            language,
                        )
                    )
        for _ in range(12):
            noise = " ".join(rng.choice(JUNK) for _ in range(rng.randint(1, 3)))
            other = " ".join(rng.choice(JUNK) for _ in range(rng.randint(1, 3)))
            speech = regime - rng.uniform(0.2, 0.45)
            out.append(
                Heard(
                    hyp(noise, -1.5, "fast"),
                    hyp(other, None, "slow"),
                    "",
                    Context(None, speech, regime),
                )
            )
    return out


@pytest.fixture(scope="module")
def base():
    return fit_reader(public())


def french_reader(base) -> Reader:
    """A reader that has heard voice v1 speak French, as a live stream would."""
    reader = Reader(base)
    for text in FR:
        reader.languages.heard("v1", "fr", text, 1.0, True, 4.0)
    return reader


def test_text_utilities():
    assert tokens("Élève, l’école !") == ["eleve", "l'ecole"]
    assert [op for op, _, _ in align(["a", "b", "c"], ["a", "x", "c", "d"])] == ["=", "~", "=", "+"]
    assert edits(tokens("call pi x now"), tokens("call kpihx now")) == [("pi x", "kpihx")]
    assert similarity(["a", "b"], ["a", "b"]) == 1 and similarity(["a"], ["z"]) == 0


def test_two_recognisers_agreeing_is_not_a_doubt(base):
    decided = Reader(base).read(hyp(EN[0], -0.05, "fast"), hyp(EN[0], None, "slow"), Context())
    assert decided["text"] == EN[0] and decided["doubt"] in ("clear", "unseen")
    assert all(w["agreed"] for w in decided["words"])


def test_one_word_apart_in_a_long_sentence_is_not_ambiguous(base):
    """v2's first build named "ambiguous" any two readings one word apart: on live
    television, 2 texts out of 662 were not. The doubt is the reader's posterior now."""
    long = " ".join(EN[:3])
    other = long.replace("projector", "protector")
    decided = Reader(base).read(hyp(long, -0.1, "fast"), hyp(other, None, "slow"), Context())
    assert decided["doubt"] != "ambiguous", decided["p"]
    disputed = [w for w in decided["words"] if not w["agreed"]]
    agreed = [w for w in decided["words"] if w["agreed"]]
    assert disputed and max(w["trust"] for w in disputed) < min(w["trust"] for w in agreed)


def test_the_named_doubt_is_the_posterior_over_hundreds_of_pieces(base):
    """``ambiguous`` exactly when the reader cannot choose, ``partial`` when the best
    reading holds more probably-wrong words than right ones; every reading is kept,
    "I did not understand" among them; and the named doubt follows the garbling."""
    reader, rng, seen = Reader(base), random.Random(3), set()
    for _ in range(300):
        truth = rng.choice(EN + FR)
        fast = hyp(garble(truth, rng, rng.uniform(0, 0.7)), -rng.uniform(0, 2), "fast")
        slow = hyp(garble(truth, rng, rng.uniform(0, 0.7)), None, "slow")
        d = reader.read(fast, slow, Context(None, CLEAN - rng.uniform(0, 0.5), CLEAN))
        seen.add(d["doubt"])
        assert NONE in [r["text"] for r in d["readings"]]
        if d["text"] and d["doubt"] != "taught":
            assert (d["doubt"] == "ambiguous") == (d["p"] < 1 - d["p"])
            trust = np.mean([w["trust"] for w in d["words"]])
            if d["doubt"] != "ambiguous":
                assert (d["doubt"] == "partial") == (trust < 1 - trust)
        assert (d["doubt"] == "not_understood") == (d["text"] == "")
    assert (
        "not_understood" in seen and seen & {"partial", "ambiguous"} and seen & {"clear", "unseen"}
    )


def test_noise_is_not_understood(base):
    decided = Reader(base).read(
        hyp("hm oh", -1.6, "fast"), hyp("uh", None, "slow"), Context(None, CLEAN - 0.4, CLEAN)
    )
    assert decided["doubt"] == "not_understood" and decided["text"] == ""


def test_quiet_speech_is_speech_where_all_speech_is_quiet(base):
    """The same Silero mean (0.5) is little for clean speech and ordinary for a television
    heard by a laptop: the reader weighs it against the regime, not against a level."""
    reader = Reader(base)
    fast, slow = hyp(FR[3], -0.4, "fast"), hyp(FR[3], None, "slow")
    on_tv = reader.read(fast, slow, Context(None, TV, TV))
    assert on_tv["text"] == FR[3]
    p_clean = reader.read(fast, slow, Context(None, TV, CLEAN))["p"]
    assert p_clean < on_tv["p"]


def test_a_translation_loses_to_the_language_of_its_voice(base):
    """Measured live: Parakeet wrote English for French studio talk. When the voice is
    known to speak French, a French reading wins over an English one of equal standing."""
    reader = french_reader(base)
    fast = hyp(EN[1], -0.2, "fast")  # the fast recogniser translated
    slow = hyp(FR[1], None, "slow", "fr")
    decided = reader.read(fast, slow, Context("v1", TV, TV))
    assert decided["text"] == FR[1], decided["readings"]
    plain = Reader(base).read(fast, slow, Context("v1", TV, TV))  # knowing no language
    assert plain["text"] == EN[1]  # the fast recogniser's translation would have won


def test_an_error_corrected_once_does_not_come_back(base):
    """The second law, for words: KπX says once what was said, and the next time the
    recognisers make the same mistake, the reader writes the truth."""
    reader = Reader(base)
    heard, again = (
        hyp("send the notes to pi x before noon", -0.3, "fast"),
        hyp("send the notes to pi x before noon", None, "slow"),
    )
    truth = "send the notes to kpihx before noon"
    context = Context("v1", CLEAN, CLEAN)
    assert reader.read(heard, again, context)["text"] != truth
    reader.learn(Heard(heard, again, truth, context))
    reader.refit()
    decided = reader.read(heard, again, context)
    assert decided["text"] == truth and decided["doubt"] == "taught"
    other = hyp("ask pi x about the meeting", -0.3, "fast")
    decided = reader.read(other, hyp("ask pi x about the meeting", None, "slow"), context)
    assert decided["text"] == "ask kpihx about the meeting"


def test_the_reader_round_trips_and_rebuilds_from_what_it_was_taught(base, tmp_path):
    from π_ear.io import append_jsonl

    reader = Reader(base)
    heard, again = hyp("call pi x now", -0.3, "fast"), hyp("call pi x now", None, "slow")
    context = Context("v1", CLEAN, CLEAN)
    reader.learn(Heard(heard, again, "call kpihx now", context))
    reader.refit()
    append_jsonl(
        tmp_path / "taught.jsonl",
        {
            "line": 1,
            "piece": "r#0",
            "voice": "v1",
            "speech": CLEAN,
            "regime": CLEAN,
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
        assert other.read(heard, again, context)["text"] == "call kpihx now"
    forgotten = rebuild_reader(tmp_path, base, undone={1})
    assert forgotten.read(heard, again, context)["text"] != "call kpihx now"


def test_weights_saved_under_another_feature_list_are_kept_by_name():
    old = ["fast", "slow", "agree", "none"]
    moved = _fit(np.array([1.0, 2.0, 3.0, 4.0]), old)
    assert moved[FEATURES.index("agree")] == 3.0 and moved[FEATURES.index("none")] == 4.0
    assert "out_language" not in FEATURES
    reader = Reader()
    reader.load(
        {"features": old, "weights": [1.0, 2.0, 3.0, 4.0], "examples": [[[[1, 0, 1, 0]], 0]]}
    )
    assert reader.examples[0][0].shape == (1, len(FEATURES))


def test_a_saved_reader_is_refitted_on_the_base_it_runs_on_not_frozen(base):
    """Measured live: a memory saved under v2.0 reloaded v2.0's weights and ignored the
    base it ran on. What a memory keeps is what it learned (examples, lexicon,
    languages); the weights are always the base plus that, refitted."""
    old = Reader(base)
    old.weights = old.weights * 0 + 7.0  # whatever an older base had
    again = Reader(base)
    again.load(old.to_dict())
    assert np.allclose(again.weights, Reader(base).weights)


def test_the_lexicon_learns_the_words_both_recognisers_wrote_even_if_not_all_agree():
    reader = Reader()
    reader.agree(
        hyp("le cours de Galois est demain", -0.1, "fast"),
        hyp("le cours de gallois est demain", None, "slow"),
        "v1",
    )
    assert reader.memory.known("v1", ["cours", "demain"]) == 1
    assert reader.memory.known("v1", ["galois"]) == 0


def test_memory_recalls_only_the_same_voice():
    memory = Memory()
    memory.teach("v1", "le cours de gallois", "le cours de Galois", ["le cours de gallois"])
    assert memory.recall("v1", "le cours de gallois") and not memory.recall(
        "v2", "le cours de gallois"
    )
    assert memory.known("v1", ["galois"]) == 1 and memory.known("v2", ["galois"]) == 0


# -- languages ---------------------------------------------------------------------------


def test_a_voice_language_starts_where_the_room_is_and_follows_what_the_voice_says():
    languages = Languages()
    assert languages.of_voice("v1") is None and languages.choose("v1", 3.0) is None
    for text in FR:
        languages.heard("tv", "fr", text, 1.0, True, 5.0)
    assert max(languages.of_voice("new").items(), key=lambda kv: kv[1])[0] == "fr"
    for text in EN * 3:
        languages.heard("guest", "en", text, 1.0, True, 5.0)
    spoken = languages.of_voice("guest")
    assert spoken["en"] > spoken["fr"]


def test_the_language_is_given_only_when_it_is_more_probable_than_detection_being_right():
    """Detection on short pieces is often wrong (measured live: Russian, Portuguese,
    Chinese for KπX's French); on long ones it is right. The voice's language is given
    when it is the better bet, at this length."""
    languages = Languages()
    rng = np.random.default_rng(0)
    for _ in range(40):
        seconds = float(rng.uniform(0.5, 20))
        right = rng.uniform() < seconds / (seconds + 2)
        languages.heard(
            "v1", "fr" if right else str(rng.choice(["ru", "pt", "zh"])), FR[0], 1.0, True, seconds
        )
    languages.refit()
    assert languages.detection_right(1.0) < languages.detection_right(15.0)
    assert languages.choose("v1", 1.0) == "fr"
    spoken = languages.of_voice("v1")["fr"]
    assert (languages.choose("v1", 60.0) is None) == (languages.detection_right(60.0) >= spoken)


def test_the_language_of_a_text_is_read_from_the_words_written_in_it():
    languages = Languages()
    for text in FR:
        languages.heard(None, "fr", text, 1.0, False, 3.0)
    for text in EN:
        languages.heard(None, "en", text, 1.0, False, 3.0)
    assert languages.of_text("le rapport avant la fin")["fr"] > 0.9
    assert languages.of_text("the report before the end")["en"] > 0.9
    assert languages.of_text("") is None


def test_languages_round_trip():
    languages = Languages()
    languages.heard("v1", "fr", FR[0], 0.8, True, 2.0)
    again = Languages()
    again.load(languages.to_dict())
    assert again.of_voice("v1") == languages.of_voice("v1")
    assert again.of_text(FR[1]) == languages.of_text(FR[1])


def test_a_language_told_is_no_evidence_of_the_language_spoken():
    """Measured on the live corpus: one short phrase of KπX misdetected as English, then
    every reading forced into English counted as English, and his voice was locked
    there. A told reading teaches words, never what a voice speaks."""
    languages = Languages()
    for _ in range(30):
        languages.measured(1.5, False)
        languages.measured(12.0, True)
    languages.heard("kpihx", "en", "i'm so happy to have you", 1.0, True, 1.5)
    for _ in range(20):
        languages.heard("kpihx", "en", "and function in english", 1.0, False, 1.5)
    for _ in range(6):
        languages.heard("kpihx", "fr", "je sais pas comment expliquer", 1.0, True, 1.5)
    languages.refit()
    assert languages.voices["kpihx"]["en"] == 1.0
    spoken = languages.of_voice("kpihx")
    assert spoken["fr"] > spoken["en"]
    assert languages.choose("kpihx", 1.5) == "fr"  # detection at 1.5 s is the worse bet here
