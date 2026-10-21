"""The reader of *what*, on readings whose truth is known. No model, no download."""

from __future__ import annotations

import random

import numpy as np
import pytest

from π_ear.asr import Hypothesis, Word
from π_ear.language import Languages
from π_ear.lexicon import Memory, align, edits, similarity, tokens
from π_ear.scribe import rebuild_reader
from π_ear.words import FEATURES, NONE, Context, Heard, Reader, Reading, _fit, fit_reader, repeated

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
        reader.languages.heard("v1", "fr", text, True, 4.0)
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
    # what the stream taught without a correction is not forgotten with it
    reader.languages.heard("v1", "fr", FR[0], True, 9.0)
    reader.agree(hyp("call the team", -0.1, "fast"), hyp("call the team", None, "slow"), "v1")
    reader.heard("le match le match le match le match")
    kept = rebuild_reader(tmp_path, base, undone={1}, current=reader)
    assert kept.read(heard, again, context)["text"] != "call kpihx now"
    assert kept.languages.detections == reader.languages.detections
    reader.languages.refit()
    assert kept.languages.of_voice("v1") == reader.languages.of_voice("v1")
    assert kept.repeats == reader.repeats
    assert kept.memory.known("v1", ["team"]) == 1 and kept.memory.known("v1", ["kpihx"]) == 0
    assert kept.memory.lexicon["v1"]["call"] == 1  # confirmed once, taught once: one left


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
        languages.heard("tv", "fr", text, True, 5.0)
    assert max(languages.of_voice("new").items(), key=lambda kv: kv[1])[0] == "fr"
    for text in EN * 3:
        languages.heard("guest", "en", text, True, 5.0)
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
            "v1", "fr" if right else str(rng.choice(["ru", "pt", "zh"])), FR[0], True, seconds
        )
    languages.refit()
    assert languages.detection_right(1.0) < languages.detection_right(15.0)
    assert languages.choose("v1", 1.0) == "fr"
    spoken = languages.of_voice("v1")["fr"]
    assert (languages.choose("v1", 60.0) is None) == (languages.detection_right(60.0) >= spoken)


def test_a_detection_is_judged_without_itself():
    """A room of French television; each guest is heard once, on a short phrase, and
    Whisper names another language every time. Judged against the voice's language
    *including* that detection, each was right (the voice spoke only it), and short
    detections came out reliable: they are judged against the voice without them."""
    languages = Languages()
    for _ in range(20):
        languages.heard("tv", "fr", FR[0], True, 20.0)
    for i, language in enumerate(["ru", "pt", "zh", "ja", "tr"] * 4):
        languages.heard(f"guest{i}", language, FR[1], True, 1.0)
    languages.refit()
    assert languages.detection_right(1.0) < 1 - languages.detection_right(1.0)
    assert languages.detection_right(20.0) > languages.detection_right(1.0)


def test_the_language_of_a_text_is_read_from_the_words_written_in_it():
    languages = Languages()
    for text in FR:
        languages.heard(None, "fr", text, False, 3.0)
    for text in EN:
        languages.heard(None, "en", text, False, 3.0)
    assert languages.of_text("le rapport avant la fin")["fr"] > 0.9
    assert languages.of_text("the report before the end")["en"] > 0.9
    assert languages.of_text("") is None


def test_languages_round_trip():
    languages = Languages()
    languages.heard("v1", "fr", FR[0], True, 2.0)
    again = Languages()
    again.load(languages.to_dict())
    languages.refit()
    assert again.of_voice("v1") == languages.of_voice("v1")
    assert again.detections == languages.detections
    # v2.1's first build saved a weight per detection and its counts: recounted
    old = {**languages.to_dict(), "detections": [["v1", "fr", 2.0, 0.8]], "voices": {"v1": {}}}
    again.load(old)
    assert again.detections == [("v1", "fr", 2.0)] and again.of_voice("v1")["fr"] > 0
    assert again.of_text(FR[1]) == languages.of_text(FR[1])


def test_a_language_told_is_no_evidence_of_the_language_spoken():
    """Measured on the live corpus: one short phrase of KπX misdetected as English, then
    every reading forced into English counted as English, and the voice was locked
    there. A told reading teaches words, never what a voice speaks."""
    languages = Languages()
    for _ in range(30):
        languages.measured(1.5, False)
        languages.measured(12.0, True)
    languages.heard("kpihx", "en", "i'm so happy to have you", True, 1.5)
    for _ in range(20):
        languages.heard("kpihx", "en", "and function in english", False, 1.5)
    for _ in range(6):
        languages.heard("kpihx", "fr", "je sais pas comment expliquer", True, 1.5)
    languages.refit()
    assert sum(d[:2] == ("kpihx", "en") for d in languages.detections) == 1
    spoken = languages.of_voice("kpihx")
    assert spoken["fr"] > spoken["en"]
    assert languages.choose("kpihx", 1.5) == "fr"  # detection at 1.5 s is the worse bet here


def test_a_reading_that_loops_is_not_speech(base):
    """Measured on the live replay: Whisper's loop "la Céline de la Céline de..." was
    accepted. Speech almost never repeats a word trigram, a loop any share of them;
    both are learned, unmixed, from what the recognisers write."""
    reader = Reader(base)
    for text in FR * 3 + [" ".join(["on y va"] * 6)]:
        reader.heard(text)
    loop = " ".join(["la céline de"] * 8)
    normal = "le premier tour de la céline"
    context = Context(None, TV, TV, 2.4)
    fast, slow = hyp(normal, -0.3, "fast"), hyp(loop, None, "slow")
    decided = reader.read(fast, slow, context)
    assert decided["text"] != loop, decided["readings"]  # the loop is never written
    readings, _ = reader.matrix(fast, slow, context)
    terms = dict(zip([r.text for r in readings], reader.repetition_terms(readings), strict=False))
    assert -0.1 < terms[normal] <= 0 and terms[loop] < -10


PLAYER = (
    "Non, c'est vrai que chaque jour, avant chaque match, le coach donne la corderie. C'est à"
    " ce moment-là où il m'a dit que j'avais débuté le match. Comme j'ai toujours fait,"
    " j'essaie toujours de donner mon maximum sur le terrain. Des fois, c'est vrai que tu as"
    " des meilleures soirées que d'autres, mais en tout cas, cette envie de tout donner sur le"
    " terrain, je n'ai jamais perdu ça."
)


def test_long_speech_that_repeats_itself_naturally_is_not_a_loop(base):
    """Measured on the live replay: a player's 67-trigram answer repeats "c'est vrai que"
    and "sur le terrain"; scored as Poisson against its mode with the base's rate (short
    sentences: 0.0014), it paid 5.4 nats and lost to "I did not understand". Against a
    loop, before the stream taught anything, it costs a fraction of a nat."""
    reader = Reader(base)
    assert repeated(PLAYER) == (2, 67)
    readings = [Reading(PLAYER, {"slow"}), Reading(" ".join(["la céline de"] * 8), {"slow"})]
    natural, loop, _ = reader.repetition_terms(readings)
    assert natural > -0.5 and loop < -10


def test_a_voice_misdetected_in_its_regime_is_told_the_language_of_the_room():
    """The live replay, reduced: the television's voices detected French on long groups,
    KπX's short phrases over it detected French 6 times in 18 and Portuguese, English,
    Chinese, Russian, Polish or Turkish otherwise. Counted as spoken, those made KπX
    35 % French and Whisper was never told (the base says detection at 2 s is right 6
    times in 7: on clean speech). Learned per voice, KπX's detections are right one
    time in three, the room's voices speak its language, and French is given; the
    television keeps detecting; a guest who really speaks English is heard so."""
    languages = Languages()
    for _ in range(45):
        languages.measured(2.0, True)  # the base: clean speech
    languages.measured(2.0, False)
    for voice in ("tv1", "tv2", "tv3"):
        for _ in range(8):
            languages.heard(voice, "fr", FR[0], True, 25.0)
    for language in ["fr"] * 6 + ["pt", "pt", "pt", "en", "en", "en", "zh", "zh", "ru", "ru"]:
        languages.heard("kpihx", language, FR[1], True, 2.0)
    languages.heard("kpihx", "pl", FR[1], True, 1.5)
    languages.heard("kpihx", "tr", FR[1], True, 4.0)
    languages.refit()
    assert languages.detection_right(2.0, "kpihx") < languages.detection_right(2.0, "tv1")
    assert languages.choose("kpihx", 2.0) == "fr"
    assert languages.choose("tv1", 25.0) is None
    assert max(languages.of_voice("never-heard"), key=languages.of_voice("never-heard").get) == "fr"
    for _ in range(12):
        languages.heard("guest", "en", EN[0], True, 25.0)
    languages.refit()
    spoken = languages.of_voice("guest")
    assert max(spoken, key=spoken.get) == "en"
    assert languages.choose("guest", 2.0) in ("en", None)
