"""Which language a voice speaks, learned, and when to tell the recognisers.

Measured live (2026-10-20, PSG - Barça on a laptop microphone): Whisper's own language
detection wrote English, Russian, Portuguese, Turkish, Japanese or Chinese on 15 % of
the runs of a French broadcast, and on most of KπX's own short phrases; Parakeet, which
cannot be told a language, answered in English on studio talk (2 % of the words on 4 s
pieces, 19 % on 8 s, 77 % on 32 s). A language is a property of a voice, not of one run.

Three things are learned here, nothing is set:

* per voice, a Dirichlet over languages: what its slow readings were *detected* in (a
  reading the recogniser was told the language of proves nothing), each weighted by the
  probability that a detection that long is right; its prior is the room (one
  pseudo-reading), itself a
  Dirichlet over every voice with one pseudo-reading spread over the known languages;
* how often Whisper's detection is right, as a function of the run's length (short
  runs fool it): a Bayesian logit on the runs it detected alone, against the language
  their voice turned out to speak;
* what words each language is written with (the slow recogniser's words, by the
  language it wrote them in), to say in which language a reading *is*: a Parakeet
  reading in English for a French voice is a translation, and the reader is told.

The decision has no threshold: the voice's language is given to the recogniser when
it is more probable than the detection being right at this length.

The law: no numeric literal but 0 and 1 in this module (``tests/test_law.py``).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

from π_ear.contract import TEXT_GRAM
from π_ear.lexicon import tokens
from π_ear.sphere import choice_probabilities, fit_choice

OTHER = "?"  # a language this ear has never written


def trigrams(text: str) -> list[str]:
    """The character trigrams of a text's words, each word framed by spaces."""
    out = []
    for word in tokens(text):
        framed = f" {word} "
        out += [framed[i : i + TEXT_GRAM] for i in range(len(framed) - TEXT_GRAM + 1)]
    return out


class Languages:
    def __init__(self):
        self.voices: dict[str, Counter] = defaultdict(Counter)
        self.words: dict[str, Counter] = defaultdict(Counter)
        self.detections: list[tuple[str, str, float]] = []  # (voice, detected, seconds)
        self.base: list[tuple[float, float]] = []  # (seconds, right): the public measurement
        self.weights: np.ndarray | None = None
        self.dirty = False
        self._cache: dict[str, tuple[int, Counter]] = {}

    # -- learning -----------------------------------------------------------------------
    def heard(
        self,
        voice: str | None,
        language: str | None,
        text: str,
        weight: float,
        detected: bool,
        seconds: float,
    ) -> None:
        """A slow reading: ``language`` is what the slow recogniser wrote it in,
        ``detected`` whether it chose that language itself, ``weight`` how much it counts
        (the probability that a detection this long is right).

        Only a detection is evidence of what a voice speaks. A reading the recogniser
        was *told* to write in a language is in that language by construction: counting
        it made the first build lock KπX's voice into English after one misdetected
        phrase, then confirm it with every reading it forced (measured on the live
        corpus). Its words still teach what that language is written with."""
        if not language:
            return
        self.words[language].update(tokens(text))
        if voice is not None and detected:
            self.voices[voice][language] += weight
            self.detections.append((voice, language, seconds))
            self.dirty = True

    def measured(self, seconds: float, right: bool) -> None:
        """A detection on public audio whose language is known (the base)."""
        self.base.append((seconds, float(right)))
        self.dirty = True

    def refit(self) -> None:
        """The detector's reliability by length: the public measurement, then every
        detection of the stream against the language its voice turned out to speak."""
        examples = [
            (np.array([[math.log(seconds), 1], [0, 0]]), 0 if right else 1)
            for seconds, right in self.base
        ]
        for voice, detected, seconds in self.detections:
            spoken = self.of_voice(voice)
            if spoken is None:
                continue
            best = max(spoken, key=spoken.get)
            examples.append(
                (np.array([[math.log(seconds), 1], [0, 0]]), 0 if detected == best else 1)
            )
        outcomes = {y for _, y in examples}
        self.weights = (
            fit_choice(examples, len(examples[0][0][0]))[0] if {0, 1} <= outcomes else None
        )
        self.dirty = False

    # -- using --------------------------------------------------------------------------
    def room(self) -> dict[str, float] | None:
        """P(a reading of a voice never heard is in each language, or in another one
        (``OTHER``)): a Dirichlet over what every voice was detected speaking, its prior
        one pseudo-reading spread over the languages this ear can write and one more for
        a language it cannot yet. One English detection made the first build 100 % sure
        the room spoke English (measured live); it is not, now, and it learns."""
        total = Counter()
        for counts in self.voices.values():
            total.update(counts)
        if not total:
            return None
        known = (set(self.words) | set(total)) - {OTHER}
        mass = sum(total.values()) + 1
        share = 1 / (len(known) + 1)
        room = {k: (total.get(k, 0) + share) / mass for k in known}
        room[OTHER] = share / mass
        return room

    def of_voice(self, voice: str | None) -> dict[str, float] | None:
        """P(the next reading of this voice is in each language): Dirichlet predictive,
        the room as its prior (one pseudo-reading)."""
        room = self.room()
        if room is None:
            return None
        mine = self.voices.get(voice or "", Counter())
        mass = sum(mine.values()) + 1
        return {k: (mine.get(k, 0) + room.get(k, 0)) / mass for k in set(room) | set(mine)}

    def detection_right(self, seconds: float) -> float | None:
        if self.weights is None:
            return None
        x = np.array([[math.log(seconds), 1], [0, 0]])
        return float(choice_probabilities(x, self.weights)[0])

    def choose(self, voice: str | None, seconds: float) -> str | None:
        """The language to give the recogniser, or None to let it detect: the voice's
        most probable language, when it is a better bet than the detection at this length.
        Before anything is known of the detector, it is left to detect."""
        spoken, right = self.of_voice(voice), self.detection_right(seconds)
        if spoken is None or right is None:
            return None
        best = max((k for k in spoken if k != OTHER), key=spoken.get)
        return best if spoken[best] > right else None

    def of_text(self, text: str) -> dict[str, float] | None:
        """P(this text is in each language, or in one this ear never wrote): naive Bayes
        on the character trigrams of the words written in each, with an open class
        (``OTHER``) for which every trigram is as likely as one never seen.

        Trigrams, not words: on the live corpus a word model called "J'imagine des
        mains." another language (two words it had never seen); the letters of a
        language generalise to its new words, and Chinese characters in a French room
        are still nobody's language here."""
        grams = trigrams(text)
        if not grams or not self.words:
            return None
        counts = {language: self._grams(language) for language in self.words}
        alphabet = len(set().union(*counts.values())) + 1
        scores = {OTHER: -len(grams) * math.log(alphabet)}
        for language, seen in counts.items():
            mass = sum(seen.values()) + alphabet
            scores[language] = sum(math.log((seen.get(g, 0) + 1) / mass) for g in grams)
        top = max(scores.values())
        z = sum(math.exp(v - top) for v in scores.values())
        return {k: math.exp(v - top) / z for k, v in scores.items()}

    def _grams(self, language: str) -> Counter:
        cached = self._cache.get(language)
        size = sum(self.words[language].values())
        if cached is None or cached[0] != size:
            grams = Counter()
            for word, n in self.words[language].items():
                for g in trigrams(word):
                    grams[g] += n
            cached = self._cache[language] = (size, grams)
        return cached[1]

    def in_language(self, voice: str | None, text: str) -> float | None:
        """P(this reading is in the language its voice speaks); None when either is unknown."""
        spoken, written = self.of_voice(voice), self.of_text(text)
        if not spoken or not written:
            return None
        return sum(spoken.get(k, spoken[OTHER]) * p for k, p in written.items())

    def same_language(self, a: str, b: str) -> float:
        """P(two texts are in one language): 1 when nothing is known of either."""
        pa, pb = self.of_text(a), self.of_text(b)
        if not pa or not pb:
            return 1.0
        return sum(p * pb.get(k, 0.0) for k, p in pa.items())

    def log_prior(self, voice: str | None, text: str) -> float:
        """log P(the reading's language | its voice) - log P(the voice's most probable
        language), the reading's language being the text's most probable one: 0 for a
        reading in the voice's language however uncertain the voice still is, about -4
        nats for an English translation of a voice of a French room, more for words of
        a language never written here, but never infinitely (a new language can be
        learned). Added as it is to the reader's scores: Bayes rule, no weight to learn.

        Measured live, three wrong designs before this one: a learned weight on
        ``1 - in_language`` (it grew to -53 on separable base episodes); the same against
        the other readings only (a lone reading in Chinese paid nothing); and the
        expectation over the text's language (the text detector's own doubt on a right
        French reading made it pay against "nothing was said")."""
        spoken, written = self.of_voice(voice), self.of_text(text)
        if not spoken or not written:
            return 0.0
        language = max(written, key=written.get)
        return math.log(spoken.get(language, spoken[OTHER])) - math.log(max(spoken.values()))

    # -- persistence ----------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "voices": {k: dict(v) for k, v in self.voices.items()},
            "words": {k: dict(v) for k, v in self.words.items()},
            "detections": [list(d) for d in self.detections],
            "base": [list(b) for b in self.base],
        }

    def load(self, raw: dict) -> None:
        self.voices = defaultdict(
            Counter, {k: Counter(v) for k, v in raw.get("voices", {}).items()}
        )
        self.words = defaultdict(Counter, {k: Counter(v) for k, v in raw.get("words", {}).items()})
        self.detections = [tuple(d) for d in raw.get("detections", [])]
        self.base = [tuple(b) for b in raw.get("base", [])]
        self.refit()
