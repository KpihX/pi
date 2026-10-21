"""The learned reader: which reading of a piece of speech is right, and how sure it is.

Two recognisers hear each piece (``asr.py``): the fast one at once, the slow one when
the scribe gets to it (``scribe.py``). The memory (``lexicon.py``) adds what it was
taught: readings with the learned substitutions applied, and corrected pieces that
resemble this one. Each reading becomes a candidate, plus one more that is not a
reading at all: **"I did not understand"**. The reader scores them all and the most
probable wins; nothing is compared to a number.

    features of a reading            what it says
    fast, slow                       which recogniser wrote it
    agree                            how much the other recogniser agrees, word by word
    logp                             the fast recogniser's own token confidence, against
                                     what it usually is here (on Piper voices -0.1, on the
                                     live television -0.8: as a level, it sent real speech
                                     to "I did not understand", measured)
    edits                            learned substitutions it needed
    recall                           how much it resembles a corrected piece
    known                            its share of words this ear has confirmed
    none                             "I did not understand", and why it might be:
    none_logp, none_disagree         low token confidence, two recognisers apart (in one
                                     language: a translation is not a doubt),
    none_unspoken                    Silero heard less speech here than it hears in the
                                     speech of this place (relative: on a TV through a
                                     laptop microphone, real speech is at 0.3-0.7)

And two terms that are not learned features but probabilities, added to every reading's
score with the coefficient Bayes gives them (one): ``log P(it is not a loop | how its
words repeat)`` (``repetition_terms``: Whisper's loops repeat most of their word
trigrams, speech a few for its length), and ``log P(its language | its voice)``,
against a reading fully in the voice's most probable language (``language.py``). A
Parakeet reading in English of a French voice is a translation (measured live), a
Whisper reading in Chinese of KπX a misdetection; neither is evidence against "nothing
was said". As a learned weight it was a disaster twice: fitted on translation episodes
the base made separable, it grew to -18, then -53, and a right French reading paid for
the room's still uncertain language.

The weights are a Bayesian conditional logit (``sphere.fit_choice``): fitted on public
pieces whose truth is known (the base), then on every text correction, with a prior
whose strength is estimated, not set. The probability of the chosen reading is its
calibrated trust (W6); per word, a second small model says how much each word
deserves it.

The doubt is named (the third law), from the reader's own posterior, never from an
equality of strings (v2's first build called "ambiguous" any two readings differing by
one word, and on two hours of live television it said "clear" twice):

* ``not_understood``: "I did not understand" is the most probable reading;
* ``taught``: what KπX taught (a correction recalled, a substitution learned) is the
  most probable reading: the ear knows this case;
* ``ambiguous``: the reader cannot choose: the chosen reading is more probably not the
  best one than the best one;
* ``partial``: it can choose, but the best reading there is holds more words probably
  wrong than right (the word model's own trust): "I understood bits of it". Its ``p``
  alone would hide this: p is "the best of the readings", not "a right reading";
* ``unseen``: more probably right, but holding a word never confirmed for this voice;
* ``clear``: more probably right, on words this ear knows.

The law: no numeric literal but 0 and 1 in this module (``tests/test_law.py``).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from π_ear.asr import Hypothesis
from π_ear.language import FAST, Languages
from π_ear.lexicon import Memory, align, errors, tokens
from π_ear.sphere import choice_probabilities, fit_choice

FEATURES = (
    "fast",
    "slow",
    "agree",
    "logp",
    "edits",
    "recall",
    "known",
    "none",
    "none_logp",
    "none_disagree",
    "none_unspoken",
)
WORD_FEATURES = ("agreed", "logp", "known", "edited", "bias")
# the list v2.0 saved its weights under, without names (the live test's memory has it)
V2_FEATURES = (
    "fast",
    "slow",
    "agree",
    "logp",
    "edits",
    "recall",
    "known",
    "none",
    "none_logp",
    "none_disagree",
    "none_unspoken",
)
NONE = "∅"  # the reading "I did not understand"


@dataclass(slots=True)
class Context:
    """What the reader knows of a piece besides its readings."""

    voice: str | None = None
    speech: float | None = None  # Silero's mean probability over the piece
    regime: float | None = None  # the same, over the speech of known voices of this place
    seconds: float | None = None  # how long the piece is
    confidence: float | None = None  # the fast recogniser's mean token log-probability here


@dataclass(slots=True)
class Heard:
    """A piece whose truth is known: a base example, or a correction."""

    fast: Hypothesis
    slow: Hypothesis | None
    truth: str
    context: Context = field(default_factory=Context)
    language: str | None = None  # the truth's language, when known


@dataclass(slots=True)
class Reading:
    text: str
    origins: set[str]
    edits: float = 0.0
    recall: float = 0.0


def mean_logp(hypothesis: Hypothesis | None) -> float:
    scores = [w.logp for w in hypothesis.words if w.logp is not None] if hypothesis else []
    return float(np.mean(scores)) if scores else 0.0


def agreement(a: list[str], b: list[str]) -> float:
    if not a and not b:
        return 1.0
    matched = sum(op == "=" for op, _, _ in align(a, b))
    return matched / max(len(a), len(b))


class Reader:
    def __init__(self, base: dict | None = None):
        self.memory = Memory()
        self.languages = Languages()
        dim, words = len(FEATURES), len(WORD_FEATURES)
        self.base = base or {}
        # how speech repeats its word trigrams: known speech (the base's truths), and
        # every reading heard, (repeated, trigrams) -> how many, a mixture to unmix
        self.speech = dict(self.base.get("repeats", {"repeated": 0.0, "trigrams": 0.0}))
        # what the recognisers wrote on the base's public pieces: readings too
        self.base_repeats = Counter({(k, m): n for k, m, n in self.base.get("readings", [])})
        self.repeats: Counter = Counter()
        self._loops: tuple | None = None
        self.mean = _fit(
            np.asarray(self.base.get("weights", np.zeros(dim)), dtype=np.float64),
            self.base.get("features"),
        )
        self.word_mean = np.asarray(
            self.base.get("word_weights", np.zeros(words)), dtype=np.float64
        )
        self.weights = self.mean.copy()
        self.word_weights = self.word_mean.copy()
        self.examples: list[tuple[np.ndarray, int]] = []
        self.word_examples: list[tuple[np.ndarray, int]] = []
        if "languages" in self.base:
            self.languages.load(self.base["languages"])
        self.dirty = False

    # -- the candidates ---------------------------------------------------------------
    def readings(
        self, fast: Hypothesis, slow: Hypothesis | None, voice: str | None
    ) -> list[Reading]:
        found: dict[str, Reading] = {}

        def add(text: str, origin: str, edits: float = 0.0, recall: float = 0.0):
            key = " ".join(tokens(text))
            if not key:
                return
            reading = found.setdefault(key, Reading(text, set()))
            reading.origins.add(origin)
            reading.edits = max(reading.edits, edits)
            reading.recall = max(reading.recall, recall)

        for hypothesis, origin in ((fast, "fast"), (slow, "slow")):
            if hypothesis is not None:
                add(hypothesis.text, origin)
                for text, used in self.memory.variants(hypothesis.text):
                    add(text, origin, edits=used)
        for text, score in self.memory.recall(voice, fast.text):
            add(text, "recall", recall=score)
        return list(found.values())

    def features(
        self, reading: Reading, fast: Hypothesis, slow: Hypothesis | None, context: Context
    ) -> np.ndarray:
        words = tokens(reading.text)
        heard = [tokens(h.text) for h in (fast, slow) if h is not None]
        out = np.zeros(len(FEATURES))
        out[FEATURES.index("fast")] = "fast" in reading.origins
        out[FEATURES.index("slow")] = "slow" in reading.origins
        out[FEATURES.index("agree")] = np.mean([agreement(words, h) for h in heard])
        if "fast" in reading.origins:
            out[FEATURES.index("logp")] = _relative(mean_logp(fast), context.confidence)
        out[FEATURES.index("edits")] = reading.edits
        out[FEATURES.index("recall")] = reading.recall
        out[FEATURES.index("known")] = self.memory.known(context.voice, words)
        return out

    def none_features(
        self, fast: Hypothesis, slow: Hypothesis | None, context: Context
    ) -> np.ndarray:
        out = np.zeros(len(FEATURES))
        out[FEATURES.index("none")] = 1
        out[FEATURES.index("none_logp")] = _relative(mean_logp(fast), context.confidence)
        if slow is not None:
            # two readings apart are a doubt about speech only if they are in one
            # language: Parakeet's English for Whisper's French (measured live, studio
            # talk) is a translation, not a sign that nothing was said
            same = self.languages.same_language(fast.text, slow.text)
            out[FEATURES.index("none_disagree")] = same * (
                1 - agreement(tokens(fast.text), tokens(slow.text))
            )
        if context.speech is not None and context.regime is not None:
            out[FEATURES.index("none_unspoken")] = context.regime - context.speech
        return out

    def matrix(self, fast, slow, context: Context) -> tuple[list[Reading], np.ndarray]:
        readings = self.readings(fast, slow, context.voice)
        rows = [self.features(r, fast, slow, context) for r in readings]
        rows.append(self.none_features(fast, slow, context))
        return readings, np.vstack(rows)

    # -- the decision ----------------------------------------------------------------
    def language_terms(self, readings: list[Reading], context: Context) -> np.ndarray:
        """log P(each reading's language | its voice), against the voice's best; 0 for
        "nothing was said" and whenever nothing is known."""
        terms = [self.languages.log_prior(context.voice, r.text) for r in readings]
        return np.array(terms + [0.0])

    def repetition_terms(self, readings: list[Reading]) -> np.ndarray:
        """log P(a reading is speech, not a loop | it repeats k of its m word trigrams).

        Speech repeats a trigram now and then, Poisson(q m); a loop ("la Céline de la
        Céline de la Céline...") repeats any share of them, uniformly. Both, and how
        often loops come, are learned without a single label (``loops``): known speech
        (the base's truths) and every reading either recogniser wrote, unmixed. 90 % of
        2405 live readings repeat nothing, the loops 30-80 %.

        Three designs failed on the live replay first: the pace (words per second: the
        loops ran at 2.5-5.6, inside speech's own 0.6-4.4); a cost per repeated trigram
        (a 100-word studio reading with three natural repeats paid 20 nats and lost to a
        short translation: 0.78 -> 0.55); and Poisson(q m) against its most probable
        count, q learned on accepted readings: the base's short sentences made q 0.0014,
        and a player's 67-trigram answer repeating "c'est vrai que" and "sur le terrain"
        paid 5.4 nats and lost to "I did not understand" (0.75 -> 0.57). Against a loop,
        what such a reading's two repeats cost is a few hundredths of a nat."""
        q, loops = self.loops()
        terms = []
        for r in readings:
            k, m = repeated(r.text)
            terms.append(_not_a_loop(k, m, q, loops) if m else 0.0)
        return np.array(terms + [0.0])

    def loops(self) -> tuple[float, float]:
        """(q, the share of readings that are loops): EM on the readings heard, known
        speech held as speech, one pseudo-count each, until it stops moving (or one pass
        per kind of reading)."""
        key = (sum(self.repeats.values()), self.speech["trigrams"])
        if self._loops is not None and self._loops[0] == key:
            return self._loops[1]
        heard = self.base_repeats + self.repeats
        seen = [(k, m, n) for (k, m), n in heard.items() if m]
        total = sum(n for _, _, n in seen)
        q = (self.speech["repeated"] + 1) / (self.speech["trigrams"] + 1 + 1)
        share = 1 / (total + 1 + 1)
        if self._loops is not None:  # from where it was: one reading moves it little
            q, share = self._loops[1]
        for _ in range(len(seen) + 1):
            repeated_, trigrams, loops = self.speech["repeated"], self.speech["trigrams"], 0.0
            for k, m, n in seen:
                speech = math.exp(_not_a_loop(k, m, q, share))
                repeated_ += n * speech * k
                trigrams += n * speech * m
                loops += n * (1 - speech)
            again = ((repeated_ + 1) / (trigrams + 1 + 1), (loops + 1) / (total + 1 + 1))
            if again == (q, share):
                break
            q, share = again
        self._loops = (key, (q, share))
        return q, share

    def heard(self, text: str) -> None:
        """A reading either recogniser wrote: how its words repeat joins the mixture."""
        k, m = repeated(text)
        if m:
            self.repeats[(k, m)] += 1

    def probabilities(self, readings, features, context: Context) -> np.ndarray:
        scores = (
            features @ self.weights
            + self.language_terms(readings, context)
            + self.repetition_terms(readings)
        )
        scores = scores - scores.max()
        return np.exp(scores) / np.exp(scores).sum()

    def read(self, fast: Hypothesis, slow: Hypothesis | None, context: Context) -> dict:
        readings, features = self.matrix(fast, slow, context)
        p = self.probabilities(readings, features, context)
        order = np.argsort(-p)
        best = int(order[0])
        runner = int(order[1]) if order[1:].size else best
        chosen = readings[best] if best < len(readings) else None
        log_p = np.log(np.maximum(p, np.finfo(float).tiny))
        words = self.word_marks(chosen, fast, slow, context.voice) if chosen else []
        if chosen is None:
            doubt = "not_understood"
        elif chosen.edits or chosen.recall:
            doubt = "taught"
        elif p[best] < 1 - p[best]:
            doubt = "ambiguous"
        elif _mean(words) < 1 - _mean(words):
            doubt = "partial"
        elif any(w["new"] for w in words):
            doubt = "unseen"
        else:
            doubt = "clear"
        return {
            "marks": words,
            "text": chosen.text if chosen else "",
            "origins": sorted(chosen.origins) if chosen else [],
            "p": float(p[best]),
            "margin": float(log_p[best] - log_p[runner]),
            "doubt": doubt,
            "words": [{k: v for k, v in w.items() if k != "features"} for w in words],
            "readings": [
                {
                    "text": readings[i].text if i < len(readings) else NONE,
                    "p": float(p[i]),
                    "origins": sorted(readings[i].origins) if i < len(readings) else [],
                }
                for i in order
            ],
        }

    def word_features(
        self, word: str, agreed: bool, logp: float | None, voice, edited: bool
    ) -> np.ndarray:
        return np.array(
            [
                agreed,
                logp if logp is not None else 0.0,
                self.memory.known(voice, [word]),
                edited,
                1.0,
            ]
        )

    def word_marks(
        self, reading: Reading, fast: Hypothesis, slow: Hypothesis | None, voice
    ) -> list[dict]:
        """Per word of the chosen reading: did the other recogniser hear it, how sure, is it new."""
        words = tokens(reading.text)
        fast_words = tokens(fast.text)
        other = tokens(slow.text) if slow is not None else []
        confidence = {i: w.logp for i, w in enumerate(fast.words)}
        heard_by_fast = {
            i: j for op, i, j in align(words, fast_words) if op == "=" and i is not None
        }
        heard_by_other = {i for op, i, j in align(words, other) if op == "=" and i is not None}
        marks = []
        for index, word in enumerate(words):
            agreed = index in heard_by_fast and (index in heard_by_other or slow is None)
            logp = confidence.get(heard_by_fast.get(index)) if index in heard_by_fast else None
            edited = reading.edits and index not in heard_by_fast
            f = self.word_features(word, agreed, logp, voice, bool(edited))
            trust = float(
                choice_probabilities(np.vstack([f, np.zeros_like(f)]), self.word_weights)[0]
            )
            marks.append(
                {
                    "w": word,
                    "agreed": bool(agreed),
                    "trust": trust,
                    "new": not self.memory.known(voice, [word]),
                    "features": f,
                }
            )
        return marks

    # -- learning ----------------------------------------------------------------------
    def label(self, readings: list[Reading], truth: str) -> int:
        """The best reading there was: fewest word errors against the truth; and if
        every reading is wrong on every word, the truth is "I did not understand"."""
        truth_words = tokens(truth)
        if not readings:
            return len(readings)
        wrong = [errors(truth_words, tokens(r.text)) for r in readings]
        best = int(np.argmin(wrong))
        nothing = max(len(truth_words), len(tokens(readings[best].text)))
        return len(readings) if wrong[best] == nothing else best

    def learn(self, heard: Heard) -> None:
        """One piece whose truth is known: an example for the weights, and memory."""
        readings, features = self.matrix(heard.fast, heard.slow, heard.context)
        self.examples.append((features, self.label(readings, heard.truth)))
        self.learn_words(heard, readings)
        voice = heard.context.voice
        self.memory.teach(voice, heard.fast.text, heard.truth, [r.text for r in readings])
        self.dirty = True

    def learn_words(self, heard: Heard, readings: list[Reading]) -> None:
        """Every word of every reading, right or wrong against the truth: what a word's
        trust is learned from. (It once learned from the reading the reader chose, with
        weights not fitted yet: arbitrary readings, and agreed words came out untrusted.)"""
        truth_words = tokens(heard.truth)
        for reading in readings:
            right = {
                i
                for op, i, j in align(tokens(reading.text), truth_words)
                if op == "=" and i is not None
            }
            marks = self.word_marks(reading, heard.fast, heard.slow, heard.context.voice)
            for index, mark in enumerate(marks):
                f = mark["features"]
                self.word_examples.append(
                    (np.vstack([f, np.zeros_like(f)]), 0 if index in right else 1)
                )

    def agree(self, fast: Hypothesis, slow: Hypothesis | None, voice: str | None) -> None:
        """The words both recognisers wrote at the same place: the lexicon counts them
        (word by word: two long readings never agree on every word, and v2's first build,
        which asked for that, confirmed almost nothing in two hours of television)."""
        if slow is None:
            return
        a, b = tokens(fast.text), tokens(slow.text)
        self.memory.confirm(voice, [a[i] for op, i, _ in align(a, b) if op == "="])

    def refit(self) -> None:
        """The slow clock: every example ever seen, refitted (replay, never forgetting).
        Clean first: a refit that fails must not be retried forever (a first build's
        scribe did, and the ear waited on it)."""
        self.dirty = False
        if self.examples:
            self.weights, _ = fit_choice(self.examples, len(FEATURES), self.mean)
        if self.word_examples:
            self.word_weights, _ = fit_choice(
                self.word_examples, len(WORD_FEATURES), self.word_mean
            )
        if self.languages.dirty:
            self.languages.refit()

    # -- persistence ----------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "features": list(FEATURES),
            "weights": self.weights.tolist(),
            "word_weights": self.word_weights.tolist(),
            "memory": self.memory.to_dict(),
            "languages": self.languages.to_dict(),
            "repeats": [[k, m, n] for (k, m), n in self.repeats.items()],
            "examples": [[f.tolist(), y] for f, y in self.examples],
            "word_examples": [[f.tolist(), y] for f, y in self.word_examples],
        }

    def load(self, raw: dict) -> None:
        """What this ear learned: its memory, its languages, its examples. Not its weights:
        they are refitted from the base it runs on now plus those examples (a memory
        saved under an older base kept that base's weights frozen, measured live)."""
        names = raw.get("features")
        self.memory.load(raw.get("memory", {}))
        if "languages" in raw:
            self.languages.load(raw["languages"])
        # (the first build of v2.1 saved accepted counts here, a dict: not readings)
        saved = raw.get("repeats")
        self.repeats = Counter({(k, m): n for k, m, n in saved} if isinstance(saved, list) else {})
        self.examples = [(_fit_rows(np.asarray(f), names), y) for f, y in raw.get("examples", [])]
        self.word_examples = [(np.asarray(f), y) for f, y in raw.get("word_examples", [])]
        self.weights, self.word_weights = self.mean.copy(), self.word_mean.copy()
        self.refit()


def _poisson(k: int, rate: float) -> float:
    return k * math.log(rate) - rate - math.lgamma(k + 1)


def _not_a_loop(k: int, m: int, q: float, loops: float) -> float:
    """log P(speech | k of m trigrams repeated): Poisson(q m) against uniform on 0..m-1."""
    speech = math.log1p(-loops) + _poisson(k, q * m)
    loop = math.log(loops) - math.log(m)
    return speech - float(np.logaddexp(speech, loop))


def repeated(text: str) -> tuple[int, int]:
    """(word trigrams seen before in this text, word trigrams)."""
    words = tokens(text)
    grams = list(zip(words, words[1:], words[1 + 1 :], strict=False))
    return len(grams) - len(set(grams)), len(grams)


def _relative(value: float, usual: float | None) -> float:
    """A sensor's level against what it usually is in this place (itself if unknown)."""
    return value - usual if usual is not None else value


def _mean(words: list[dict]) -> float:
    """The expected share of right words of a reading, from each word's own trust."""
    return float(np.mean([w["trust"] for w in words])) if words else 0.0


def _fit(weights: np.ndarray, names=None, base=None) -> np.ndarray:
    """Weights saved under another feature list, by name: a feature that did not exist
    takes its base weight (or none), one that no longer exists is dropped."""
    names = list(names) if names else _legacy(weights.size)
    base = base if base is not None else np.zeros(len(FEATURES))
    return np.array(
        [weights[names.index(f)] if f in names else base[i] for i, f in enumerate(FEATURES)]
    )


def _legacy(size: int) -> list[str]:
    """The names of weights saved without them: v2.0's list (and its first features)."""
    return list(V2_FEATURES[:size])


def _fit_rows(features: np.ndarray, names=None) -> np.ndarray:
    names = list(names) if names else _legacy(features.shape[1])
    out = np.zeros((features.shape[0], len(FEATURES)))
    for i, f in enumerate(FEATURES):
        if f in names:
            out[:, i] = features[:, names.index(f)]
    return out


def fit_reader(pieces: list[Heard]) -> dict:
    """The base weights of the reader, from public pieces whose truth is known.

    Two kinds of example. Plain pieces: what each recogniser wrote, against the truth.
    And correction episodes: every piece the recognisers got wrong is corrected once,
    then heard again; what the memory now offers is weighed against the raw readings
    (without these, a corrected error came back once before the reader learned to trust
    what it was taught). The language term is a probability, not a weight: it needs no
    example (the translation episodes it once had made it separable, and huge).
    """
    reader = Reader()
    for heard in pieces:
        if heard.language:
            # what each language is written with, and what each base voice speaks
            reader.languages.words[heard.language].update(tokens(heard.truth))
            reader.languages.known[heard.context.voice][heard.language] += 1
        k, m = repeated(heard.truth)
        reader.speech["repeated"] += k
        reader.speech["trigrams"] += m
        for hypothesis in (heard.fast, heard.slow):
            if hypothesis is not None:
                reader.heard(hypothesis.text)
        if heard.language and heard.slow is not None and heard.slow.language:
            seconds = heard.context.seconds or (heard.fast.words[-1].t1 if heard.fast.words else 1)
            reader.languages.measured(seconds, heard.slow.language == heard.language)
    for heard in pieces:  # the fast recogniser's text, as a detector of its language
        written = reader.languages.of_text(heard.fast.text) if heard.language else None
        if written:
            seconds = heard.context.seconds or (heard.fast.words[-1].t1 if heard.fast.words else 1)
            reader.languages.measured(
                seconds, max(written, key=written.get) == heard.language, FAST
            )
    episodes = 0
    for heard in pieces:
        readings, features = reader.matrix(heard.fast, heard.slow, heard.context)
        reader.examples.append((features, reader.label(readings, heard.truth)))
        reader.learn_words(heard, readings)
    for heard in pieces:
        readings, _ = reader.matrix(heard.fast, heard.slow, heard.context)
        if tokens(heard.truth) and not any(tokens(r.text) == tokens(heard.truth) for r in readings):
            taught = Reader()
            taught.languages = reader.languages
            taught.memory.teach(
                heard.context.voice, heard.fast.text, heard.truth, [r.text for r in readings]
            )
            again, features = taught.matrix(heard.fast, heard.slow, heard.context)
            reader.examples.append((features, taught.label(again, heard.truth)))
            # the same mistake in a piece never corrected: only the substitutions and
            # the confirmed words can help, not the recall of this very piece
            taught.memory.examples.clear()
            again, features = taught.matrix(heard.fast, heard.slow, heard.context)
            reader.examples.append((features, taught.label(again, heard.truth)))
            episodes += 1
    reader.refit()
    return {
        "features": list(FEATURES),
        "weights": reader.weights.tolist(),
        "word_weights": reader.word_weights.tolist(),
        "languages": {
            "words": reader.languages.to_dict()["words"],
            "base": reader.languages.to_dict()["base"],
        },
        "repeats": reader.speech,
        "readings": [[k, m, n] for (k, m), n in reader.repeats.items()],
        "runs": len(pieces),
        "episodes": episodes,
    }
