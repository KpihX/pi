"""Which language a voice speaks, learned, and when to tell the recognisers.

Measured live (2026-10-20, PSG - Barça on a laptop microphone): Whisper's own language
detection wrote English, Russian, Portuguese, Turkish, Japanese or Chinese on 15 % of
the runs of a French broadcast, and on most of KπX's own short phrases; Parakeet, which
cannot be told a language, answered in English on studio talk (2 % of the words on 4 s
pieces, 19 % on 8 s, 77 % on 32 s). A language is a property of a voice, not of one run.

What is learned here, nothing set:

* **how often a detection is right**, by the length it heard: a Bayesian logit on the
  public base (clean speech, right 99 % of the time even at 2 s), then on the stream,
  each detection judged against the language its voice turned out to speak *without
  it*; and per voice, how much worse or better than that its own detections are (on
  KπX's voice in a room over a television, one in three was right where the base says
  six in seven: the regime is the voice's, not the length's alone);
* **what a voice speaks**: its detections, each counted by the probability that it is
  right, over a prior that is the room. A voice speaks like the room as strongly as the
  voices of this room do: the strength of that prior (a Dirichlet concentration) is
  estimated on every voice at once (empirical Bayes, under a log-logistic prior of
  scale one pseudo-reading), never set. A room of French television makes a new voice
  French before it says anything; a guest speaking English moves it with every
  reliable reading;
* **the room**: every voice's counted detections, with one pseudo-reading spread over
  the languages this ear can write and one more for a language it never wrote;
* **what words each language is written with** (the slow recogniser's words, by the
  language it wrote them in), to say in which language a reading *is*: a Parakeet
  reading in English for a French voice is a translation, and the reader is told.

The decision has no threshold: the voice's language is given to the recogniser when it
is more probable than this voice's detection being right at this length.

The law: no numeric literal but 0 and 1 in this module (``tests/test_law.py``).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import minimize_scalar

from π_ear.contract import TEXT_GRAM
from π_ear.lexicon import tokens
from π_ear.sphere import fit_logit

OTHER = "?"  # a language this ear has never written


def trigrams(text: str) -> list[str]:
    """The character trigrams of a text's words, each word framed by spaces."""
    out = []
    for word in tokens(text):
        framed = f" {word} "
        out += [framed[i : i + TEXT_GRAM] for i in range(len(framed) - TEXT_GRAM + 1)]
    return out


def _logit(p: float) -> float:
    tiny = np.finfo(float).eps
    p = min(max(p, tiny), 1 - tiny)
    return math.log(p) - math.log1p(-p)


def _sigmoid(z: float) -> float:
    return float(np.exp(-np.logaddexp(0, -z)))


class Languages:
    def __init__(self):
        self.words: dict[str, Counter] = defaultdict(Counter)
        self.known: dict[str, Counter] = defaultdict(Counter)  # languages known spoken
        self.detections: list[tuple[str, str, float]] = []  # (voice, detected, seconds)
        self.base: list[tuple[float, float]] = []  # (seconds, right): the public measurement
        self.weights: np.ndarray | None = None  # reliability by length (log-odds)
        self.offsets: dict[str, float] = {}  # per voice, log-odds against that
        self.counts: dict[str, Counter] = defaultdict(Counter)  # detections, by reliability
        self.counted: list[float] = []  # what each detection was counted with
        self.concentration = 1.0  # how strongly a voice speaks like the room, estimated
        self._room: tuple | None = None
        self._version = 0  # bumped whenever a count changes
        self.dirty = False
        self._cache: dict[str, tuple[int, Counter]] = {}

    # -- learning -----------------------------------------------------------------------
    def heard(
        self, voice: str | None, language: str | None, text: str, detected: bool, seconds: float
    ) -> None:
        """A slow reading: ``language`` is what the slow recogniser wrote it in,
        ``detected`` whether it chose that language itself.

        Only a detection is evidence of what a voice speaks. A reading the recogniser
        was *told* to write in a language is in that language by construction: counting
        it made the first build lock KπX's voice into English after one misdetected
        phrase, then confirm it with every reading it forced (measured on the live
        corpus). Its words still teach what that language is written with."""
        if not language:
            return
        self.words[language].update(tokens(text))
        if voice is not None and detected:
            right = self.detection_right(seconds, voice)
            weight = 1 if right is None else right
            self.detections.append((voice, language, seconds))
            self.counted.append(weight)
            self.counts[voice][language] += weight
            self._version += 1
            self.dirty = True

    def measured(self, seconds: float, right: bool) -> None:
        """A detection on public audio whose language is known (the base)."""
        self.base.append((seconds, float(right)))
        self.dirty = True

    def refit(self) -> None:
        """Reliability, counts and concentration, then each detection judged again
        against its voice without it, until the judgements stop changing (a pass moves
        only what the previous one moved; at most one pass per detection)."""
        self.dirty = False  # first: a refit that fails is not retried forever
        self.weights, self.offsets = self._reliability([]), {}
        judged = None
        for _ in range(len(self.detections) + 1):
            self._recount()
            self.concentration = self._concentration()
            again = self._judged()
            if again == judged:
                break
            judged = again
            self.weights = self._reliability(judged)
            self.offsets = self._offsets(judged) if self.weights is not None else {}

    def _reliability(self, judged: list[tuple[str, float, bool]]) -> np.ndarray | None:
        seen = [(seconds, right) for seconds, right in self.base]
        seen += [(seconds, right) for _, seconds, right in judged]
        right = np.array([float(r) for _, r in seen])
        if not seen or right.min() == right.max():
            return None
        return fit_logit(np.array([_x(seconds) for seconds, _ in seen]), right)

    def _recount(self) -> None:
        self.counts, self.counted = defaultdict(Counter), []
        self._version += 1
        for voice, detected, seconds in self.detections:
            right = self.detection_right(seconds, voice)
            weight = 1 if right is None else right
            self.counts[voice][detected] += weight
            self.counted.append(weight)

    def _judged(self) -> list[tuple[str, float, bool]]:
        """Each detection against its voice's most probable language without it: with
        it, a voice heard once was always detected right, and the short detections that
        fool Whisper came out reliable (measured: 0.98 where it was 0.02)."""
        out, total = [], self._total()
        for (voice, detected, seconds), weight in zip(self.detections, self.counted, strict=True):
            mine, rest = self._mine(voice), Counter(total)
            mine[detected] -= weight
            rest[detected] -= weight  # out of the room too: alone in it, a voice was its room
            room = self._room_of(rest)
            if room is None:  # nothing else heard yet: nothing to judge it against
                continue
            spoken = self.of_voice(voice, mine, room)
            if spoken is not None:
                out.append((voice, seconds, detected == max(spoken, key=spoken.get)))
        return out

    def _offsets(self, judged: list[tuple[str, float, bool]]) -> dict[str, float]:
        """Per voice, the log-odds by which its detections are more or less often right
        than their lengths predict: its own record, with the prediction as one more
        detection (a voice with no record keeps the reliability of its lengths)."""
        record: dict[str, list[tuple[float, bool]]] = defaultdict(list)
        for voice, seconds, right in judged:
            record[voice].append((self.detection_right(seconds), right))
        out = {}
        for voice, seen in record.items():
            predicted = float(np.mean([p for p, _ in seen]))
            observed = (sum(right for _, right in seen) + predicted) / (len(seen) + 1)
            out[voice] = _logit(observed) - _logit(predicted)
        return out

    def _concentration(self) -> float:
        """How strongly a voice speaks like the room: the Dirichlet concentration under
        which every voice's counted detections are most probable (Dirichlet-multinomial),
        with a log-logistic prior of scale one pseudo-reading (the scale the room's own
        prior has). Voices that all speak the room's language make it large, a room of
        many languages small."""
        room = self.room()
        voices = [self._mine(v) for v in set(self.counts) | set(self.known)]
        voices = [c for c in voices if sum(c.values())]
        if room is None or not voices:
            return self.concentration

        def minus_log_posterior(t: float) -> float:
            a = math.exp(t)
            out = t - (1 + 1) * math.log1p(a)  # the standard logistic, on log a
            for counts in voices:
                out += math.lgamma(a) - math.lgamma(a + sum(counts.values()))
                for language, n in counts.items():
                    share = a * room.get(language, room[OTHER])
                    out += math.lgamma(n + share) - math.lgamma(share)
            return -out

        return math.exp(minimize_scalar(minus_log_posterior).x)

    # -- using --------------------------------------------------------------------------
    def _mine(self, voice: str | None) -> Counter:
        mine = Counter(self.counts.get(voice or "", Counter()))
        mine.update(self.known.get(voice or "", Counter()))
        return mine

    def room(self) -> dict[str, float] | None:
        """P(a reading of a voice never heard is in each language, or in another one
        (``OTHER``)): every voice's counted detections, with one pseudo-reading spread
        over the languages this ear can write and one more for a language it cannot yet.
        One English detection made the first build 100 % sure the room spoke English
        (measured live); it is not, now, and it learns."""
        key = (
            self._version,
            tuple(sorted(self.words)),
            sum(map(sum, map(Counter.values, self.known.values()))),
        )
        if self._room is not None and self._room[0] == key:
            return self._room[1]
        room = self._room_of(self._total())
        self._room = (key, room)
        return room

    def _total(self) -> Counter:
        total = Counter()
        for counts in (*self.counts.values(), *self.known.values()):
            total.update(counts)
        return total

    def _room_of(self, total: Counter) -> dict[str, float] | None:
        total = Counter({k: max(n, 0) for k, n in total.items() if max(n, 0)})  # no residue
        if not total:
            return None
        known = (set(self.words) | set(total)) - {OTHER}
        mass = sum(total.values()) + 1
        share = 1 / (len(known) + 1)
        room = {k: (total.get(k, 0) + share) / mass for k in known}
        room[OTHER] = share / mass
        return room

    def of_voice(
        self, voice: str | None, mine: Counter | None = None, room: dict | None = None
    ) -> dict[str, float] | None:
        """P(the next reading of this voice is in each language): Dirichlet predictive
        over its counted detections, the room as its mean, as concentrated as the voices
        of this room show; ``mine`` and ``room``: other counts, to judge without one."""
        room = self.room() if room is None else room
        if room is None:
            return None
        mine = self._mine(voice) if mine is None else mine
        a = self.concentration
        mass = sum(mine.values()) + a
        return {k: (mine.get(k, 0) + a * p) / mass for k, p in room.items()}

    def detection_right(self, seconds: float, voice: str | None = None) -> float | None:
        """P(a detection is right) at this length, for this voice's detections."""
        if self.weights is None:
            return None
        z = float(_x(seconds) @ self.weights) + self.offsets.get(voice or "", 0)
        return _sigmoid(z)

    def choose(self, voice: str | None, seconds: float) -> str | None:
        """The language to give the recogniser, or None to let it detect: the voice's
        most probable language, when it is a better bet than its detection at this
        length. Before anything is known of the detector, it is left to detect."""
        spoken, right = self.of_voice(voice), self.detection_right(seconds, voice)
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
            "words": {k: dict(v) for k, v in self.words.items()},
            "known": {k: dict(v) for k, v in self.known.items()},
            "detections": [list(d) for d in self.detections],
            "base": [list(b) for b in self.base],
        }

    def load(self, raw: dict) -> None:
        """Everything is refitted from what was heard (the first build of v2.1 also
        saved counts, ``voices``, and a weight per detection: recounted, not trusted)."""
        self.words = defaultdict(Counter, {k: Counter(v) for k, v in raw.get("words", {}).items()})
        self.known = defaultdict(Counter, {k: Counter(v) for k, v in raw.get("known", {}).items()})
        self.detections = [
            (voice, detected, seconds) for voice, detected, seconds, *_ in raw.get("detections", [])
        ]
        self.base = [tuple(b) for b in raw.get("base", [])]
        self.refit()


def _x(seconds: float) -> np.ndarray:
    """A detection's features: the log of its length, and a bias."""
    return np.array([math.log(seconds), 1])
