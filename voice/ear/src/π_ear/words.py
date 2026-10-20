"""The learned reader: which reading of a speech run is right, and how sure it is.

Two recognisers hear each run (``asr.py``): the fast one first, the slow one in
the next silence. The memory (``lexicon.py``) adds what it was taught: readings
with the learned substitutions applied, and corrected runs that resemble this
one. Each reading becomes a candidate, plus one more that is not a reading at
all: **"I did not understand"**. The reader scores them all and the most
probable wins; nothing is compared to a number.

    features of a reading                       what it says
    fast, slow                                  which recogniser wrote it
    agree                                       how much the other recogniser agrees
    logp                                        the fast recogniser's own token confidence
    edits                                       learned substitutions it needed
    recall                                      how much it resembles a corrected run
    known                                       its share of words this ear has confirmed
    none, none_logp, none_disagree              "I did not understand", and why it might be:
    none_unspoken                               low token confidence, two recognisers apart,
                                                a run Silero itself found little like speech

The weights are a Bayesian conditional logit (``sphere.fit_choice``): fitted on
public runs whose truth is known (the base), then on every text correction,
with a prior whose strength is estimated, not set. The probability of the
chosen reading is its calibrated trust (W6); per word, a second small model
says how much each word deserves it.

The doubt is named (the third law), from what the evidence is, not from a cut:

* ``not_understood``: "I did not understand" is the most probable reading;
* ``taught``: the two recognisers disagree, and what KπX taught (a correction
  recalled, a substitution learned) is the most probable reading: the ear knows
  this case, it does not ask again;
* ``ambiguous``: the two recognisers disagree, and the reader chose between them;
* ``unseen``: they agree, but some word was never confirmed for this voice;
* ``clear``: they agree, on words this ear knows.

The law: no numeric literal but 0 and 1 in this module (``tests/test_law.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from π_ear.asr import Hypothesis
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
NONE = "∅"  # the reading "I did not understand"


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
        dim, words = len(FEATURES), len(WORD_FEATURES)
        self.base = base or {}
        self.mean = np.asarray(self.base.get("weights", np.zeros(dim)), dtype=np.float64)
        self.word_mean = np.asarray(
            self.base.get("word_weights", np.zeros(words)), dtype=np.float64
        )
        self.weights = self.mean.copy()
        self.word_weights = self.word_mean.copy()
        self.examples: list[tuple[np.ndarray, int]] = []
        self.word_examples: list[tuple[np.ndarray, int]] = []
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
        self, reading: Reading, fast: Hypothesis, slow: Hypothesis | None, voice: str | None
    ) -> np.ndarray:
        words = tokens(reading.text)
        heard = [tokens(h.text) for h in (fast, slow) if h is not None]
        agree = float(np.mean([agreement(words, h) for h in heard])) if heard else 0.0
        return np.array(
            [
                "fast" in reading.origins,
                "slow" in reading.origins,
                agree,
                mean_logp(fast) if "fast" in reading.origins else 0.0,
                reading.edits,
                reading.recall,
                self.memory.known(voice, words),
            ]
            + [0.0] * sum(f.startswith("none") for f in FEATURES),
            dtype=np.float64,
        )

    def none_features(
        self, fast: Hypothesis, slow: Hypothesis | None, speech: float | None = None
    ) -> np.ndarray:
        disagree = 1 - agreement(tokens(fast.text), tokens(slow.text)) if slow else 0.0
        out = np.zeros(len(FEATURES))
        out[FEATURES.index("none")] = 1
        out[FEATURES.index("none_logp")] = mean_logp(fast)
        out[FEATURES.index("none_disagree")] = disagree
        # how little the run sounded like speech to Silero (its mean probability):
        # measured, 0.82-0.94 on speech, under 0.78 on 27 of 28 indoor sounds
        out[FEATURES.index("none_unspoken")] = 1 - speech if speech is not None else 0.0
        return out

    def matrix(self, fast, slow, voice, speech=None) -> tuple[list[Reading], np.ndarray]:
        readings = self.readings(fast, slow, voice)
        rows = [self.features(r, fast, slow, voice) for r in readings]
        rows.append(self.none_features(fast, slow, speech))
        return readings, np.vstack(rows)

    # -- the decision ----------------------------------------------------------------
    def read(
        self,
        fast: Hypothesis,
        slow: Hypothesis | None,
        voice: str | None,
        speech: float | None = None,
    ) -> dict:
        """``speech``: Silero's mean speech probability over the run, if known."""
        readings, features = self.matrix(fast, slow, voice, speech)
        p = choice_probabilities(features, self.weights)
        order = np.argsort(-p)
        best = int(order[0])
        runner = int(order[1]) if order[1:].size else best
        chosen = readings[best] if best < len(readings) else None
        log_p = np.log(np.maximum(p, np.finfo(float).tiny))
        words = self.word_marks(chosen, fast, slow, voice) if chosen else []
        agreed = slow is not None and agreement(tokens(fast.text), tokens(slow.text)) == 1
        if chosen is None:
            doubt = "not_understood"
        elif not agreed and slow is not None:
            doubt = "taught" if chosen.edits or chosen.recall else "ambiguous"
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

    def learn(
        self,
        fast: Hypothesis,
        slow: Hypothesis | None,
        voice: str | None,
        truth: str,
        speech: float | None = None,
    ) -> None:
        """One run whose truth is known: an example for the weights, and memory."""
        readings, features = self.matrix(fast, slow, voice, speech)
        self.examples.append((features, self.label(readings, truth)))
        self.learn_words(fast, slow, voice, truth, readings, speech)
        heard = [r.text for r in readings]
        self.memory.teach(voice, fast.text, truth, heard)
        self.dirty = True

    def learn_words(self, fast, slow, voice, truth, readings, speech=None) -> None:
        truth_words = tokens(truth)
        decided = self.read(fast, slow, voice, speech)
        chosen = next((r for r in readings if r.text == decided["text"]), None)
        if chosen is None:
            return
        right = {
            i for op, i, j in align(tokens(chosen.text), truth_words) if op == "=" and i is not None
        }
        for index, mark in enumerate(decided["marks"]):
            f = mark["features"]
            self.word_examples.append(
                (np.vstack([f, np.zeros_like(f)]), 0 if index in right else 1)
            )

    def agree(self, fast: Hypothesis, slow: Hypothesis | None, voice: str | None) -> None:
        """Both recognisers wrote the same words: the lexicon may count them (not the weights)."""
        if slow is not None and tokens(fast.text) == tokens(slow.text):
            self.memory.confirm(voice, tokens(fast.text))

    def refit(self) -> None:
        """The slow clock: every example ever seen, refitted (replay, never forgetting)."""
        if self.examples:
            self.weights, _ = fit_choice(self.examples, len(FEATURES), self.mean)
        if self.word_examples:
            self.word_weights, _ = fit_choice(
                self.word_examples, len(WORD_FEATURES), self.word_mean
            )
        self.dirty = False

    # -- persistence ----------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "weights": self.weights.tolist(),
            "word_weights": self.word_weights.tolist(),
            "memory": self.memory.to_dict(),
            "examples": [[f.tolist(), y] for f, y in self.examples],
            "word_examples": [[f.tolist(), y] for f, y in self.word_examples],
        }

    def load(self, raw: dict) -> None:
        weights = np.asarray(raw.get("weights", self.weights), dtype=np.float64)
        # a memory from before a feature existed: the new feature starts at its base
        self.weights = np.concatenate([weights, self.mean[weights.size :]])
        self.word_weights = np.asarray(raw.get("word_weights", self.word_weights), dtype=np.float64)
        self.memory.load(raw.get("memory", {}))
        self.examples = [(_widen(np.asarray(f)), y) for f, y in raw.get("examples", [])]
        self.word_examples = [(np.asarray(f), y) for f, y in raw.get("word_examples", [])]


def _widen(features: np.ndarray) -> np.ndarray:
    """Examples saved before a feature existed: that feature was not seen (zero)."""
    missing = len(FEATURES) - features.shape[1]
    return np.hstack([features, np.zeros((features.shape[0], missing))]) if missing else features


def fit_reader(runs: list[tuple[Hypothesis, Hypothesis | None, str, float | None]]) -> dict:
    """The base weights of the reader, from public runs whose truth is known.

    Two kinds of example. Plain runs: what each recogniser wrote, against the
    truth. And correction episodes: every run the recognisers got wrong is
    corrected once, then heard again; what the memory now offers (the learned
    substitution, the recalled correction, the confirmed words) is weighed
    against the raw readings. Without these, the weights of the memory would
    start at zero and a corrected error would come back once before the reader
    learned to trust what it was taught.
    """
    reader = Reader()
    for fast, slow, truth, speech in runs:
        readings, features = reader.matrix(fast, slow, None, speech)
        reader.examples.append((features, reader.label(readings, truth)))
        reader.learn_words(fast, slow, None, truth, readings, speech)
    episodes = 0
    for fast, slow, truth, speech in runs:
        readings, _ = reader.matrix(fast, slow, None, speech)
        if not tokens(truth) or any(tokens(r.text) == tokens(truth) for r in readings):
            continue
        taught = Reader()
        taught.memory.teach(None, fast.text, truth, [r.text for r in readings])
        again, features = taught.matrix(fast, slow, None, speech)
        reader.examples.append((features, taught.label(again, truth)))
        # the same mistake in an utterance never corrected: only the substitutions
        # and the confirmed words can help, not the recall of this very run
        taught.memory.examples.clear()
        again, features = taught.matrix(fast, slow, None, speech)
        reader.examples.append((features, taught.label(again, truth)))
        episodes += 1
    reader.refit()
    return {
        "weights": reader.weights.tolist(),
        "word_weights": reader.word_weights.tolist(),
        "runs": len(runs),
        "episodes": episodes,
    }
