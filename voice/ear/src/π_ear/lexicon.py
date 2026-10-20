"""The ear's memory of words: what it was taught, kept outside any model's weights.

Three stores, all learned from corrections and from the two recognisers agreeing,
all on disk, all rebuilt exactly by replay:

* **lexicon**: per voice, the words this ear has confirmed (a correction, or the
  fast and the slow recogniser writing the same word at the same place);
* **substitutions**: what the ear heard where the truth was something else
  ("pi x" where KπX said "KπX"), learned from the alignment of a reading with
  its correction. Applied to every new reading, they are candidates, never
  silent rewrites: the reader (``words.py``) weighs them;
* **examples**: every corrected run, for recall: a new run whose first reading
  is close to a corrected one brings that correction as a candidate.

Text utilities live here too: normalisation and word alignment.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict

WORD = re.compile(r"[\w'’-]+", re.UNICODE)


def tokens(text: str) -> list[str]:
    """Words for comparison: case and accents folded, punctuation dropped."""
    folded = unicodedata.normalize("NFKD", text.casefold())
    folded = "".join(c for c in folded if not unicodedata.combining(c)).replace("’", "'")
    return WORD.findall(folded)


def align(a: list[str], b: list[str]) -> list[tuple[str, int | None, int | None]]:
    """Word alignment by edit distance: ('=', i, j), ('~', i, j), ('-', i, None), ('+', None, j)."""
    n, m = len(a), len(b)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        cost[i][0] = i
    for j in range(m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i][j] = min(
                cost[i - 1][j] + 1, cost[i][j - 1] + 1, cost[i - 1][j - 1] + (a[i - 1] != b[j - 1])
            )
    out, i, j = [], n, m
    while i or j:
        if i and j and cost[i][j] == cost[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            out.append(("=" if a[i - 1] == b[j - 1] else "~", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and cost[i][j] == cost[i - 1][j] + 1:
            out.append(("-", i - 1, None))
            i -= 1
        else:
            out.append(("+", None, j - 1))
            j -= 1
    return out[::-1]


def similarity(a: list[str], b: list[str]) -> float:
    """Matched words over the longer reading: 1 is identical, 0 shares nothing."""
    if not a and not b:
        return 1.0
    matched = sum(op == "=" for op, _, _ in align(a, b))
    return matched / max(len(a), len(b))


def errors(reference: list[str], hypothesis: list[str]) -> int:
    return sum(op != "=" for op, _, _ in align(reference, hypothesis))


def edits(heard: list[str], said: list[str]) -> list[tuple[str, str]]:
    """Contiguous blocks where the heard reading differs from the truth."""
    out, block_h, block_s = [], [], []
    for op, i, j in align(heard, said) + [("=", None, None)]:
        if op == "=":
            if block_h or block_s:
                out.append((" ".join(block_h), " ".join(block_s)))
            block_h, block_s = [], []
            continue
        if i is not None:
            block_h.append(heard[i])
        if j is not None:
            block_s.append(said[j])
    return [(h, s) for h, s in out if h]


class Memory:
    def __init__(self):
        self.lexicon: dict[str, Counter] = defaultdict(Counter)
        self.substitutions: dict[str, Counter] = defaultdict(Counter)
        self.examples: list[dict] = []

    # -- learning -------------------------------------------------------------------
    def confirm(self, voice: str | None, words: list[str]) -> None:
        self.lexicon[voice or ""].update(words)

    def teach(self, voice: str | None, heard: str, said: str, readings: list[str]) -> None:
        """One corrected run: its words, its errors, and the run itself for recall."""
        truth = tokens(said)
        self.confirm(voice, truth)
        for reading in dict.fromkeys(readings):
            for wrong, right in edits(tokens(reading), truth):
                self.substitutions[wrong][right] += 1
        self.examples.append({"voice": voice or "", "heard": tokens(heard), "said": said})

    # -- using ----------------------------------------------------------------------
    def known(self, voice: str | None, words: list[str]) -> float:
        """The share of these words this ear has confirmed for this voice (or anyone)."""
        if not words:
            return 0.0
        mine, everyone = self.lexicon.get(voice or "", Counter()), self.lexicon.get("", Counter())
        return sum(1 for w in words if w in mine or w in everyone) / len(words)

    def variants(self, text: str) -> list[tuple[str, int]]:
        """The reading with each learned substitution applied: (text, edits used)."""
        words = tokens(text)
        out = []
        for wrong, rights in self.substitutions.items():
            pattern = wrong.split()
            for start in range(len(words) - len(pattern) + 1):
                if words[start : start + len(pattern)] == pattern:
                    right = rights.most_common(1)[0][0]
                    fixed = words[:start] + right.split() + words[start + len(pattern) :]
                    out.append((" ".join(fixed), rights[right]))
        return out

    def recall(self, voice: str | None, heard: str) -> list[tuple[str, float]]:
        """Corrected runs of this voice whose first reading resembles this one."""
        words = tokens(heard)
        out = []
        for example in self.examples:
            if example["voice"] != (voice or ""):
                continue
            score = similarity(words, example["heard"])
            if score:
                out.append((example["said"], score))
        return out

    # -- persistence ----------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "lexicon": {k: dict(v) for k, v in self.lexicon.items()},
            "substitutions": {k: dict(v) for k, v in self.substitutions.items()},
            "examples": self.examples,
        }

    def load(self, raw: dict) -> None:
        self.lexicon = defaultdict(
            Counter, {k: Counter(v) for k, v in raw.get("lexicon", {}).items()}
        )
        self.substitutions = defaultdict(
            Counter, {k: Counter(v) for k, v in raw.get("substitutions", {}).items()}
        )
        self.examples = list(raw.get("examples", []))
