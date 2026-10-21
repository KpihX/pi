"""Second thoughts: what the ear said of who spoke, said again once it knows better.

A person in the room has a first idea at once, and a second one later: "ah, that was
Paul all along", "these two voices were the same man on two microphones". The ear's
model of who changes its mind the same way, and on its own: two clusters fuse when one
voice explains them better (``model._merge``), a candidate becomes a speaker, a
correction moves windows. Until now a decision, once written, stayed as it was said.

**Revision.** Every window the ear published is kept with what it said (``Second``).
At each silence the ear asks the model again, for each of them, what it would say
now: a learned window, its cluster's voice; a window never learned from (a shadow, a
glimpse, a forgotten candidate), its best explanation under the current model, scored
without learning. Where the answer changed, a ``revision`` is published, with why:

    merged     the voice it was given is now part of another (a fusion, a join)
    corrected  KπX said something about this window since
    learned    the model, with more evidence, explains it otherwise

The model is not told anything by a revision: it is the model's own new state,
read back. So a revision never feeds itself.

**The question worth asking** (value of information). Two voices the model keeps apart,
but not by much: if KπX said "the same", every window of the smaller would change hands.
The expected number of decisions an answer changes is ``min(P, 1 - P) * n``: P the
model's own probability that the smaller voice's main condition belongs to the other
voice (against staying where it is, ``model.voice_scores``), n its windows. It is
asked only if an answer is expected to change at least one decision, the most valuable
first, one at a time, never while a person speaks, and never twice for one pair. No
other number: the budget is KπX's attention, one open question.

The law: no numeric literal but 0 and 1 in this module (``tests/test_law.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from scipy.special import expit

from π_ear.model import VOICE, Model

VOICED = ("known", "new")


def resolve(key: str | None, absorbed: dict[str, str]) -> str | None:
    """Where a voice key went: followed through every fusion, join and rename."""
    seen: set[str] = set()
    while key is not None and key in absorbed and key not in seen:
        seen.add(key)
        key = absorbed[key]
    return key


def now_of(
    model: Model, row: int, x, here: bool = True, best: str | None = None, changed=None
) -> tuple[str, str | None, str | None] | None:
    """(kind, speaker, cluster) the model would say of a window now: ``known`` (a voice),
    ``unknown`` (a candidate or nothing yet) or ``noise``. A learned window is where the
    model put it; any other is scored without learning, if it was heard in the place the
    model is in (``here``: a new voice's prior is centred on the place), else None.

    ``best``, ``changed``: the cluster it was last given, and the clusters that changed
    since. Only they can now take it, so only they are scored again (with "a new voice"),
    the others being where they were: a long session stays cheap to think again about."""
    key = model.where.get(row)
    if key is None:
        if not here:
            return None
        if best is not None and best in model.clusters and changed is not None:
            among = {best, None} | changed
        else:
            among = None  # it was nobody's, or its cluster is gone: all are scored
        scored = model.scores(x, VOICE + ("noise",), among=among)
        key = max(scored, key=scored.get)
    cluster = model.clusters.get(key) if key is not None else None
    if cluster is None:
        return "unknown", None, None
    if cluster.kind == "noise":
        return "noise", None, key
    voice = model.voices.get(cluster.voice) if cluster.voice else None
    return ("known", voice.key, key) if voice is not None else ("unknown", None, key)


def _said(event: dict) -> tuple[bool, str | None]:
    voiced = event["kind"] in VOICED
    return voiced, event["speaker"] if voiced else None


@dataclass
class Second:
    """The ear's published windows of this session, and what changed since."""

    said: dict[int, dict] = field(default_factory=dict)  # row -> its decision, as published
    corrected: set[int] = field(default_factory=set)  # rows a correction named, not yet revised
    asked: set[frozenset] = field(default_factory=set)  # voice pairs asked "the same?"
    open: str | None = None  # the open "same?" question
    best: dict[int, str | None] = field(default_factory=dict)  # row -> its cluster, last said
    sizes: dict[str, tuple] = field(default_factory=dict)  # each cluster, as last thought over

    def keep(self, row: int, event: dict) -> None:
        self.said[row] = event
        self.best[row] = event.get("condition")

    def check(self, model: Model, vec_of, here=lambda event: True) -> list[tuple[int, dict, dict]]:
        """Every published window the model would now say otherwise:
        (row, was, now), ``now`` with its ``why``. ``here(event)``: was it heard in
        the place the model is in now."""
        out = []
        sizes = {k: (c.n, c.kind, c.voice) for k, c in model.clusters.items()}
        changed = {k for k, v in sizes.items() if self.sizes.get(k) != v}
        for row, event in self.said.items():
            best = resolve(self.best.get(row), model.absorbed)
            now = now_of(model, row, vec_of(row), here(event), best, changed)
            if now is None:
                continue
            kind, speaker, self.best[row] = now
            voiced, was = _said(event)
            if (kind == "known") == voiced and speaker == was:
                continue
            if row in self.corrected:
                why = "corrected"
            elif was is not None and resolve(was, model.absorbed) == speaker:
                why = "merged"
            else:
                why = "learned"
            now = {"kind": kind, "speaker": speaker, "why": why}
            out.append((row, {"kind": event["kind"], "speaker": event["speaker"]}, now))
        self.corrected.clear()
        self.sizes = sizes
        return out

    def same_question(self, model: Model, person_of) -> dict | None:
        """The most valuable "are these two voices one?", if one is worth asking."""
        rows = {
            v.key: [r for c in v.conditions for r in model.clusters[c].members]
            for v in model.voices.values()
            if person_of(v.key)
        }
        best = None
        for small, members in rows.items():
            voice = model.voices[small]
            main = max(voice.conditions, key=lambda c: model.clusters[c].n)
            scored = model.voice_scores(model.clusters[main])
            for big, theirs in rows.items():
                pair = frozenset((small, big))
                if big == small or big not in scored or pair in self.asked:
                    continue
                if len(theirs) < len(members):
                    continue  # the smaller one is asked about, joining the larger
                # against where it is now: a voice of its own, or the rest of its voice
                here = scored.get(small, scored[None])
                p = float(expit(scored[big] - here))
                value = min(p, 1 - p) * len(members)
                # worth it: the less likely answer is at least one window's worth of them
                if min(p, 1 - p) < 1 / len(members):
                    continue
                if best is not None and value <= best["value"]:
                    continue
                best = {
                    "speakers": [big, small],
                    "rows": [theirs[-1], members[-1]],
                    "p": p,
                    "value": value,
                }
        return best
