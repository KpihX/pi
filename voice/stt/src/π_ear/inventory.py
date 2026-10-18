"""The inventory: the part that learns while it listens.

This is the centre of gravity of the whole POC, and the thing the first pass
missed. The encoder is a sensor; *this* is the system. It holds three kinds of
object and it updates them on every observation:

* **candidates**: observations that do not clearly belong to anyone yet. They
  accumulate evidence and they are completely disposable;
* **registered speakers**: candidates that crossed the evidence bar. Their
  identity is a short hash of the committed centroid, with an explicit version;
* **the boundary**: an admission criterion MEASURED from the sensor's own
  outputs, never counted from the inventory. A window belongs to a known voice
  when it is at least ``tau_floor`` close to the closest one AND closer to it than
  to the runner-up by ``z_margin`` times the spread of the margins this sensor has
  been producing. There is no constant to hand-tune, and no path from the size of
  the inventory back into who gets admitted.

Three traps are deliberately avoided:

1. a single noisy observation can never create a speaker, because registration
   needs ``k_evidence`` mutually compatible observations;
2. the identity key is computed from the **committed** centroid and never
   recomputed after the fact. A centroid keeps moving, so a key recomputed on
   every observation would silently change identity overnight. Hence ``version``;
3. **two entries never describe a voice this sensor cannot tell apart.** Before a
   candidate becomes a speaker it is put through the same admission test a window
   goes through: if it passes, the candidate IS that voice, so it is absorbed
   instead of registered. Measured 2026-10-19, before this rule existed: one real
   voice became twelve entries, all of them at a cosine of 0.9999 from each other,
   and no threshold could have repaired that. An absorbed key is never lost: it is
   mapped to the surviving one in ``absorbed``, because a re-attribution after the
   fact is normal in this record.
"""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from π_ear.physics import cosine, identity_vector, percentile


def centroid_key(vec: np.ndarray, version: int) -> str:
    """A short, stable identity key for a committed centroid.

    Quantised on purpose: the key must be stable across processes and machines,
    and float noise must not change who someone is. The version is part of the
    input, so a re-commitment is a new key rather than a silent mutation.
    """
    quantised = np.round(np.asarray(vec, dtype=np.float64) * 1000.0).astype(np.int32)
    payload = quantised.tobytes() + version.to_bytes(2, "little", signed=False)
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


@dataclass(slots=True)
class Entry:
    """One candidate, or one registered speaker."""

    id: str
    kind: str
    vec: np.ndarray
    evidence: int = 0
    version: int = 1
    first_seen: float = 0.0
    last_seen: float = 0.0
    f0_history: deque[float] = field(default_factory=lambda: deque(maxlen=200))
    levels: deque[float] = field(default_factory=lambda: deque(maxlen=200))
    label: str | None = None

    def absorb(self, vec: np.ndarray) -> None:
        """Running mean, weighted by evidence: the centroid moves, slowly."""
        self.vec = (self.vec * self.evidence + vec) / (self.evidence + 1)
        norm = float(np.linalg.norm(self.vec))
        if norm > 0.0:
            self.vec = self.vec / norm
        self.evidence += 1

    def absorb_entry(self, other: Entry) -> None:
        """Merge another entry into this one, weighted by evidence.

        Two entries that describe the same voice are one entry carrying the
        evidence and the history of both. Nothing is thrown away here: the pitch
        history, the levels and the time span of the absorbed entry are all kept,
        so an unmerge remains possible from the data.
        """
        total = self.evidence + other.evidence
        if total > 0:
            merged = (self.vec * self.evidence + other.vec * other.evidence) / total
            norm = float(np.linalg.norm(merged))
            if norm > 0.0:
                merged = merged / norm
            self.vec = merged
        self.evidence = total
        self.first_seen = min(self.first_seen, other.first_seen)
        self.last_seen = max(self.last_seen, other.last_seen)
        self.f0_history.extend(other.f0_history)
        self.levels.extend(other.levels)

    def to_dict(self, now: float) -> dict[str, Any]:
        pitch = float(np.median(self.f0_history)) if self.f0_history else 0.0
        level = float(np.median(self.levels)) if self.levels else -120.0
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "evidence": self.evidence,
            "version": self.version,
            "first_seen_s": round(self.first_seen, 2),
            "last_seen_s": round(self.last_seen, 2),
            "silent_for_s": round(max(0.0, now - self.last_seen), 2),
            "f0_med_hz": round(pitch, 1),
            "level_med_dbfs": round(level, 1),
        }


class Inventory:
    def __init__(self, cfg):
        self.cfg = cfg
        self.registered: dict[str, Entry] = {}
        self.candidates: dict[str, Entry] = {}
        self.counters = {
            "observations": 0,
            "silent": 0,
            "assigned_known": 0,
            "registered_new": 0,
            "merged_into_existing": 0,
            "candidates_created": 0,
            "candidates_evicted": 0,
            "windows": 0,
        }
        self._cand_seq = 0
        # The sensor's own noise, as it is being produced: the margin between the
        # best and the runner-up match for every window that was scored. This is
        # what calibrates the ear, and it is the reason no threshold has to be
        # hand-tuned.
        self.margins: deque[float] = deque(maxlen=int(cfg.margin_samples))
        # Identity is not deleted: an absorbed key maps to the entry that absorbed
        # it, so a reader who saw the old key can follow it instead of losing it.
        self.absorbed: dict[str, str] = {}

    # -- the measured boundary ------------------------------------------------
    @property
    def tau(self) -> float:
        """The cosine floor in force: how close the closest voice must be."""
        return float(self.cfg.tau_floor)

    @property
    def sigma(self) -> float:
        """The spread of the margins this sensor has been producing.

        The ear measuring its own noise. Until there is enough history for a
        spread to mean anything, it is 0.0 and the criterion rests on the floor.
        """
        if len(self.margins) < 8:
            return 0.0
        return float(np.std(np.asarray(self.margins, dtype=np.float64)))

    @property
    def margin_required(self) -> float:
        """How far ahead of the runner-up the winner has to be, right now."""
        return float(max(self.cfg.margin_floor, self.cfg.z_margin * self.sigma))

    @property
    def separation(self) -> float | None:
        """The largest cosine between any two registered centroids.

        The honest health indicator of the inventory: near 0.0 the entries are
        distinct voices, near 1.0 they are one voice wearing several keys, and no
        threshold can repair that. ``None`` means there are fewer than two entries,
        so the question does not arise: a record field that answered 1.0 in that
        case would be saying "identical" when it means "nothing to compare".

        Above 256 entries the audit is skipped and -1.0 is returned: the cost is
        quadratic and the answer would not be actionable.
        """
        keys = list(self.registered)
        if len(keys) < 2:
            return None
        if len(keys) > 256:
            return -1.0
        worst = 0.0
        for index, first in enumerate(keys):
            for second in keys[index + 1 :]:
                worst = max(worst, cosine(self.registered[first].vec, self.registered[second].vec))
        return float(worst)

    # -- one observation ------------------------------------------------------
    def tick(self, features: dict, t: float) -> dict[str, Any]:
        """Decide what this window is, update the state, return the decision.

        The decision is returned, never hidden: the event record keeps the score,
        the runner-up score and the threshold that was in force, so a threshold
        can later be tuned on data instead of on opinion.
        """
        self.counters["windows"] += 1
        vec = identity_vector(features)

        if (
            features.get("voiced_ratio", 0.0) < self.cfg.voiced_floor
            or features.get("f0_hz", 0.0) <= 0.0
        ):
            self.counters["silent"] += 1
            return {
                "kind": "silent",
                "id": None,
                "cos_best": None,
                "cos_second": None,
                "margin": None,
                "margin_req": round(self.margin_required, 5),
                "tau": round(self.tau, 4),
            }

        self.counters["observations"] += 1
        # Read the bar BEFORE the observation is counted: a window must never be
        # judged by a threshold it has just moved itself.
        required = self.margin_required
        key, cos_best, cos_second, margin = self._match(vec)

        if margin is not None:
            self.margins.append(margin)

        # 1. does it belong to someone we already know?
        if key is not None:
            self._absorb(self.registered[key], vec, features, t)
            self.counters["assigned_known"] += 1
            return {
                "kind": "known",
                "id": key,
                "cos_best": round(cos_best, 4),
                "cos_second": round(cos_second, 4) if cos_second is not None else None,
                "margin": round(margin, 5) if margin is not None else None,
                "margin_req": round(required, 5),
                "tau": round(self.tau, 4),
            }

        # 2. does it join a candidate?
        best_candidate, best_cos = self._best_candidate(vec)
        if best_candidate is not None and best_cos >= self.cfg.tau_candidate:
            self._absorb(best_candidate, vec, features, t)
            if best_candidate.evidence >= self.cfg.k_evidence:
                promoted, merged = self._promote(best_candidate)
                if merged:
                    self.counters["merged_into_existing"] += 1
                    return {
                        "kind": "merged",
                        "id": promoted.id,
                        "cos_best": round(best_cos, 4),
                        "cos_second": round(cos_second, 4) if cos_second is not None else None,
                        "margin": round(margin, 5) if margin is not None else None,
                        "margin_req": round(required, 5),
                        "tau": round(self.tau, 4),
                        "evidence": promoted.evidence,
                    }
                self.counters["registered_new"] += 1
                return {
                    "kind": "new",
                    "id": promoted.id,
                    "cos_best": round(best_cos, 4),
                    "cos_second": round(cos_second, 4) if cos_second is not None else None,
                    "margin": round(margin, 5) if margin is not None else None,
                    "margin_req": round(required, 5),
                    "tau": round(self.tau, 4),
                    "evidence": promoted.evidence,
                }
            return {
                "kind": "candidate",
                "id": best_candidate.id,
                "cos_best": round(best_cos, 4),
                "cos_second": round(cos_second, 4) if cos_second is not None else None,
                "margin": round(margin, 5) if margin is not None else None,
                "margin_req": round(required, 5),
                "tau": round(self.tau, 4),
                "evidence": best_candidate.evidence,
                "needed": self.cfg.k_evidence,
            }

        # 3. a new candidate, evicting the least evidenced one if the pool is full
        candidate = self._new_candidate(vec, features, t)
        return {
            "kind": "unknown",
            "id": candidate.id,
            "cos_best": round(cos_best, 4),
            "cos_second": round(cos_second, 4) if cos_second is not None else None,
            "margin": round(margin, 5) if margin is not None else None,
            "margin_req": round(required, 5),
            "tau": round(self.tau, 4),
            "evidence": candidate.evidence,
            "needed": self.cfg.k_evidence,
        }

    # -- internals ------------------------------------------------------------
    def _match(self, vec: np.ndarray) -> tuple[str | None, float, float | None, float | None]:
        """Score a vector against the registered cohort, and decide.

        One test, used twice: on a window, and on a candidate asking to become a
        speaker. Two objects this sensor cannot tell apart must never become two
        entries, so the rule that says "this window is that voice" also says "this
        candidate is that voice, do not create anyone".

        With fewer than two registered speakers there is no runner-up, so no margin
        can be measured and the cosine floor is the only guard. That is a real
        limit of the criterion and it is written down rather than hidden.
        """
        if not self.registered:
            return None, 0.0, None, None
        scored = sorted(
            ((cosine(vec, entry.vec), key) for key, entry in self.registered.items()),
            reverse=True,
        )
        best_key = scored[0][1]
        cos_best = float(scored[0][0])
        cos_second = float(scored[1][0]) if len(scored) > 1 else None
        margin = None if cos_second is None else float(cos_best - cos_second)
        admitted = cos_best >= self.tau and (margin is None or margin >= self.margin_required)
        return (best_key if admitted else None), cos_best, cos_second, margin

    def _absorb(self, entry: Entry, vec: np.ndarray, features: dict, t: float) -> None:
        entry.absorb(vec)
        entry.last_seen = t
        if features.get("f0_hz"):
            entry.f0_history.append(float(features["f0_hz"]))
        if features.get("rms_dbfs") is not None:
            entry.levels.append(float(features["rms_dbfs"]))

    def _best_candidate(self, vec: np.ndarray) -> tuple[Entry | None, float]:
        best: Entry | None = None
        best_cos = 0.0
        for entry in self.candidates.values():
            value = cosine(vec, entry.vec)
            if value > best_cos:
                best, best_cos = entry, value
        return best, best_cos

    def _new_candidate(self, vec: np.ndarray, features: dict, t: float) -> Entry:
        if len(self.candidates) >= self.cfg.max_candidates:
            weakest = min(
                self.candidates.values(), key=lambda entry: (entry.evidence, entry.last_seen)
            )
            del self.candidates[weakest.id]
            self.counters["candidates_evicted"] += 1
        self._cand_seq += 1
        entry = Entry(id=f"cand_{self._cand_seq}", kind="candidate", vec=vec, first_seen=t)
        self._absorb(entry, vec, features, t)
        self.candidates[entry.id] = entry
        self.counters["candidates_created"] += 1
        return entry

    def _promote(self, candidate: Entry) -> tuple[Entry, bool]:
        """Make a speaker out of a candidate, unless that speaker already exists.

        The key is computed here, once, on the committed centroid. The invariant
        comes first: if the candidate passes the same admission test a window
        passes, this sensor cannot tell it apart from a voice it already holds, and
        the honest outcome is to absorb it instead of multiplying entries. This is
        the rule that stops one voice from becoming eight entries that all describe
        it, which is exactly what was measured on real audio before it existed.
        """
        self.candidates.pop(candidate.id, None)
        key, _cos_best, _cos_second, _margin = self._match(candidate.vec)
        if key is not None:
            target = self.registered[key]
            self.absorbed[candidate.id] = target.id
            if len(self.absorbed) > 256:
                self.absorbed.pop(next(iter(self.absorbed)))
            target.absorb_entry(candidate)
            return target, True
        key = centroid_key(candidate.vec, 1)
        while key in self.registered:
            key = centroid_key(candidate.vec, candidate.version + 1)
            candidate.version += 1
        candidate.id = key
        candidate.kind = "registered"
        candidate.version = 1
        self.registered[key] = candidate
        return candidate, False

    # -- rendering ------------------------------------------------------------
    def as_dict(self, now: float) -> dict[str, Any]:
        return {
            "tau_now": round(self.tau, 4),
            "margin_now": round(self.margin_required, 5),
            "margin_sigma": round(self.sigma, 5),
            "separation": None if self.separation is None else round(self.separation, 4),
            "registered": [entry.to_dict(now) for entry in self.registered.values()],
            "candidates": [entry.to_dict(now) for entry in self.candidates.values()],
            "absorbed": dict(self.absorbed),
            "counters": dict(self.counters),
        }

    def f0_percentile(self, speaker_id: str, value: float) -> float:
        entry = self.registered.get(speaker_id) or self.candidates.get(speaker_id)
        if entry is None or not entry.f0_history:
            return 50.0
        return round(percentile(value, entry.f0_history), 1)
