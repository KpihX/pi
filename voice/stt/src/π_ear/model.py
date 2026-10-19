"""The learned ear: who is speaking, decided as a posterior, never as a threshold.

One generative model, and every decision is its argmax:

    speaker means   mu_k ~ vMF(mean0, kappa0)        the population of voices
    an observation  x    ~ vMF(mu_k, kappa)          one voice, many windows
    who it is       z    ~ CRP(alpha)                the number of voices is open

Each observation is scored against every cluster it could belong to and against
"someone new". Nothing is compared to a number: the winner is the most probable
explanation, and "unknown" is simply what a voice heard once is called.

What v0 did with rules, this model does with probability:

* ``k_evidence``: a cluster becomes a speaker when "these windows share one
  voice" is more probable than "they are all different voices". With one
  observation the two are equal, so one noisy window never makes a person;
* ``tau_floor`` / ``z_margin``: the vMF likelihood, with ``kappa`` estimated
  from the clusters themselves;
* the merge invariant: two clusters fuse when one voice explains them better
  than two, which repairs early fragmentation as evidence grows;
* the voicing gate: corrections teach ``noise`` clusters, which then compete
  with the voices for every future observation.

The weights are sufficient statistics: the base ones were measured on public
corpora (``weights/``), the stream adds its own, and ``refresh`` re-estimates
``kappa``, ``kappa0``, ``mean0`` and ``alpha`` after every decision, never before
it. A correction changes the clusters at once (fast clock) and the weights at the
next refresh (slow clock): the error is obeyed now and absorbed into the
parameters, so it does not come back.

Two more things are learned, never set (v1.1):

* **forgetting**: a candidate (a voice heard once) loses weight as it waits.
  The rate is the inverse of the mean time real candidates took to be confirmed
  in this stream; before any confirmation, nothing is forgotten;
* **continuity**: the chance that the next window of a speech run keeps the
  same voice. It learns only from *labelled* transitions (a correction on two
  adjacent windows), never from the model's own decisions, which would feed
  themselves. Before any such label, windows are decided independently.

The law: this module contains no numeric literal but 0 and 1, and compares
nothing to a constant (``tests/test_law.py``).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from π_ear.contract import KEY_BYTES
from π_ear.sphere import crp_alpha, cure_rate, kappa_from_rho, log_c, rho2, unit

VOICE = ("candidate", "speaker")


@dataclass(slots=True)
class Cluster:
    key: str
    kind: str
    total: np.ndarray
    members: list[int] = field(default_factory=list)
    first_seen: float = 0.0
    last_seen: float = 0.0
    label: str | None = None
    heard: int = 0  # windows absorbed from public data, held only as a sum

    @property
    def n(self) -> int:
        return len(self.members) + self.heard

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "kind": self.kind,
            "label": self.label,
            "members": self.members,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "heard": self.heard,
            "total": self.total.tolist(),
        }

    @classmethod
    def from_dict(cls, raw: dict) -> Cluster:
        return cls(
            key=raw["key"],
            kind=raw["kind"],
            total=np.asarray(raw["total"], dtype=np.float64),
            members=list(raw["members"]),
            first_seen=raw["first_seen"],
            last_seen=raw["last_seen"],
            label=raw["label"],
            heard=raw.get("heard", 0),
        )


@dataclass(slots=True)
class Weights:
    """Base sufficient statistics, measured on public data, plus what they imply."""

    dim: int
    within_norm2: float
    within_count: float
    within_pairs: float
    between_sum: np.ndarray
    between_count: float
    tables_obs: float
    tables_k: float
    source: str = ""
    noise: list[dict] = field(default_factory=list)
    kappa: float = 0.0
    kappa0: float = 0.0
    mean0: np.ndarray | None = None
    alpha: float = 0.0

    def to_dict(self) -> dict:
        return {
            "dim": self.dim,
            "within_norm2": self.within_norm2,
            "within_count": self.within_count,
            "within_pairs": self.within_pairs,
            "between_sum": self.between_sum.tolist(),
            "between_count": self.between_count,
            "tables_obs": self.tables_obs,
            "tables_k": self.tables_k,
            "source": self.source,
            "noise": self.noise,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> Weights:
        return cls(
            dim=raw["dim"],
            within_norm2=raw["within_norm2"],
            within_count=raw["within_count"],
            within_pairs=raw["within_pairs"],
            between_sum=np.asarray(raw["between_sum"], dtype=np.float64),
            between_count=raw["between_count"],
            tables_obs=raw["tables_obs"],
            tables_k=raw["tables_k"],
            source=raw.get("source", ""),
            noise=raw.get("noise", []),
        )


def fit_weights(groups: dict[str, list[list[np.ndarray]]], source: str) -> Weights:
    """Base statistics from labelled public windows: the "good bases" of the ear.

    ``groups`` maps a voice to its recordings, each a list of windows. ``kappa``
    is measured on pairs of windows from two DIFFERENT recordings of one voice
    only, because the ear will meet a voice across rooms and days, not inside
    one sentence.
    """
    dim = len(next(iter(groups.values()))[0][0])
    cross, pairs, speakers, count = 0.0, 0.0, np.zeros(dim), 0
    for recordings in groups.values():
        sums = [sum(r, np.zeros(dim)) for r in recordings]
        sizes = [len(r) for r in recordings]
        total = sum(sums, np.zeros(dim))
        cross += float(total @ total) - sum(float(s @ s) for s in sums)
        pairs += sum(sizes) * sum(sizes) - sum(m * m for m in sizes)
        speakers = speakers + unit(total)
        count += sum(sizes)
    return Weights(
        dim=dim,
        within_norm2=cross,
        within_count=0,
        within_pairs=pairs,
        between_sum=speakers,
        between_count=len(groups),
        tables_obs=count,
        tables_k=len(groups),
        source=source,
    )


def centroid_key(total: np.ndarray) -> str:
    """A short identity key for a committed centroid. Computed once, then kept."""
    direction = unit(np.asarray(total, dtype=np.float64))
    digest = hashlib.blake2b(direction.astype(np.float32).tobytes(), digest_size=KEY_BYTES)
    return digest.hexdigest()


class Model:
    def __init__(self, weights: Weights, vec_of: Callable[[int], np.ndarray]):
        self.base = weights
        self.dim = weights.dim
        self.vec_of = vec_of
        self.clusters: dict[str, Cluster] = {}
        self.where: dict[int, str] = {}
        self.cannot: set[tuple[int, int]] = set()
        self.absorbed: dict[str, str] = {}
        self.pairs: dict[str, float] = {"dot": 0.0, "count": 0.0}
        self.confirm: dict[str, float] = {"gaps": 0.0, "count": 0.0, "forgotten": 0.0}
        self.cure: tuple[float, float] | None = None
        self.turns: dict[str, float] = {"stay": 0.0, "switch": 0.0}
        self.after: dict[int, int] = {}  # row -> the previous learned row of its run
        self.now = 0.0
        for index, raw in enumerate(weights.noise):
            key = f"base-noise-{index}"
            total = np.asarray(raw["total"], dtype=np.float64)
            self.clusters[key] = Cluster(key, "noise", total, heard=raw["heard"])
        self.refresh()

    # -- the parameters -------------------------------------------------------
    def refresh(self) -> None:
        """Re-estimate every parameter from the base statistics plus the stream."""
        base = self.base
        # kappa, the spread of one voice, learns only from pairs KNOWN to share a
        # voice: the labelled public windows of the base, then every "same"
        # correction. Never from the clusters this model built itself: tighter
        # clusters would raise kappa, which would cut tighter clusters, the
        # self-feeding loop that made v0 create 56 identities. Not from two
        # consecutive windows of one run either: measured, they are closer to each
        # other than one voice is to itself across days, and kappa went from 283 to
        # 527 in five minutes of a simulated room.
        # A pair (x, y) adds ||x + y||^2 = 2 + 2 x.y over 2 unit vectors and 2
        # ordered pairs, which is what rho2 expects.
        twice = self.pairs["count"] + self.pairs["count"]
        self.kappa = kappa_from_rho(
            self.dim,
            math.sqrt(
                rho2(
                    base.within_norm2 + twice + self.pairs["dot"] + self.pairs["dot"],
                    base.within_count + twice,
                    base.within_pairs + twice,
                )
            ),
        )

        speakers = [c for c in self.clusters.values() if c.kind == "speaker"]

        directions = base.between_sum + sum((unit(c.total) for c in speakers), np.zeros(base.dim))
        k = base.between_count + len(speakers)
        between = float(directions @ directions)
        self.kappa0 = kappa_from_rho(self.dim, math.sqrt(rho2(between, k, k * (k - 1))))
        self.mean0 = unit(directions)
        self.prior = self.kappa0 * self.mean0

        self.cure = self._confirmation()
        voices = [c for c in self.clusters.values() if c.kind in VOICE]
        alive = [self.survival(c) for c in voices]
        self.alpha = crp_alpha(
            base.tables_obs + sum(c.n * w for c, w in zip(voices, alive, strict=True)),
            base.tables_k + sum(alive),
        )
        self.log_alpha = math.log(self.alpha)
        self.log_c_kappa = log_c(self.dim, self.kappa)
        self.log_c_kappa0 = log_c(self.dim, self.kappa0)

    def _lc(self, total: np.ndarray) -> float:
        """log C_d(||kappa0 mean0 + kappa total||): the posterior of a cluster mean."""
        return log_c(self.dim, float(np.linalg.norm(self.prior + self.kappa * total)))

    def predictive(self, total: np.ndarray, x: np.ndarray) -> float:
        """log p(x | the windows already in a cluster), the mean integrated out."""
        return self.log_c_kappa + self._lc(total) - self._lc(total + x)

    # -- the decision (fast clock) ---------------------------------------------
    def _confirmation(self):
        """(pi, lam): how many candidates are real voices, and how fast they confirm.

        Survival analysis with a cured fraction (``sphere.cure_rate``), on this
        stream's own candidates. Before the first confirmation: None, nothing fades.
        """
        ages = [self.now - c.first_seen for c in self.clusters.values() if c.kind == "candidate"]
        return cure_rate(
            self.confirm["gaps"], self.confirm["count"], ages, self.confirm["forgotten"]
        )

    def survival(self, cluster: Cluster) -> float:
        """A candidate's weight: the posterior that it is a voice still to be confirmed."""
        if cluster.kind != "candidate" or self.cure is None:
            return 1.0
        pi, lam = self.cure
        alive = pi * math.exp(-lam * (self.now - cluster.first_seen))
        return alive / (alive + 1 - pi)

    def stay(self) -> float | None:
        """P(the next window of a run keeps its voice), from labelled transitions only."""
        stay, switch = self.turns["stay"], self.turns["switch"]
        return stay / (stay + switch) if stay and switch else None

    def scores(
        self, x: np.ndarray, kinds: tuple[str, ...], previous: str | None = None
    ) -> dict[str | None, float]:
        """Log posterior of every explanation of ``x``, up to one constant. None is "new"."""
        weights: dict[str | None, float] = {
            key: c.n * self.survival(c) for key, c in self.clusters.items() if c.kind in kinds
        }
        weights[None] = self.alpha
        weights = {k: w for k, w in weights.items() if w}
        stay = self.stay()
        if stay is not None and previous in weights:
            mass = sum(weights.values())
            weights = {k: (1 - stay) * w / mass for k, w in weights.items()}
            weights[previous] += stay
        zero = np.zeros(self.dim)
        return {
            key: math.log(w) + self.predictive(self.clusters[key].total if key else zero, x)
            for key, w in weights.items()
        }

    def observe(
        self, row: int, x: np.ndarray, t: float, learn: bool, follows: int | None = None
    ) -> dict:
        """Decide who ``x`` is. If ``learn``, the decision also becomes evidence.

        ``follows`` is the previous learned window of the same speech run, if any:
        the continuity prior uses it once labelled transitions exist.
        """
        self.now = max(self.now, t)
        previous = self.where.get(follows) if follows is not None else None
        scored = self.scores(x, VOICE + ("noise",), previous)
        ranked = sorted(scored.items(), key=lambda item: item[1], reverse=True)
        top = ranked[0][1]
        mass = sum(math.exp(value - top) for _, value in ranked)
        choice = ranked[0][0]
        runner = ranked[1] if ranked[1:] else (None, top)
        speakers = {k for k, c in self.clusters.items() if c.kind == "speaker"}
        if learn:
            if follows is not None:
                self.after[row] = follows
            choice = self._add(choice, row, x, t, "candidate")
            choice = self._merge(self._promote(choice))
            self._forget()
            self.refresh()
        cluster = self.clusters.get(choice) if choice is not None else None
        kind = self._kind(cluster, speakers)
        return {
            "kind": kind,
            "speaker": cluster.key if cluster is not None and kind != "unknown" else None,
            "label": cluster.label if cluster is not None else None,
            "p": math.exp(ranked[0][1] - top) / mass,
            "runner_up": {"id": runner[0], "p": math.exp(runner[1] - top) / mass},
            "margin": top - runner[1],  # nats between the winner and the runner up
            "evidence": cluster.n if cluster is not None else 0,
            "learned": learn,
        }

    @staticmethod
    def _kind(cluster: Cluster | None, speakers: set[str]) -> str:
        if cluster is None or cluster.kind == "candidate":
            return "unknown"
        if cluster.kind == "noise":
            return "noise"
        return "known" if cluster.key in speakers else "new"

    # -- learning --------------------------------------------------------------
    def _add(self, key: str | None, row: int, x: np.ndarray, t: float, kind: str) -> str:
        if key is None:
            key = f"c{row}"
            self.clusters[key] = Cluster(key, kind, np.zeros(self.dim), first_seen=t)
        cluster = self.clusters[key]
        cluster.total = cluster.total + x
        cluster.members.append(row)
        cluster.last_seen = t
        self.where[row] = key
        return key

    def _remove(self, row: int) -> None:
        key = self.where.pop(row, None)
        if key is None:
            return
        cluster = self.clusters[key]
        cluster.members.remove(row)
        cluster.total = cluster.total - self.vec_of(row)
        if not cluster.n:
            del self.clusters[key]

    def _one_voice(self, members: list[int]) -> tuple[float, float]:
        """Log probability of "one voice" and of "all different", same windows."""
        total = sum((self.vec_of(r) for r in members), np.zeros(self.dim))
        n = len(members)
        one = math.lgamma(n) - self._lc(total) - (n - 1) * self.log_c_kappa0
        many = (n - 1) * self.log_alpha - sum(self._lc(self.vec_of(r)) for r in members)
        return one, many

    def _promote(self, key: str) -> str:
        """A candidate becomes a speaker when one voice explains it best."""
        cluster = self.clusters[key]
        if cluster.kind != "candidate":
            return key
        one, many = self._one_voice(cluster.members)
        if many < one:
            cluster.kind = "speaker"
            self.confirm["gaps"] += cluster.last_seen - cluster.first_seen
            self.confirm["count"] += 1
            self._rename(key, centroid_key(cluster.total))
        return cluster.key

    def _forget(self) -> None:
        """Drop the candidates whose weight has run out (an exact zero, not a cut)."""
        for key in [k for k, c in self.clusters.items() if not self.survival(c)]:
            self.confirm["forgotten"] += 1
            for row in self.clusters.pop(key).members:
                self.where.pop(row, None)

    def _rename(self, old: str, new: str) -> None:
        cluster = self.clusters.pop(old)
        cluster.key = new
        self.clusters[new] = cluster
        for row in cluster.members:
            self.where[row] = new
        self.absorbed[old] = new

    def _linked(self, a: Cluster, b: Cluster) -> bool:
        rows = set(b.members)
        return any((r, s) in self.cannot or (s, r) in self.cannot for r in a.members for s in rows)

    def merge_gain(self, a: Cluster, b: Cluster) -> tuple[float, float]:
        """Log probability of "one voice" against "two voices", for two clusters."""
        one = math.lgamma(a.n + b.n) - self._lc(a.total + b.total) - self.log_c_kappa0
        two = math.lgamma(a.n) + math.lgamma(b.n) + self.log_alpha - self._lc(a.total)
        return one, two - self._lc(b.total)

    def _merge(self, key: str) -> str:
        """Fuse ``key`` with every cluster that one voice explains better than two."""
        while True:
            here = self.clusters[key]
            if here.kind not in VOICE:
                return key  # noise never fuses into a voice, nor a voice into noise
            best, best_gain = None, None
            for other in self.clusters.values():
                if other is here or other.kind not in VOICE or self._linked(here, other):
                    continue
                one, two = self.merge_gain(here, other)
                gain = one - two
                if two < one and (best_gain is None or best_gain < gain):
                    best, best_gain = other, gain
            if best is None:
                return key
            key = self._fuse(here, best)

    def _fuse(self, a: Cluster, b: Cluster) -> str:
        """Keep the older speaker's key: a name, once given, survives a merge."""
        keep, drop = sorted((a, b), key=lambda c: (c.kind != "speaker", c.first_seen))
        keep.total = keep.total + drop.total
        keep.members = sorted(keep.members + drop.members)
        keep.first_seen = min(keep.first_seen, drop.first_seen)
        keep.last_seen = max(keep.last_seen, drop.last_seen)
        keep.label = keep.label or drop.label
        keep.heard += drop.heard
        for row in drop.members:
            self.where[row] = keep.key
        del self.clusters[drop.key]
        self.absorbed[drop.key] = keep.key
        self._promote(keep.key)
        return self.where[keep.members[0]]

    def hear_noise(self, x: np.ndarray) -> None:
        """A public window known not to be a voice: the base learns what noise is like."""
        scored = self.scores(x, ("noise",))
        key = max(scored, key=scored.get)
        if key is None:
            key = f"base-noise-{sum(c.kind == 'noise' for c in self.clusters.values())}"
            self.clusters[key] = Cluster(key, "noise", np.zeros(self.dim))
        self.clusters[key].total = self.clusters[key].total + x
        self.clusters[key].heard += 1

    def noise_weights(self) -> list[dict]:
        return [
            {"total": c.total.tolist(), "heard": c.heard}
            for c in self.clusters.values()
            if c.kind == "noise"
        ]

    # -- corrections -------------------------------------------------------------
    def correct(self, kind: str, rows: list[int], label: str | None = None) -> None:
        """Obey one correction now. ``refresh`` then absorbs it into the weights."""
        known = [r for r in rows if r in self.where]
        adjacent = [(a, b) for a in rows for b in rows if self.after.get(b) == a]
        if kind == "same":
            self.turns["stay"] += len(adjacent)
        elif kind == "different":
            self.turns["switch"] += len(adjacent)
        if kind == "name":
            for row in known:
                self.clusters[self.where[row]].label = label
        elif kind == "same":
            # Every window of one cluster against every window of the other is now a
            # labelled same-voice pair, and the dot products of all of them are one
            # product of the two sums. One sentence from KπX, many pairs of evidence.
            self.cannot -= {(a, b) for a in rows for b in rows}
            keys = list(dict.fromkeys(self.where[r] for r in known))
            while keys[1:]:
                a, b = self.clusters[keys[0]], self.clusters[keys[1]]
                self.pairs["dot"] += float(a.total @ b.total)
                self.pairs["count"] += a.n * b.n
                self._fuse(a, b)
                keys = list(dict.fromkeys(self.where[r] for r in known))
        elif kind == "different":
            self.cannot |= {(a, b) for a in rows for b in rows if a != b}
            self._split(known)
        elif kind == "not_voice":
            for row in known:
                when = self.clusters[self.where[row]].last_seen
                self._remove(row)
                x = self.vec_of(row)
                scored = self.scores(x, ("noise",))
                self._add(max(scored, key=scored.get), row, x, when, "noise")
        self.refresh()

    def _split(self, seeds: list[int]) -> None:
        """Re-cut the cluster(s) holding seeds that may not share a voice."""
        groups: dict[str, list[int]] = {}
        for row in seeds:
            groups.setdefault(self.where[row], []).append(row)
        for key, inside in groups.items():
            if len(inside) == 1:
                continue
            cluster = self.clusters.pop(key)
            stamp = (cluster.first_seen, cluster.last_seen)
            for row in cluster.members:
                self.where.pop(row, None)
            parts = {
                row: Cluster(f"c{row}", "candidate", self.vec_of(row).copy(), [row], *stamp)
                for row in inside
            }
            for row in inside:
                self.where[row] = parts[row].key
            for row in cluster.members:
                if row in parts:
                    continue
                x = self.vec_of(row)
                free = [p for p in parts.values() if not self._forbids(p, row)]
                best = max(free or parts.values(), key=lambda p: self.predictive(p.total, x))
                best.total = best.total + x
                best.members.append(row)
                self.where[row] = best.key
            ranked = sorted(parts.values(), key=lambda p: p.n, reverse=True)
            for part in ranked:
                self.clusters[part.key] = part
            heir = ranked[0]
            heir.label = cluster.label
            if cluster.kind == "speaker":
                heir.kind = "speaker"
                self._rename(heir.key, key)
            for part in ranked:
                self._promote(part.key)

    def _forbids(self, part: Cluster, row: int) -> bool:
        return any((row, r) in self.cannot or (r, row) in self.cannot for r in part.members)

    # -- persistence --------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "clusters": [c.to_dict() for c in self.clusters.values()],
            "cannot": sorted(self.cannot),
            "absorbed": self.absorbed,
            "pairs": self.pairs,
            "confirm": self.confirm,
            "turns": self.turns,
            "after": [[row, previous] for row, previous in self.after.items()],
            "now": self.now,
        }

    def load(self, raw: dict) -> None:
        self.clusters = {c["key"]: Cluster.from_dict(c) for c in raw.get("clusters", [])}
        self.where = {row: c.key for c in self.clusters.values() for row in c.members}
        self.cannot = {tuple(pair) for pair in raw.get("cannot", [])}
        self.absorbed = dict(raw.get("absorbed", {}))
        self.pairs = dict(raw.get("pairs", self.pairs))
        self.confirm = dict(raw.get("confirm", self.confirm))
        self.turns = dict(raw.get("turns", self.turns))
        self.after = {row: previous for row, previous in raw.get("after", [])}
        self.now = raw.get("now", self.now)
        self.refresh()

    def summary(self) -> dict:
        return {
            "kappa": self.kappa,
            "kappa0": self.kappa0,
            "alpha": self.alpha,
            "speakers": sum(c.kind == "speaker" for c in self.clusters.values()),
            "candidates": sum(c.kind == "candidate" for c in self.clusters.values()),
            "noise": sum(c.kind == "noise" for c in self.clusters.values()),
            "stay": self.stay(),
            "confirmations": self.confirm["count"],
            "candidates_real": self.cure[0] if self.cure else None,
            "confirm_rate_per_s": self.cure[1] if self.cure else None,
        }

    def explain(self, x: np.ndarray, previous: str | None = None) -> list[dict]:
        """Every explanation of ``x`` under the current state, best first: the why."""
        scored = self.scores(x, VOICE + ("noise",), previous)
        top = max(scored.values())
        out = []
        for key, value in sorted(scored.items(), key=lambda item: item[1], reverse=True):
            cluster = self.clusters.get(key) if key else None
            out.append(
                {
                    "id": key,
                    "kind": cluster.kind if cluster else "new",
                    "label": cluster.label if cluster else None,
                    "evidence": cluster.n if cluster else 0,
                    "behind_nats": top - value,
                }
            )
        return out
