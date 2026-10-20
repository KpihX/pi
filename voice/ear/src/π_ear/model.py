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
from π_ear.sphere import (
    choice_probabilities,
    crp_alpha,
    cure_rate,
    fit_choice,
    kappa_from_rho,
    log_c,
    rho2,
    rho_of,
    unit,
)

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
    voice: str | None = None  # the voice this condition belongs to (speakers only)

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
            "voice": self.voice,
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
            voice=raw.get("voice"),
        )


@dataclass(slots=True)
class Voice:
    """One person (or one device): the conditions it was heard in, and what is known of it."""

    key: str
    conditions: list[str] = field(default_factory=list)
    label: str | None = None
    source: str | None = None  # "person" | "device", once known
    first_seen: float = 0.0

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "conditions": self.conditions,
            "label": self.label,
            "source": self.source,
            "first_seen": self.first_seen,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> Voice:
        return cls(**raw)


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
    conditions_norm2: float = 0.0  # recording means of one voice: how far its conditions drift
    conditions_count: float = 0.0
    conditions_pairs: float = 0.0
    places_norm2: float = 0.0  # condition means of DIFFERENT voices in one place: kappa_in
    places_count: float = 0.0
    places_pairs: float = 0.0
    place_sum: np.ndarray | None = None  # sum of the unit means of the places: kappa_p
    place_count: float = 0.0
    source: str = ""
    noise: list[dict] = field(default_factory=list)
    spectra: dict = field(default_factory=dict)  # person/device spectral shapes (source.py)
    reader: dict = field(default_factory=dict)  # the reader's base weights (words.py)
    calibration: list = field(default_factory=list)  # [margin, right] on public windows (Q1)
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
            "conditions_norm2": self.conditions_norm2,
            "conditions_count": self.conditions_count,
            "conditions_pairs": self.conditions_pairs,
            "places_norm2": self.places_norm2,
            "places_count": self.places_count,
            "places_pairs": self.places_pairs,
            "place_sum": None if self.place_sum is None else self.place_sum.tolist(),
            "place_count": self.place_count,
            "source": self.source,
            "noise": self.noise,
            "spectra": self.spectra,
            "reader": self.reader,
            "calibration": self.calibration,
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
            conditions_norm2=raw.get("conditions_norm2", 0.0),
            conditions_count=raw.get("conditions_count", 0.0),
            conditions_pairs=raw.get("conditions_pairs", 0.0),
            places_norm2=raw.get("places_norm2", 0.0),
            places_count=raw.get("places_count", 0.0),
            places_pairs=raw.get("places_pairs", 0.0),
            place_sum=(
                None
                if raw.get("place_sum") is None
                else np.asarray(raw["place_sum"], dtype=np.float64)
            ),
            place_count=raw.get("place_count", 0.0),
            source=raw.get("source", ""),
            noise=raw.get("noise", []),
            spectra=raw.get("spectra", {}),
            reader=raw.get("reader", {}),
            calibration=raw.get("calibration", []),
        )


def fit_weights(
    groups: dict[str, list[list[np.ndarray]]],
    source: str,
    places: list[list[tuple[str, int]]] | None = None,
) -> Weights:
    """Base statistics from labelled public windows: the "good bases" of the ear.

    ``groups`` maps a voice to its recordings, each a list of windows. Two
    levels, measured separately (v2, Q2): ``kappa`` on pairs of windows of the
    SAME recording (how much one voice varies inside one condition), and
    ``kappa_c`` on the means of DIFFERENT recordings of one voice (how far a voice
    drifts between rooms, days and microphones). v1 measured kappa across
    recordings, which folded the drift into the window level and split a voice
    heard in two conditions into two people.

    ``places`` (v2), when given, lists the recordings heard in one same place, as
    (voice, recording index): several voices through one room and one microphone.
    Measured there: ``kappa_in``, how close different voices of one place are to
    each other (they share the channel), and ``kappa_p``, how far places are from
    each other. Without it, two people in one room were predicted as far apart as
    two people anywhere, and measured on the pyannote sample (cos 0.49 between its
    two voices where the population said 0.35), the ear heard them as one.
    """
    dim = len(next(iter(groups.values()))[0][0])
    cross, pairs, speakers, count = 0.0, 0.0, np.zeros(dim), 0
    cond_norm2, cond_count, cond_pairs = 0.0, 0, 0
    for recordings in groups.values():
        sums = [sum(r, np.zeros(dim)) for r in recordings]
        sizes = [len(r) for r in recordings]
        total = sum(sums, np.zeros(dim))
        cross += sum(float(s @ s) for s in sums) - sum(sizes)
        pairs += sum(m * (m - 1) for m in sizes)
        speakers = speakers + unit(total)
        count += sum(sizes)
        means = sum((unit(s) for s in sums), np.zeros(dim))
        cond_norm2 += float(means @ means)
        cond_count += len(sums)
        cond_pairs += len(sums) * (len(sums) - 1)
    in_norm2, in_count, in_pairs, place_sum = 0.0, 0, 0, np.zeros(dim)
    for place in places or []:
        means = [unit(sum(groups[v][i], np.zeros(dim))) for v, i in place]
        together = sum(means, np.zeros(dim))
        in_norm2 += float(together @ together)
        in_count += len(means)
        in_pairs += len(means) * (len(means) - 1)
        place_sum = place_sum + unit(together)
    return Weights(
        dim=dim,
        within_norm2=cross,
        within_count=0,  # the norms above already have the diagonal removed
        within_pairs=pairs,
        between_sum=speakers,
        between_count=len(groups),
        tables_obs=count,
        tables_k=len(groups),
        conditions_norm2=cond_norm2,
        conditions_count=cond_count,
        conditions_pairs=cond_pairs,
        places_norm2=in_norm2,
        places_count=in_count,
        places_pairs=in_pairs,
        place_sum=place_sum if places else None,
        place_count=len(places or []),
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
        self.voices: dict[str, Voice] = {}
        self.where: dict[int, str] = {}
        self.cannot: set[tuple[int, int]] = set()
        self.must: set[tuple[int, int]] = set()  # rows said to share a voice: never unlinked
        self.links: dict[str, float] = {"dot": 0.0, "count": 0.0}  # labelled condition pairs
        self.absorbed: dict[str, str] = {}
        self.pairs: dict[str, float] = {"dot": 0.0, "count": 0.0}
        self.confirm: dict[str, float] = {"gaps": 0.0, "count": 0.0, "forgotten": 0.0}
        self.cure: tuple[float, float] | None = None
        self.turns: dict[str, float] = {"stay": 0.0, "switch": 0.0}
        self.after: dict[int, int] = {}  # row -> the previous learned row of its run
        self.margin_of: dict[int, float] = {}  # row -> the margin it was decided with
        self.place: str | None = None  # where the ear is listening (the ear's session)
        self.here: set[str] = set()  # the speaker conditions heard in this place
        # [margin, was it right]: the base's public windows, then every correction
        self.calibration: list[list[float]] = [list(x) for x in weights.calibration]
        self.trust_weights: np.ndarray | None = None
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

        # kappa_c, how far one voice drifts between conditions (rooms, days,
        # microphones), learns like kappa: from the recordings of the base, then
        # from every pair of conditions a "same" correction joined.
        twice = self.links["count"] + self.links["count"]
        self.kappa_c = kappa_from_rho(
            self.dim,
            math.sqrt(
                rho2(
                    base.conditions_norm2 + twice + self.links["dot"] + self.links["dot"],
                    base.conditions_count + twice,
                    base.conditions_pairs + twice,
                )
            ),
        )
        means = [unit(self.voice_total(v)) for v in self.voices.values()]
        directions = base.between_sum + sum(means, np.zeros(base.dim))
        k = base.between_count + len(means)
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
        conditions = sum(len(v.conditions) for v in self.voices.values())
        self.alpha_v = crp_alpha(
            base.conditions_count + conditions, base.tables_k + len(self.voices)
        )
        self.log_alpha_v = math.log(self.alpha_v)
        self.log_c_kappa_c = log_c(self.dim, self.kappa_c)
        self._centre()

    def _centre(self) -> None:
        """The prior of a new voice, centred on the place the ear is in.

        Voices heard through one room and one microphone share that channel: their
        means lie around the place's mean with ``kappa_in``, and places lie around
        the population's mean with ``kappa_p`` (both measured on the base's
        simulated places). The place's mean is estimated from the voices already
        heard here; a new voice is predicted around it, with the place's own
        uncertainty composed in (A_d(k_new) = A_d(kappa_in) A_d(||a||)). Before
        any voice is heard here, this is the population prior again, broadened.
        """
        base = self.base
        if base.place_sum is None or not base.places_pairs:
            return
        kappa_in = kappa_from_rho(
            self.dim, math.sqrt(rho2(base.places_norm2, base.places_count, base.places_pairs))
        )
        spread = float(base.place_sum @ base.place_sum)
        count = base.place_count
        kappa_p = kappa_from_rho(self.dim, math.sqrt(rho2(spread, count, count * (count - 1))))
        here = [c for c in self.here if c in self.clusters and self.clusters[c].kind == "speaker"]
        a = kappa_p * self.mean0 + kappa_in * sum(
            (unit(self.clusters[c].total) for c in here), np.zeros(self.dim)
        )
        self.kappa_in, self.kappa_p = kappa_in, kappa_p
        self.kappa0 = kappa_from_rho(
            self.dim, rho_of(self.dim, kappa_in) * rho_of(self.dim, float(np.linalg.norm(a)))
        )
        self.prior = self.kappa0 * unit(a)
        self.log_c_kappa0 = log_c(self.dim, self.kappa0)

    def _lc(self, total: np.ndarray) -> float:
        """log C_d(||kappa0 mean0 + kappa total||): the posterior of a cluster mean."""
        return log_c(self.dim, float(np.linalg.norm(self.prior + self.kappa * total)))

    def predictive(self, total: np.ndarray, x: np.ndarray) -> float:
        """log p(x | the windows already in a cluster), the mean integrated out."""
        return self.log_c_kappa + self._lc(total) - self._lc(total + x)

    # -- the second level: conditions grouped into voices ----------------------------
    def voice_total(self, voice: Voice, without: str | None = None) -> np.ndarray:
        """Sum of the unit means of a voice's conditions (one excluded, if asked)."""
        return sum(
            (unit(self.clusters[c].total) for c in voice.conditions if c != without),
            np.zeros(self.dim),
        )

    def _rows_of(self, keys) -> set[int]:
        return {r for k in keys for r in self.clusters[k].members}

    def _conflicts(self, cluster: Cluster, voice: Voice) -> bool:
        rows = self._rows_of(c for c in voice.conditions if c != cluster.key)
        return any(
            (a, b) in self.cannot or (b, a) in self.cannot for a in cluster.members for b in rows
        )

    def _bound(self, cluster: Cluster) -> str | None:
        """The voice a correction tied this condition to, if any."""
        for voice in self.voices.values():
            rows = self._rows_of(c for c in voice.conditions if c != cluster.key)
            if any(
                (a, b) in self.must or (b, a) in self.must for a in cluster.members for b in rows
            ):
                return voice.key
        return None

    def voice_scores(self, cluster: Cluster) -> dict[str | None, float]:
        """Log posterior of "this condition is that voice", for every voice, and "a new one".

        Conditions of one voice spread around it with concentration ``kappa_c``;
        a condition's windows are integrated through ``kappa``. A new voice is
        drawn from the population (``kappa0``, ``mean0``).
        """
        out: dict[str | None, float] = {}
        for key, voice in self.voices.items():
            others = [c for c in voice.conditions if c != cluster.key]
            if not others or self._conflicts(cluster, voice):
                continue
            centre = self.kappa_c * unit(self.voice_total(voice, cluster.key))
            spread = float(np.linalg.norm(centre + self.kappa * cluster.total))
            out[key] = math.log(len(others)) + self.log_c_kappa_c - log_c(self.dim, spread)
        out[None] = self.log_alpha_v + self.log_c_kappa0 - self._lc(cluster.total)
        return out

    def _link(self, key: str) -> None:
        """Put a speaker condition in its most probable voice (it may move as it grows)."""
        cluster = self.clusters.get(key)
        if cluster is None or cluster.kind != "speaker":
            return
        bound = self._bound(cluster)
        if bound is None:
            scored = self.voice_scores(cluster)
            bound = max(scored, key=scored.get)
        if bound is not None and bound == cluster.voice:
            return
        old = self._detach(cluster)
        if bound is None:
            vkey = (
                cluster.key
                if cluster.key not in self.voices
                else centroid_key(np.append(cluster.total, float(len(self.voices))))
            )
            self.voices[vkey] = Voice(vkey, [], label=cluster.label, first_seen=cluster.first_seen)
            bound = vkey
        voice = self.voices[bound]
        voice.conditions.append(cluster.key)
        voice.label = voice.label or cluster.label
        cluster.voice = bound
        if old is not None and old not in self.voices:
            self.absorbed[old] = bound

    def _detach(self, cluster: Cluster) -> str | None:
        old = cluster.voice
        if old is not None and old in self.voices:
            voice = self.voices[old]
            if cluster.key in voice.conditions:
                voice.conditions.remove(cluster.key)
            if not voice.conditions:
                del self.voices[old]
        cluster.voice = None
        return old

    def voice_of(self, row: int) -> Voice | None:
        key = self.where.get(row)
        cluster = self.clusters.get(key) if key else None
        return self.voices.get(cluster.voice) if cluster and cluster.voice else None

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
        self,
        row: int,
        x: np.ndarray,
        t: float,
        learn: bool,
        follows: int | None = None,
        place: str | None = None,
    ) -> dict:
        """Decide who ``x`` is. If ``learn``, the decision also becomes evidence.

        ``follows`` is the previous learned window of the same speech run, if any:
        the continuity prior uses it once labelled transitions exist. ``place`` is
        where the ear listens (its session): voices of one place share a channel.
        """
        self.now = max(self.now, t)
        if place != self.place:
            self.place, self.here = place, set()
            self._centre()
        previous = self.where.get(follows) if follows is not None else None
        scored = self.scores(x, VOICE + ("noise",), previous)
        ranked = sorted(scored.items(), key=lambda item: item[1], reverse=True)
        top = ranked[0][1]
        self.margin_of[row] = top - (ranked[1][1] if ranked[1:] else top)
        mass = sum(math.exp(value - top) for _, value in ranked)
        choice = ranked[0][0]
        runner = ranked[1] if ranked[1:] else (None, top)
        known = set(self.voices)
        if learn:
            if follows is not None:
                self.after[row] = follows
            choice = self._add(choice, row, x, t, "candidate")
            choice = self._merge(self._promote(choice))
            if choice is not None and self.clusters[choice].kind == "speaker":
                self.here.add(choice)
            self._link(choice)
            self._forget()
            self.refresh()
        cluster = self.clusters.get(choice) if choice is not None else None
        voice = self.voices.get(cluster.voice) if cluster is not None and cluster.voice else None
        kind = self._kind(cluster, voice, known)
        return {
            "kind": kind,
            "speaker": voice.key if voice else (cluster.key if kind == "noise" else None),
            "condition": cluster.key if cluster is not None else None,
            "label": voice.label if voice else (cluster.label if cluster is not None else None),
            "source": voice.source if voice else None,
            "p": math.exp(ranked[0][1] - top) / mass,
            "runner_up": {"id": runner[0], "p": math.exp(runner[1] - top) / mass},
            "margin": top - runner[1],  # nats between the winner and the runner up
            "trust": self.trust(top - runner[1]),
            "evidence": cluster.n if cluster is not None else 0,
            "learned": learn,
        }

    @staticmethod
    def _kind(cluster: Cluster | None, voice: Voice | None, known: set[str]) -> str:
        if cluster is not None and cluster.kind == "noise":
            return "noise"
        if voice is None:
            return "unknown"
        return "known" if voice.key in known else "new"

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
            self._detach(cluster)
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
            self._link(cluster.key)
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
        voice = self.voices.get(cluster.voice) if cluster.voice else None
        if voice is not None and old in voice.conditions:
            voice.conditions[voice.conditions.index(old)] = new
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
        dropped_voice = self._detach(drop)
        del self.clusters[drop.key]
        self.absorbed[drop.key] = keep.key
        key = self._promote(keep.key)
        self._link(key)
        survivor = self.clusters[key].voice
        if dropped_voice is not None and dropped_voice not in self.voices and survivor:
            self.absorbed[dropped_voice] = survivor
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
        """Obey one correction now. ``refresh`` then absorbs it into the weights.

        ``same``/``different`` speak about voices (people), not about conditions:
        "these windows are one person" joins their voices and teaches ``kappa_c``;
        "these are two people" separates them, down to the window if needed.
        """
        known = [r for r in rows if r in self.where]
        self._calibrate(kind, rows)
        adjacent = [(a, b) for a in rows for b in rows if self.after.get(b) == a]
        if kind == "same":
            self.turns["stay"] += len(adjacent)
        elif kind == "different":
            self.turns["switch"] += len(adjacent)
        if kind == "name":
            for row in known:
                self._as_speaker(row)
                voice = self.voice_of(row)
                if voice is not None:
                    voice.label = label
                self.clusters[self.where[row]].label = label
        elif kind in ("person", "device"):
            for row in known:
                self._as_speaker(row)
                voice = self.voice_of(row)
                if voice is not None:
                    voice.source = kind
        elif kind == "same":
            self.cannot -= {(a, b) for a in rows for b in rows}
            self.must |= {(a, b) for a in known for b in known if a != b}
            for row in known:
                self._as_speaker(row)
            self._join([self.voice_of(r).key for r in known if self.voice_of(r)])
        elif kind == "different":
            self.must -= {(a, b) for a in rows for b in rows}
            self.cannot |= {(a, b) for a in rows for b in rows if a != b}
            self._split(known)
            for row in known[1:]:
                key = self.where.get(row)
                if key and self.clusters[key].kind == "speaker":
                    self._link(key)
                    cluster = self.clusters[key]
                    if self._conflicts(cluster, self.voices[cluster.voice]):
                        self._detach(cluster)
                        self._link(key)
        elif kind == "not_voice":
            for row in known:
                when = self.clusters[self.where[row]].last_seen
                self._remove(row)
                x = self.vec_of(row)
                scored = self.scores(x, ("noise",))
                self._add(max(scored, key=scored.get), row, x, when, "noise")
        self.refresh()

    # -- calibration (Q1): is a margin of so many nats often right? -------------------------
    def _calibrate(self, kind: str, rows: list[int]) -> None:
        """What a correction says about the decisions it touches, kept against their margin."""
        decided = [(r, self.voice_of(r)) for r in rows if r in self.margin_of]
        voices = [v.key if v else None for _, v in decided]
        right: list[tuple[int, bool]] = []
        if kind == "same" and voices:
            counts = {v: voices.count(v) for v in voices if v is not None}
            majority = max(counts, key=counts.get) if counts else None
            right = [
                (r, v is not None and v == majority)
                for (r, _), v in zip(decided, voices, strict=True)
            ]
        elif kind == "different":
            seen: set[str] = set()
            for (r, _), v in zip(decided, voices, strict=True):
                right.append((r, v is None or v not in seen))
                if v is not None:
                    seen.add(v)
        elif kind == "not_voice":
            right = [(r, v is None) for (r, _), v in zip(decided, voices, strict=True)]
        elif kind == "name":
            right = [(r, True) for (r, _), v in zip(decided, voices, strict=True) if v is not None]
        for row, ok in right:
            self.calibration.append([self.margin_of[row], float(ok)])
        if right:
            self.trust_weights = None

    def trust(self, margin: float) -> float | None:
        """P(this decision is right | its margin), learned from corrections; None before
        corrections have shown both a right and a wrong decision."""
        outcomes = {ok for _, ok in self.calibration}
        if {0.0, 1.0} <= outcomes and self.trust_weights is None:
            examples = [
                (np.array([[m, 1], [0, 0]], dtype=np.float64), 0 if ok else 1)
                for m, ok in self.calibration
            ]
            self.trust_weights, _ = fit_choice(examples, examples[0][0].shape[1])
        if self.trust_weights is None:
            return None
        return float(choice_probabilities(np.array([[margin, 1], [0, 0]]), self.trust_weights)[0])

    def _as_speaker(self, row: int) -> None:
        """A window a correction speaks about is a voice, whatever it was decided to be."""
        cluster = self.clusters[self.where[row]]
        if cluster.kind == "candidate":
            cluster.kind = "speaker"
            self._rename(cluster.key, centroid_key(cluster.total))
            self._link(self.where[row])
        elif cluster.kind == "speaker" and cluster.voice is None:
            self._link(cluster.key)

    def _join(self, keys: list[str]) -> None:
        """Make one voice of several: the oldest keeps its key, its name and its source.

        Every pair of conditions joined is a labelled pair for ``kappa_c``: the
        dot products of their unit means, one correction, many pairs of evidence.
        """
        keys = [k for k in dict.fromkeys(keys) if k in self.voices]
        if not keys[1:]:
            return
        voices = sorted((self.voices[k] for k in keys), key=lambda v: v.first_seen)
        keep = voices[0]
        for voice in voices[1:]:
            mine = sum((unit(self.clusters[c].total) for c in keep.conditions), np.zeros(self.dim))
            theirs = sum(
                (unit(self.clusters[c].total) for c in voice.conditions), np.zeros(self.dim)
            )
            self.links["dot"] += float(mine @ theirs)
            self.links["count"] += len(keep.conditions) * len(voice.conditions)
            for condition in voice.conditions:
                self.clusters[condition].voice = keep.key
                keep.conditions.append(condition)
            keep.label = keep.label or voice.label
            keep.source = keep.source or voice.source
            keep.first_seen = min(keep.first_seen, voice.first_seen)
            del self.voices[voice.key]
            self.absorbed[voice.key] = keep.key

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
            voice = self.voices.get(cluster.voice) if cluster.voice else None
            if voice is not None:
                voice.conditions.remove(key)
                if not voice.conditions:
                    del self.voices[voice.key]
            if cluster.kind == "speaker":
                heir.kind = "speaker"
                self._rename(heir.key, key)
                if voice is not None:
                    self.voices.setdefault(voice.key, voice)
                    voice.conditions.append(key)
                    heir.voice = voice.key
            for part in ranked:
                self._link(self._promote(part.key))

    def _forbids(self, part: Cluster, row: int) -> bool:
        return any((row, r) in self.cannot or (r, row) in self.cannot for r in part.members)

    # -- persistence --------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "clusters": [c.to_dict() for c in self.clusters.values()],
            "voices": [v.to_dict() for v in self.voices.values()],
            "cannot": sorted(self.cannot),
            "must": sorted(self.must),
            "links": self.links,
            "absorbed": self.absorbed,
            "pairs": self.pairs,
            "confirm": self.confirm,
            "turns": self.turns,
            "after": [[row, previous] for row, previous in self.after.items()],
            "margin_of": [[row, m] for row, m in self.margin_of.items()],
            "calibration": self.calibration,
            "now": self.now,
        }

    def load(self, raw: dict) -> None:
        self.clusters = {c["key"]: Cluster.from_dict(c) for c in raw.get("clusters", [])}
        self.voices = {v["key"]: Voice.from_dict(v) for v in raw.get("voices", [])}
        self.where = {row: c.key for c in self.clusters.values() for row in c.members}
        self.cannot = {tuple(pair) for pair in raw.get("cannot", [])}
        self.must = {tuple(pair) for pair in raw.get("must", [])}
        self.links = dict(raw.get("links", self.links))
        self.absorbed = dict(raw.get("absorbed", {}))
        self.pairs = dict(raw.get("pairs", self.pairs))
        self.confirm = dict(raw.get("confirm", self.confirm))
        self.turns = dict(raw.get("turns", self.turns))
        self.after = {row: previous for row, previous in raw.get("after", [])}
        self.margin_of = {row: m for row, m in raw.get("margin_of", [])}
        self.calibration = [list(x) for x in raw.get("calibration", [])]
        self.trust_weights = None
        self.now = raw.get("now", self.now)
        self.refresh()

    def summary(self) -> dict:
        return {
            "kappa": self.kappa,
            "kappa0": self.kappa0,
            "alpha": self.alpha,
            "kappa_c": self.kappa_c,
            "kappa_in": getattr(self, "kappa_in", None),
            "kappa_p": getattr(self, "kappa_p", None),
            "speakers": len(self.voices),
            "conditions": sum(c.kind == "speaker" for c in self.clusters.values()),
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
            voice = self.voices.get(cluster.voice) if cluster and cluster.voice else None
            out.append(
                {
                    "id": voice.key if voice else key,
                    "condition": key,
                    "kind": cluster.kind if cluster else "new",
                    "label": voice.label if voice else (cluster.label if cluster else None),
                    "evidence": cluster.n if cluster else 0,
                    "behind_nats": top - value,
                }
            )
        return out
