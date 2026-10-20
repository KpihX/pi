"""Person or device: is this voice in the room, or coming out of a loudspeaker? (Q3)

The first live session registered the voice of a video as a person. For the
arbiter that is the difference between someone to answer and a film to ignore,
so every voice carries, besides who it is, where it comes from.

The evidence is the shape of the voice's long-term spectrum (``sense.spectrum``),
averaged over everything heard of that voice: a loudspeaker loses the bass, a
small cone resonates, a codec cuts the top. Two classes, each a Gaussian over
that shape with one shared diagonal spread:

* the base: public voices heard through simulated rooms (``person``), and the
  same voices played through a simulated loudspeaker into those rooms
  (``device``), ``public.fit_base``;
* then every ``person`` / ``device`` correction: the voice's own mean shape is
  added to its class, at once.

The decision is the more probable class; its probability travels with it. A
voice a correction labelled keeps its label: the model only speaks for voices
nobody has labelled.

The law: no numeric literal but 0 and 1 in this module (``tests/test_law.py``).
"""

from __future__ import annotations

import math

import numpy as np

from π_ear.sphere import diagonal_gaussian

CLASSES = ("person", "device")


class Source:
    def __init__(self, base: dict | None = None):
        self.stats = {
            name: {
                "sum": np.asarray(raw["sum"], dtype=np.float64),
                "sumsq": np.asarray(raw["sumsq"], dtype=np.float64),
                "n": raw["n"],
            }
            for name, raw in (base or {}).items()
        }

    def ready(self) -> bool:
        return all(name in self.stats and self.stats[name]["n"] for name in CLASSES)

    def add(self, name: str, shape: np.ndarray) -> None:
        shape = np.asarray(shape, dtype=np.float64)
        stats = self.stats.setdefault(
            name, {"sum": np.zeros_like(shape), "sumsq": np.zeros_like(shape), "n": 0}
        )
        stats["sum"] = stats["sum"] + shape
        stats["sumsq"] = stats["sumsq"] + shape * shape
        stats["n"] += 1

    def posterior(self, shape: np.ndarray) -> dict[str, float] | None:
        """P(person), P(device) for one voice's mean spectral shape; None before any base."""
        if not self.ready():
            return None
        means = {k: s["sum"] / s["n"] for k, s in self.stats.items()}
        total = sum(s["n"] for s in self.stats.values())
        within = sum(s["sumsq"] - s["n"] * means[k] * means[k] for k, s in self.stats.items())
        var = np.maximum(within / max(total - len(CLASSES), 1), np.finfo(float).tiny)
        logs = {
            k: math.log(self.stats[k]["n"] / total) + diagonal_gaussian(shape, means[k], var)
            for k in CLASSES
        }
        top = max(logs.values())
        mass = sum(math.exp(v - top) for v in logs.values())
        return {k: math.exp(v - top) / mass for k, v in logs.items()}

    def decide(self, shape: np.ndarray) -> tuple[str | None, float | None]:
        post = self.posterior(shape)
        if post is None:
            return None, None
        name = max(post, key=post.get)
        return name, post[name]

    def to_dict(self) -> dict:
        return {
            k: {"sum": s["sum"].tolist(), "sumsq": s["sumsq"].tolist(), "n": s["n"]}
            for k, s in self.stats.items()
        }
