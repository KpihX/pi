"""Shared fixtures. Two worlds:

* the unit tests need nothing: the model is tested on vMF samples drawn here,
  with the exact geometry of a 192-d speaker embedding;
* the real tests need the models and the public corpora (``make fetch``,
  ~470 MB, once). Without them they are skipped, and they say why.

No test ever writes to the operator's home: every place is under ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from π_ear import contract
from π_ear.model import Model, fit_weights

DIM = 192


def _have(*paths: Path) -> bool:
    return all(p.exists() for p in paths)


MODELS_READY = _have(contract.MODELS / contract.VAD_MODEL, contract.MODELS / contract.ENCODER)
DATA_READY = _have(contract.DATA / "esc50", contract.DATA / "fsdd", contract.DATA / "libri")
needs_models = pytest.mark.skipif(not MODELS_READY, reason="models absent: run `make fetch`")
needs_data = pytest.mark.skipif(
    not (MODELS_READY and DATA_READY), reason="models or public corpora absent: run `make fetch`"
)


def vmf(mu: np.ndarray, kappa: float, n: int, rng) -> np.ndarray:
    """n draws of vMF(mu, kappa) on the sphere, Wood's (1994) algorithm."""
    d = len(mu)
    b = (-2 * kappa + np.sqrt(4 * kappa**2 + (d - 1) ** 2)) / (d - 1)
    x0 = (1 - b) / (1 + b)
    c = kappa * x0 + (d - 1) * np.log(1 - x0**2)
    out = []
    while len(out) < n:
        z = rng.beta((d - 1) / 2, (d - 1) / 2)
        w = (1 - (1 + b) * z) / (1 - (1 - b) * z)
        if kappa * w + (d - 1) * np.log(1 - x0 * w) - c >= np.log(rng.uniform()):
            v = rng.standard_normal(d)
            v -= (v @ mu) * mu
            v /= np.linalg.norm(v)
            out.append(w * mu + np.sqrt(1 - w**2) * v)
    return np.array(out)


class Store(list):
    """An embedding store in RAM, called like the real one."""

    def __call__(self, row: int) -> np.ndarray:
        return self[row]


class World:
    """Voices with the geometry measured on CAM++: kappa0 ~ 130 between, kappa ~ 530 within."""

    def __init__(self, seed: int = 11, kappa0: float = 130.0, kappa: float = 530.0):
        self.rng = np.random.default_rng(seed)
        self.mean0 = self.rng.standard_normal(DIM)
        self.mean0 /= np.linalg.norm(self.mean0)
        self.kappa0, self.kappa = kappa0, kappa

    def voices(self, k: int) -> np.ndarray:
        return vmf(self.mean0, self.kappa0, k, self.rng)

    def windows(self, mu: np.ndarray, n: int) -> np.ndarray:
        return vmf(mu, self.kappa, n, self.rng)

    def model(self) -> tuple[Model, Store]:
        """A fresh model whose base was measured on five other voices of this world."""
        groups = {
            i: [list(self.windows(mu, 5)) for _ in range(4)] for i, mu in enumerate(self.voices(5))
        }
        store = Store()
        return Model(fit_weights(groups, "synthetic"), store), store

    def feed(self, model: Model, store: Store, x: np.ndarray, learn: bool = True) -> dict:
        store.append(x)
        return model.observe(len(store) - 1, x, float(len(store)), learn)


@pytest.fixture
def world() -> World:
    return World()
