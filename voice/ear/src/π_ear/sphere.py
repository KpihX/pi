"""Mathematics on the unit sphere: the von Mises-Fisher distribution.

A speaker embedding is a direction. The vMF distribution is the Gaussian of
directions: a mean direction and a concentration ``kappa``. Everything the model
needs is three closed forms:

* ``log_c(d, kappa)``: the log normaliser, through a uniform asymptotic
  (Debye) expansion of the modified Bessel function. At the orders an embedding
  lives at (d = 192 gives nu = 95) it agrees with scipy to 1e-9, it never
  underflows, and it needs no dependency;
* ``kappa_from_rho(d, rho)``: the concentration that produces a mean resultant
  length ``rho`` (Banerjee et al., JMLR 2005);
* ``rho2``: the unbiased estimate of rho squared from sums of unit vectors.

The literals in this file are mathematics (series coefficients), not decisions,
which is why this module is not under the law test and ``model.py`` is.
"""

from __future__ import annotations

import math

import numpy as np

_TINY = 1e-12


def log_bessel_i(nu: float, x: float) -> float:
    """log I_nu(x) for large nu, Debye expansion to third order."""
    x = max(float(x), _TINY)
    z = x / nu
    root = math.sqrt(1.0 + z * z)
    eta = root + math.log(z / (1.0 + root))
    t = 1.0 / root
    t2 = t * t
    u1 = t * (3.0 - 5.0 * t2) / 24.0
    u2 = t2 * (81.0 - 462.0 * t2 + 385.0 * t2 * t2) / 1152.0
    u3 = t * t2 * (30375.0 - 369603.0 * t2 + 765765.0 * t2**2 - 425425.0 * t2**3) / 414720.0
    series = 1.0 + u1 / nu + u2 / nu**2 + u3 / nu**3
    return -0.5 * math.log(2.0 * math.pi * nu) + nu * eta - 0.5 * math.log(root) + math.log(series)


def log_c(d: int, kappa: float) -> float:
    """log of the vMF normaliser C_d(kappa) on the sphere S^(d-1)."""
    nu = d / 2.0 - 1.0
    kappa = max(float(kappa), _TINY)
    return nu * math.log(kappa) - (d / 2.0) * math.log(2.0 * math.pi) - log_bessel_i(nu, kappa)


def kappa_from_rho(d: int, rho: float) -> float:
    """Concentration whose mean resultant length is ``rho`` (Banerjee approximation)."""
    rho = min(max(float(rho), _TINY), 1.0 - _TINY)
    return rho * (d - rho * rho) / (1.0 - rho * rho)


def rho_of(d: int, kappa: float) -> float:
    """Mean resultant length A_d(kappa) = I_{d/2}(kappa) / I_{d/2-1}(kappa)."""
    nu = d / 2.0 - 1.0
    kappa = max(float(kappa), _TINY)
    return math.exp(log_bessel_i(nu + 1.0, kappa) - log_bessel_i(nu, kappa))


def rho2(sum_norm2: float, count: float, pairs: float) -> float:
    """Unbiased rho^2 from ||sum of n unit vectors||^2: E = n + n(n-1) rho^2."""
    if pairs <= 0:
        return 0.0
    return max((sum_norm2 - count) / pairs, 0.0)


def unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def crp_alpha(observations: float, tables: float) -> float:
    """The Chinese-restaurant concentration that expects ``tables`` clusters.

    Solves alpha * sum_{i<n} 1/(alpha+i) = K by bisection on log alpha. The
    bracket and the iteration count are numerical, not judgements: the answer is
    fixed by n and K alone.
    """
    n = max(int(observations), 1)
    k = min(max(float(tables), 1.0), float(n))
    steps = np.arange(n, dtype=np.float64)
    low, high = -30.0, 30.0
    for _ in range(80):
        mid = 0.5 * (low + high)
        alpha = math.exp(mid)
        expected = alpha * float(np.sum(1.0 / (alpha + steps)))
        if expected < k:
            low = mid
        else:
            high = mid
    return math.exp(0.5 * (low + high))


def cure_rate(gaps: float, confirmed: float, ages: list[float], forgotten: float):
    """(pi, lam) of a cure-rate survival model, by EM.

    A candidate is a real voice with probability ``pi``, and a real voice is
    confirmed after an exponential delay of rate ``lam``; otherwise it never is
    (a cough, a door). Inputs: the delays of the confirmed candidates, the ages
    of the ones still waiting, and how many were already forgotten (their
    posterior was an exact zero, so they count as never confirmed).

    A plain exponential cannot do this: a candidate that waits long lowers its
    own rate, so it is never forgotten. The mixture is what lets "heard once,
    never again" fade to nothing.
    """
    if not confirmed:
        return None
    ages = np.asarray(ages, dtype=np.float64)
    waiting = len(ages) + forgotten
    pi = confirmed / (confirmed + waiting)
    lam = confirmed / max(gaps + float(ages.sum()), _TINY)
    for _ in range(50):
        alive = pi * np.exp(-lam * ages)
        w = alive / np.maximum(alive + (1.0 - pi), _TINY)
        pi = (confirmed + float(w.sum())) / (confirmed + waiting)
        lam = confirmed / max(gaps + float((w * ages).sum()), _TINY)
    return min(pi, 1.0 - _TINY), lam


def fit_choice(examples, dim: int, mean=None, rounds: int = 30):
    """Bayesian conditional logit: which candidate is right, from labelled choices.

    ``examples`` is a list of (features, index): a matrix with one row per
    candidate, and the row that was right. The weights are the MAP under a
    Gaussian prior centred on ``mean`` (the base, or zero), and the prior's
    precision is not chosen: it is re-estimated from the data by evidence
    maximisation (MacKay, 1992), so the weights stay where few examples leave
    them and move where many examples push. ``rounds`` is numerical (Newton
    steps), not a judgement. Returns (weights, posterior covariance).
    """
    mean = np.zeros(dim) if mean is None else np.asarray(mean, dtype=np.float64)
    w = mean.copy()
    precision = 1.0
    eye = np.eye(dim)
    hessian = eye
    for _ in range(rounds):
        grad = -precision * (w - mean)
        hessian = precision * eye
        for features, index in examples:
            scores = features @ w
            p = np.exp(scores - scores.max())
            p /= p.sum()
            expected = p @ features
            grad += features[index] - expected
            hessian = hessian + (features.T * p) @ features - np.outer(expected, expected)
        w = w + np.linalg.pinv(hessian) @ grad  # pinv: a feature no example used stays put
        covariance = np.linalg.pinv(hessian)
        used = dim - precision * float(np.trace(covariance))
        precision = max(used, _TINY) / max(float((w - mean) @ (w - mean)), _TINY)
    return w, np.linalg.pinv(hessian)


def fit_logit(x: np.ndarray, right: np.ndarray, rounds: int = 30) -> np.ndarray:
    """``fit_choice`` for two candidates, one of them all zeros, vectorised: P(right) =
    sigmoid(x w), the same MAP under a Gaussian prior centred on zero whose precision is
    re-estimated by evidence maximisation. Thousands of examples in milliseconds (the
    language model refits after every detection)."""
    x, right = np.asarray(x, dtype=np.float64), np.asarray(right, dtype=np.float64)
    dim = x.shape[1]
    w, precision, eye = np.zeros(dim), 1.0, np.eye(dim)
    hessian = eye
    for _ in range(rounds):
        p = 1.0 / (1.0 + np.exp(-(x @ w)))
        grad = x.T @ (right - p) - precision * w
        hessian = precision * eye + (x.T * (p * (1.0 - p))) @ x
        w = w + np.linalg.pinv(hessian) @ grad
        used = dim - precision * float(np.trace(np.linalg.pinv(hessian)))
        precision = max(used, _TINY) / max(float(w @ w), _TINY)
    return w


def choice_probabilities(features: np.ndarray, w: np.ndarray) -> np.ndarray:
    scores = features @ w
    p = np.exp(scores - scores.max())
    return p / p.sum()


def diagonal_gaussian(x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
    """log N(x; mean, diag(var)), up to the constant shared by every class with this var."""
    return -0.5 * float(np.sum((x - mean) ** 2 / var))
