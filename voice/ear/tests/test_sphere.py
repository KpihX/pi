"""The mathematics, checked against scipy rather than trusted."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import ive

from π_ear.sphere import crp_alpha, kappa_from_rho, log_bessel_i, log_c, rho2


@pytest.mark.parametrize("nu", [2.5, 95.0, 255.0])
@pytest.mark.parametrize("x", [5.0, 50.0, 95.0, 530.0, 3000.0])
def test_log_bessel_matches_scipy(nu, x):
    if ive(nu, x) == 0:
        pytest.skip("scipy underflows here; see the next test")
    reference = np.log(ive(nu, x)) + x
    assert log_bessel_i(nu, x) == pytest.approx(reference, abs=1e-3 if nu < 10 else 1e-8)


def test_log_bessel_does_not_underflow_where_scipy_does():
    assert np.isinf(np.log(ive(255.0, 0.5)))
    assert np.isfinite(log_bessel_i(255.0, 0.5))


def test_kappa_inverts_the_mean_resultant_length():
    d = 192
    for kappa in (50.0, 130.0, 530.0, 2000.0):
        rho = ive(d / 2, kappa) / ive(d / 2 - 1, kappa)
        assert kappa_from_rho(d, rho) == pytest.approx(kappa, rel=0.03)


def test_log_c_at_the_embedding_dimension():
    d, nu = 192, 95.0
    for kappa in (130.0, 530.0):
        expected = (
            nu * np.log(kappa) - (d / 2) * np.log(2 * np.pi) - (np.log(ive(nu, kappa)) + kappa)
        )
        assert log_c(d, kappa) == pytest.approx(expected, abs=1e-8)


def test_rho2_is_unbiased():
    rng = np.random.default_rng(0)
    mu = np.array([1.0, 0, 0])
    x = mu * 0.8 + rng.standard_normal((4000, 3)) * 0.2
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    true = float(np.linalg.norm(x.mean(axis=0)) ** 2)
    s = x[:1000].sum(axis=0)
    assert rho2(float(s @ s), 1000, 1000 * 999) == pytest.approx(true, abs=0.02)


@pytest.mark.parametrize("n,k", [(100, 5), (1000, 3), (50, 20)])
def test_crp_alpha_expects_the_observed_tables(n, k):
    alpha = crp_alpha(n, k)
    assert alpha * np.sum(1.0 / (alpha + np.arange(n))) == pytest.approx(k, rel=1e-6)
