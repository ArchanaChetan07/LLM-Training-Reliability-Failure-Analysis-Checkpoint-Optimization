"""Survival estimators and failure-clustering tests."""
import numpy as np
import pytest

from llm_reliability.stats import clustering as cl, survival as sv

RNG = np.random.default_rng(0)

def test_exponential_closed_form():
    e = sv.fit_exponential([2, 3, 5], [1, 0, 1])
    assert e.rate == pytest.approx(2 / 10)


def test_weibull_mle_recovers_parameters_with_censoring():
    k, s = 0.6, 20.0
    t = s * RNG.weibull(k, 6000)
    c = RNG.exponential(40, 6000)
    obs, ev = np.minimum(t, c), t <= c
    w = sv.fit_weibull(obs, ev)
    assert w.shape == pytest.approx(k, abs=4 * w.shape_se)
    assert w.scale == pytest.approx(s, rel=0.1)
    assert w.lrt_vs_exp_p < 1e-6


def test_weibull_lrt_does_not_reject_exponential():
    t = RNG.exponential(5, 3000)
    assert sv.fit_weibull(t, np.ones_like(t)).lrt_vs_exp_p > 0.001


def test_km_without_censoring_is_ecdf():
    t = np.array([1.0, 2, 3, 4])
    tt, S, *_ = sv.kaplan_meier(t, np.ones(4, bool))
    assert np.allclose(S, [0.75, 0.5, 0.25, 0.0])


def test_gpu_scaling_recovers_linear_beta():
    n = RNG.choice([64, 128, 256, 512, 1024], 8000)
    k, s_ref = 0.7, 30.0
    scale = s_ref * (n / 256) ** (-1 / k)       # hazard ∝ n^1
    t = scale * RNG.weibull(k, len(n))
    fit = sv.fit_weibull_gpu_scaling(t, np.ones_like(t), n, n_ref=256)
    assert fit.beta == pytest.approx(1.0, abs=4 * fit.beta_se)
    assert fit.shape == pytest.approx(k, rel=0.05)


def test_poisson_ci_contains_rate():
    lo, hi = sv.poisson_rate_ci(50, 1000.0)
    assert lo < 0.05 < hi


def test_poisson_process_is_not_flagged_as_clustered():
    f = np.sort(RNG.uniform(0, 2000, 80))
    r = cl.clustering_report(f, 0, 2000, n_sim=300, seed=1)
    assert r["p_dispersion_24h"] > 0.01 and r["p_cv"] > 0.01


def test_bursty_process_is_flagged():
    centers = RNG.uniform(0, 2000, 15)
    f = np.sort(np.concatenate([c + RNG.exponential(0.5, 5) for c in centers]))
    r = cl.clustering_report(f[f < 2000], 0, 2000, n_sim=300, seed=1)
    assert r["p_cv"] < 0.01 and r["weibull_shape_gaps"] < 0.8


def test_laplace_detects_decreasing_rate():
    f = np.sort(RNG.uniform(0, 1, 300) ** 2 * 1000)   # front-loaded events
    assert cl.laplace_trend_test(f, 0, 1000)["laplace_U"] < -3
