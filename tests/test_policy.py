"""Closed-form intervals, simulator accounting, adaptive and online policies."""
import numpy as np
import pytest

from llm_reliability.checkpointing import policy as po

RNG = np.random.default_rng(0)

def test_young_formula():
    assert po.young_interval(0.1, 20) == pytest.approx(2.0)


def test_daly_reduces_to_young_for_small_cost():
    C, M = 1e-4, 50.0
    assert po.daly_interval(C, M) == pytest.approx(po.young_interval(C, M), rel=2e-3)


def test_daly_caps_at_mtbf_when_cost_huge():
    assert po.daly_interval(10, 4) == 4


def test_daly_minimises_exact_exponential_waste():
    C, R, M = 0.2, 0.5, 10.0
    grid = np.linspace(0.3, 6, 4000)
    w = [po.expected_waste_exact_exponential(t, C, R, M) for t in grid]
    assert po.daly_interval(C, M) == pytest.approx(grid[int(np.argmin(w))], rel=0.03)


def test_sim_no_failures_overhead_only():
    r = po.simulate([], 100.0, po.fixed(1.0), C=0.1, R=1.0)
    # 100 h -> 90 full (1 h + 6 min) cycles then a partial segment counted useful
    assert r.n_failures == 0
    assert r.useful_h + r.ckpt_overhead_h == pytest.approx(100.0)
    assert r.ckpt_overhead_h == pytest.approx(0.1 * r.n_checkpoints)


def test_sim_single_failure_accounting():
    # tau=1, C=0.1: checkpoints commit at 1.1, 2.2; failure at 2.7 loses 0.5 h
    r = po.simulate([2.7], 10.0, po.fixed(1.0), C=0.1, R=0.3)
    assert r.lost_work_h == pytest.approx(0.5)
    assert r.downtime_h == pytest.approx(0.3)
    assert r.useful_h + r.lost_work_h + r.ckpt_overhead_h + r.downtime_h == pytest.approx(10.0)


def test_sim_failure_during_checkpoint_loses_interval():
    r = po.simulate([1.05], 5.0, po.fixed(1.0), C=0.1, R=0.2)
    assert r.lost_work_h == pytest.approx(1.0)


def test_sim_failure_during_recovery_extends_downtime():
    r = po.simulate([2.0, 2.1], 10.0, po.fixed(1.0), C=0.0, R=0.5)
    assert r.downtime_h == pytest.approx(0.6)
    assert r.n_failures == 2


def test_sim_matches_exact_exponential_theory():
    C, R, M, T = 0.1, 0.25, 8.0, 200000.0
    tau = po.daly_interval(C, M)
    f = po.poisson_trace(1 / M, T, RNG)
    r = po.simulate(f, T, po.fixed(tau), C, R)
    assert r.waste_frac == pytest.approx(po.expected_waste_exact_exponential(tau, C, R, M), abs=0.004)


def test_time_is_conserved_on_random_traces():
    for _ in range(20):
        f = np.sort(RNG.uniform(0, 300, RNG.integers(0, 40)))
        r = po.simulate(f, 300.0, po.fixed(RNG.uniform(0.2, 3)), C=RNG.uniform(0, 0.2), R=RNG.uniform(0, 1))
        assert r.useful_h + r.lost_work_h + r.ckpt_overhead_h + r.downtime_h == pytest.approx(300.0)


def test_adaptive_equals_daly_when_exponential():
    C, M = 0.05, 12.0
    p = po.weibull_hazard_adaptive(C, 1.0, M)
    for age in (0.1, 3, 40):
        assert p.interval(age) == pytest.approx(po.daly_interval(C, M), rel=1e-4)


def test_adaptive_checkpoints_densely_after_restart_when_k_below_1():
    p = po.weibull_hazard_adaptive(0.05, 0.5, 10.0)
    assert p.interval(0.1) < p.interval(5) < p.interval(50)


def test_online_daly_converges_to_true_mtbf():
    p = po.online_daly(0.05, prior_mtbf=1.0, prior_events=2)
    assert p.interval(0, 10000.0, 1000) == pytest.approx(po.daly_interval(0.05, 10002 / 1002), rel=1e-9)


def test_sim_counts_failure_at_time_zero():
    r = po.simulate([0.0, 5.0], 10.0, po.fixed(1.0), C=0.0, R=0.5)
    assert r.n_failures == 2
    assert r.downtime_h == pytest.approx(1.0)
