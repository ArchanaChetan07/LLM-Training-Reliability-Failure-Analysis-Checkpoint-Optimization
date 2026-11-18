"""Do failures cluster in time?  Tests against a homogeneous Poisson process.

For a homogeneous Poisson process (the assumption behind Young/Daly):
  * inter-arrival times are exponential -> CV = 1, Weibull shape = 1
  * counts in disjoint windows have variance = mean (index of dispersion 1)
  * P(next failure within h of a failure) = 1 - exp(-h / MTBF)
Every statistic is paired with a parametric-bootstrap p-value under the
fitted Poisson null, so small samples (OPT has ~50 events) are handled honestly.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from .survival import fit_weibull


def interarrivals(times_h: np.ndarray) -> np.ndarray:
    t = np.sort(np.asarray(times_h, float))
    gaps = np.diff(t)
    return gaps[gaps > 0]


def burstiness(gaps) -> float:
    """Goh & Barabasi (2008) B in [-1, 1]; 0 for Poisson, >0 bursty."""
    m, s = np.mean(gaps), np.std(gaps, ddof=1)
    return float((s - m) / (s + m))


def dispersion_index(times_h, t0, t1, window_h: float) -> float:
    edges = np.arange(t0, t1 + window_h, window_h)
    c, _ = np.histogram(times_h, edges)
    return float(c.var(ddof=1) / c.mean()) if c.mean() > 0 else np.nan


def aftershock_ratio(times_h, horizon_h: float, mtbf_h: float) -> float:
    """Observed fraction of failures followed by another within horizon_h,
    divided by the Poisson expectation."""
    t = np.sort(np.asarray(times_h, float))
    if len(t) < 2:
        return np.nan
    nxt = np.diff(t)
    obs = (nxt <= horizon_h).mean()
    exp = 1 - np.exp(-horizon_h / mtbf_h)
    return float(obs / exp)


def clustering_report(times_h, t0: float, t1: float, n_sim: int = 5000, seed: int = 0,
                      window_h: float = 24.0, horizon_h: float = 6.0) -> dict:
    times_h = np.sort(np.asarray(times_h, float))
    n, T = len(times_h), t1 - t0
    mtbf = T / n
    gaps = interarrivals(times_h)

    def stats_of(ts):
        g = interarrivals(ts)
        return np.array([
            np.std(g, ddof=1) / np.mean(g),
            burstiness(g),
            dispersion_index(ts, t0, t1, window_h),
            aftershock_ratio(ts, horizon_h, mtbf),
            fit_weibull(g, np.ones_like(g)).shape,
        ])

    obs = stats_of(times_h)
    rng = np.random.default_rng(seed)
    sims = np.array([stats_of(np.sort(rng.uniform(t0, t1, n))) for _ in range(n_sim)])
    # one-sided p-values in the "clustered" direction (shape: smaller = clustered)
    p = [(sims[:, i] >= obs[i]).mean() for i in range(4)] + [(sims[:, 4] <= obs[4]).mean()]
    names = ["cv", "burstiness", f"dispersion_{int(window_h)}h", f"aftershock_{int(horizon_h)}h",
             "weibull_shape_gaps"]
    ks = stats.kstest(gaps / gaps.mean(), "expon")
    return {
        "n_events": n, "span_h": T, "mtbf_h": mtbf,
        **{names[i]: float(obs[i]) for i in range(5)},
        **{f"p_{names[i]}": float(p[i]) for i in range(5)},
        **{f"null95_{names[i]}": [float(x) for x in np.percentile(sims[:, i], [2.5, 97.5])] for i in range(5)},
        "ks_exp_stat": float(ks.statistic), "ks_exp_p_naive": float(ks.pvalue),
    }


def laplace_trend_test(times_h, t0: float, t1: float) -> dict:
    """Laplace test for a monotone trend in a point process on [t0, t1].
    U ~ N(0,1) under a homogeneous Poisson process; U < 0 => rate decreasing."""
    t = np.asarray(times_h, float) - t0
    n, T = len(t), t1 - t0
    U = (t.mean() - T / 2) / (T * np.sqrt(1 / (12 * n)))
    return {"laplace_U": float(U), "p_two_sided": float(2 * stats.norm.sf(abs(U)))}
