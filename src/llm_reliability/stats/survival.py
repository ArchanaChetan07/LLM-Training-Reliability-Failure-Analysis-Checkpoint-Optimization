"""Parametric and non-parametric survival estimators with right-censoring.

All times are in hours unless stated.  Implemented directly on numpy/scipy so
that every estimator is auditable and unit-tested against closed forms.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import optimize, stats


# ---------------------------------------------------------------------------
# Exponential
# ---------------------------------------------------------------------------
@dataclass
class ExpFit:
    rate: float          # failures per unit exposure
    mtbf: float
    n_events: int
    exposure: float
    loglik: float
    ci95_mtbf: tuple[float, float]

    def to_dict(self):
        return asdict(self)


def poisson_rate_ci(d: int, exposure: float, alpha: float = 0.05) -> tuple[float, float]:
    """Exact (Garwood) CI for a Poisson rate d/exposure."""
    lo = stats.chi2.ppf(alpha / 2, 2 * d) / 2 / exposure if d > 0 else 0.0
    hi = stats.chi2.ppf(1 - alpha / 2, 2 * d + 2) / 2 / exposure
    return lo, hi


def fit_exponential(t, event) -> ExpFit:
    t = np.asarray(t, float)
    e = np.asarray(event, bool)
    d, T = int(e.sum()), float(t.sum())
    lam = d / T
    ll = d * np.log(lam) - lam * T if d else 0.0
    lo, hi = poisson_rate_ci(d, T)
    return ExpFit(lam, 1 / lam, d, T, ll, (1 / hi, 1 / lo if lo > 0 else np.inf))


# ---------------------------------------------------------------------------
# Weibull  S(t) = exp(-(t/scale)^shape),  h(t) = shape/scale (t/scale)^(shape-1)
# ---------------------------------------------------------------------------
@dataclass
class WeibullFit:
    shape: float
    scale: float
    shape_se: float
    scale_se: float
    loglik: float
    n: int
    n_events: int
    lrt_vs_exp_stat: float
    lrt_vs_exp_p: float

    @property
    def mean(self) -> float:
        from scipy.special import gamma
        return self.scale * gamma(1 + 1 / self.shape)

    def hazard(self, t):
        t = np.maximum(np.asarray(t, float), 1e-12)
        k, s = self.shape, self.scale
        return k / s * (t / s) ** (k - 1)

    def survival(self, t):
        return np.exp(-(np.asarray(t, float) / self.scale) ** self.shape)

    def to_dict(self):
        d = asdict(self)
        d["mean"] = self.mean
        return d


def _weibull_nll(theta, t, e, w):
    logk, logs = theta
    k, s = np.exp(logk), np.exp(logs)
    z = t / s
    ll = w * (e * (np.log(k) - np.log(s) + (k - 1) * np.log(z)) - z ** k)
    return -ll.sum()


def fit_weibull(t, event, weights=None) -> WeibullFit:
    t = np.asarray(t, float)
    e = np.asarray(event, float)
    w = np.ones_like(t) if weights is None else np.asarray(weights, float)
    if np.any(t <= 0):
        raise ValueError("durations must be positive")
    x0 = np.array([0.0, np.log(np.average(t, weights=w) + 1e-9)])
    res = optimize.minimize(_weibull_nll, x0, args=(t, e, w), method="Nelder-Mead",
                            options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 20000})
    res = optimize.minimize(_weibull_nll, res.x, args=(t, e, w), method="BFGS")
    logk, logs = res.x
    H = _num_hessian(lambda th: _weibull_nll(th, t, e, w), res.x)
    try:
        cov = np.linalg.inv(H)
        se_logk, se_logs = np.sqrt(np.maximum(np.diag(cov), 0))
    except np.linalg.LinAlgError:
        se_logk = se_logs = np.nan
    k, s = float(np.exp(logk)), float(np.exp(logs))
    ll_w = -float(res.fun)
    # exponential = Weibull with k=1
    lam = (w * e).sum() / (w * t).sum()
    ll_e = float(((w * e) * np.log(lam)).sum() - lam * (w * t).sum())
    stat = max(0.0, 2 * (ll_w - ll_e))
    return WeibullFit(k, s, k * se_logk, s * se_logs, ll_w, len(t), int(e.sum()),
                      stat, float(stats.chi2.sf(stat, 1)))


def _num_hessian(f, x, h=1e-4):
    x = np.asarray(x, float)
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ei, ej = np.eye(n)[i] * h, np.eye(n)[j] * h
            H[i, j] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h * h)
    return H


# ---------------------------------------------------------------------------
# Weibull proportional-hazards in GPU count:  h(t | n) = (n / n_ref)^beta h0(t)
# beta = 1  <=> failures arrive independently per GPU (rate linear in size).
# ---------------------------------------------------------------------------
@dataclass
class WeibullScaleFit:
    shape: float
    scale_ref: float     # scale at n_ref GPUs
    beta: float
    beta_se: float
    n_ref: float
    loglik: float
    p_beta_eq_1: float

    def to_dict(self):
        return asdict(self)


def fit_weibull_gpu_scaling(t, event, n_gpu, n_ref: float = 256.0) -> WeibullScaleFit:
    t = np.asarray(t, float)
    e = np.asarray(event, float)
    x = np.log(np.asarray(n_gpu, float) / n_ref)

    def nll(th):
        logk, logs, beta = th
        k, s = np.exp(logk), np.exp(logs)
        # cumulative hazard H = (n/n_ref)^beta (t/s)^k ; log h = beta x + log k - log s + (k-1) log(t/s)
        z = t / s
        H = np.exp(beta * x) * z ** k
        logh = beta * x + np.log(k) - np.log(s) + (k - 1) * np.log(z)
        return -(e * logh - H).sum()

    x0 = np.array([0.0, np.log(t.mean()), 1.0])
    r = optimize.minimize(nll, x0, method="Nelder-Mead", options={"maxiter": 40000, "xatol": 1e-9, "fatol": 1e-9})
    r = optimize.minimize(nll, r.x, method="BFGS")
    H = _num_hessian(nll, r.x)
    se = np.sqrt(np.maximum(np.diag(np.linalg.pinv(H)), 0))
    beta, beta_se = float(r.x[2]), float(se[2])
    z = (beta - 1) / beta_se if beta_se > 0 else np.inf
    return WeibullScaleFit(float(np.exp(r.x[0])), float(np.exp(r.x[1])), beta, beta_se,
                           n_ref, -float(r.fun), float(2 * stats.norm.sf(abs(z))))


# ---------------------------------------------------------------------------
# Non-parametric
# ---------------------------------------------------------------------------
def kaplan_meier(t, event):
    """Return (times, survival, lower95, upper95) using Greenwood + log-log CI."""
    t = np.asarray(t, float)
    e = np.asarray(event, bool)
    order = np.argsort(t)
    t, e = t[order], e[order]
    uniq = np.unique(t[e])
    S, var_acc, out = 1.0, 0.0, []
    for u in uniq:
        n_risk = float((t >= u).sum())
        d = float(((t == u) & e).sum())
        S *= 1 - d / n_risk
        if n_risk > d:
            var_acc += d / (n_risk * (n_risk - d))
        if 0 < S < 1:
            se_ll = np.sqrt(var_acc) / abs(np.log(S))
            lo = S ** np.exp(1.96 * se_ll)
            hi = S ** np.exp(-1.96 * se_ll)
        else:
            lo = hi = S
        out.append((u, S, lo, hi))
    a = np.array(out) if out else np.zeros((0, 4))
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]


def nelson_aalen(t, event):
    t = np.asarray(t, float)
    e = np.asarray(event, bool)
    uniq = np.unique(t[e])
    H, out = 0.0, []
    for u in uniq:
        H += ((t == u) & e).sum() / (t >= u).sum()
        out.append((u, H))
    a = np.array(out) if out else np.zeros((0, 2))
    return a[:, 0], a[:, 1]


def bootstrap(fn, arrays, n_boot: int = 1000, seed: int = 0):
    """Non-parametric bootstrap over rows of the given parallel arrays."""
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            vals.append(fn(*[np.asarray(a)[idx] for a in arrays]))
        except Exception:  # degenerate resample
            continue
    v = np.asarray(vals, float)
    return np.nanpercentile(v, [2.5, 50, 97.5], axis=0)
