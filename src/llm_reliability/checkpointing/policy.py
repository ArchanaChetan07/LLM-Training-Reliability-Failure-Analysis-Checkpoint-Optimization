"""Checkpoint-interval policies and an exact discrete-event replay simulator.

Notation (hours): C = blocking checkpoint cost, R = restart/recovery downtime,
M = system MTBF, tau = compute time between checkpoints.

References
----------
Young (1974)  "A first order approximation to the optimum checkpoint interval", CACM.
Daly (2006)   "A higher order estimate of the optimum checkpoint interval for
              restart dumps", FGCS 22(3).
Tiwari et al. (2014) "Lazy checkpointing: exploiting temporal locality in
              failures to mitigate checkpointing overheads", DSN.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Closed forms
# ---------------------------------------------------------------------------
def young_interval(C: float, M: float) -> float:
    return float(np.sqrt(2 * C * M))


def daly_interval(C: float, M: float) -> float:
    if C >= 2 * M:
        return float(M)
    x = C / (2 * M)
    return float(np.sqrt(2 * C * M) * (1 + np.sqrt(x) / 3 + x / 9) - C)


def expected_waste_first_order(tau: float, C: float, R: float, M: float) -> float:
    """Fraction of wall-clock lost, first-order model (exponential failures):
    checkpoint overhead C/(tau+C) plus, per failure, half an interval + restart."""
    return C / (tau + C) + (0.5 * (tau + C) + R) / M


def expected_waste_exact_exponential(tau: float, C: float, R: float, M: float) -> float:
    """Exact expected waste for Poisson failures with failures possible during
    checkpointing and recovery (Daly 2006, eq. 13 rearranged):
    E[wall per segment] = M e^{R/M} (e^{(tau+C)/M} - 1); useful = tau."""
    wall = M * np.exp(R / M) * (np.exp((tau + C) / M) - 1)
    return float(1 - tau / wall)


# ---------------------------------------------------------------------------
# Policies: interval(age_h, t_h, n_failures_so_far) -> next compute interval (h)
#   age = wall-clock time since the last restart completed
# ---------------------------------------------------------------------------
@dataclass
class Policy:
    name: str
    interval: Callable[..., float]
    meta: dict = field(default_factory=dict)


def fixed(tau_h: float, name: str | None = None) -> Policy:
    return Policy(name or f"fixed {tau_h*60:.0f} min", lambda age, *_: tau_h, {"tau_h": tau_h})


def young(C: float, M: float) -> Policy:
    tau = young_interval(C, M)
    return Policy("Young", lambda age, *_: tau, {"tau_h": tau, "M": M})


def daly(C: float, M: float) -> Policy:
    tau = daly_interval(C, M)
    return Policy("Daly", lambda age, *_: tau, {"tau_h": tau, "M": M})


def online_daly(C: float, prior_mtbf: float, prior_events: float = 2.0,
                name: str = "Online Daly") -> Policy:
    """Daly interval with MTBF re-estimated from the run's own failures.

    Conjugate Gamma prior on the failure rate: equivalent to having already
    observed ``prior_events`` failures in ``prior_events * prior_mtbf`` hours.
    Posterior-mean MTBF after n failures in t hours:
        M_hat = (prior_events*prior_mtbf + t) / (prior_events + n).
    Uses only information available at decision time (no look-ahead)."""
    a = prior_events * prior_mtbf

    def interval(age, t=0.0, n=0):
        return daly_interval(C, (a + t) / (prior_events + n))

    return Policy(name, interval, {"prior_mtbf": prior_mtbf, "prior_events": prior_events})


def weibull_hazard_adaptive(C: float, shape: float, scale: float,
                            tau_min: float | None = None, tau_max: float = 24.0,
                            name: str = "Weibull-adaptive") -> Policy:
    """Daly's rule evaluated at the *instantaneous* hazard at the current age
    since the last restart: tau(a) = Daly(C, 1 / h(a + tau/2)), solved by
    fixed-point iteration so the hazard is evaluated mid-interval.  For
    shape < 1 this checkpoints densely right after a restart, when failures
    are most likely, and lazily once the system has been stable (cf. Tiwari et
    al. 2014).  For shape = 1 it reduces exactly to Daly at M = scale."""
    tau_min = tau_min if tau_min is not None else 2 * C

    def h(a):
        a = max(a, 1e-6)
        return shape / scale * (a / scale) ** (shape - 1)

    def interval(age, *_):
        tau = np.sqrt(2 * C * scale)
        for _ in range(12):
            tau_new = float(np.clip(daly_interval(C, 1.0 / h(age + tau / 2)), tau_min, tau_max))
            if abs(tau_new - tau) < 1e-6 * tau:
                return tau_new
            tau = tau_new
        return tau

    return Policy(name, interval, {"shape": shape, "scale": scale})


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------
@dataclass
class SimResult:
    policy: str
    horizon_h: float
    useful_h: float
    lost_work_h: float
    ckpt_overhead_h: float
    downtime_h: float
    n_failures: int
    n_checkpoints: int

    @property
    def waste_frac(self) -> float:
        return 1 - self.useful_h / self.horizon_h

    def to_dict(self):
        d = self.__dict__.copy()
        d["waste_frac"] = self.waste_frac
        return d


def simulate(failures_h, horizon_h: float, policy: Policy, C: float, R: float,
             phase: float = 0.0) -> SimResult:
    """Replay a failure trace (wall-clock hours from t=0) against a policy.

    * compute progresses between checkpoints; a checkpoint blocks for C;
    * a failure loses all uncommitted work (including a checkpoint in flight)
      and triggers R hours of downtime; failures during downtime restart it;
    * ``phase`` in [0,1) shifts the first checkpoint (removes grid alignment
      luck when replaying a single real trace);
    * at the horizon, in-progress work is counted as useful (final save).
    """
    f = np.sort(np.asarray(failures_h, float))
    f = f[(f >= 0) & (f < horizon_h)]
    fi, t = 0, 0.0
    useful = lost = ck_over = down = 0.0
    n_fail = n_ck = 0
    last_fail = 0.0
    first = True

    def next_fail():
        return f[fi] if fi < len(f) else np.inf

    while t < horizon_h:
        tau = policy.interval(t - last_fail, t, n_fail)
        if first and phase > 0:
            tau = tau * (1 - phase) if tau * (1 - phase) > 1e-9 else tau
        first = False
        nf = next_fail()
        seg_end = t + tau
        if seg_end >= horizon_h and nf >= horizon_h:
            useful += horizon_h - t
            t = horizon_h
            break
        if nf < seg_end:                     # failure while computing
            lost += nf - t
            t = nf
        else:
            ck_end = seg_end + C
            if nf < ck_end:                  # failure while checkpointing
                lost += tau
                ck_over += nf - seg_end
                t = nf
            elif ck_end > horizon_h:         # horizon hits during final checkpoint
                useful += tau
                ck_over += horizon_h - seg_end
                t = horizon_h
                break
            else:
                useful += tau
                ck_over += C
                n_ck += 1
                t = ck_end
                continue
        # ---- failure at time t: recover
        n_fail += 1
        fi += 1
        rec_end = t + R
        while next_fail() < min(rec_end, horizon_h):
            n_fail += 1
            rec_end = next_fail() + R
            fi += 1
        down += min(rec_end, horizon_h) - t
        last_fail = t = min(rec_end, horizon_h)
    return SimResult(policy.name, horizon_h, useful, lost, ck_over, down, n_fail, n_ck)


def simulate_phases(failures_h, horizon_h, policy, C, R, n_phase: int = 64) -> dict:
    res = [simulate(failures_h, horizon_h, policy, C, R, phase=p)
           for p in (np.arange(n_phase) + 0.5) / n_phase]
    w = np.array([r.waste_frac for r in res])
    return {"policy": policy.name, "waste_mean": float(w.mean()),
            "waste_p05": float(np.percentile(w, 5)), "waste_p95": float(np.percentile(w, 95)),
            "useful_h_mean": float(np.mean([r.useful_h for r in res])),
            "lost_work_h_mean": float(np.mean([r.lost_work_h for r in res])),
            "ckpt_overhead_h_mean": float(np.mean([r.ckpt_overhead_h for r in res])),
            "downtime_h_mean": float(np.mean([r.downtime_h for r in res])),
            **{k: v for k, v in policy.meta.items() if isinstance(v, (int, float))}}


# ---------------------------------------------------------------------------
# Synthetic failure processes
# ---------------------------------------------------------------------------
def poisson_trace(rate: float, T: float, rng) -> np.ndarray:
    n = rng.poisson(rate * T)
    return np.sort(rng.uniform(0, T, n))


def renewal_trace(sampler: Callable[[int], np.ndarray], T: float) -> np.ndarray:
    out, t = [], 0.0
    while True:
        gaps = sampler(64)
        for g in gaps:
            t += g
            if t >= T:
                return np.asarray(out)
            out.append(t)


def weibull_renewal_trace(shape: float, scale: float, T: float, rng) -> np.ndarray:
    return renewal_trace(lambda n: scale * rng.weibull(shape, n), T)


def empirical_renewal_trace(gaps: np.ndarray, T: float, rng) -> np.ndarray:
    return renewal_trace(lambda n: rng.choice(gaps, n, replace=True), T)


def optimal_fixed_interval(failure_traces, horizon_h, C, R, grid=None) -> tuple[float, np.ndarray]:
    """Oracle: best constant interval in hindsight over a set of traces."""
    grid = grid if grid is not None else np.geomspace(max(C, 1 / 120), 12, 80)
    w = np.array([np.mean([simulate(tr, horizon_h, fixed(g), C, R, phase=0.5).waste_frac
                           for tr in failure_traces]) for g in grid])
    return float(grid[np.argmin(w)]), w
