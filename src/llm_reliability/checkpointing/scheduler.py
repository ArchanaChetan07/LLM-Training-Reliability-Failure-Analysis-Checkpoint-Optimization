"""Drop-in online checkpoint scheduler for a training loop.

Replaces ``if step % ckpt_every == 0`` with a decision that adapts to the
failure rate the run actually experiences and to the measured checkpoint cost:

    sched = OnlineCheckpointScheduler(prior_mtbf_h=10.0)
    if resumed: sched.load_state_dict(ckpt["ckpt_sched"]); sched.on_restart()
    for step in ...:
        train_step()
        if sched.should_checkpoint():
            with sched.timed_save():
                save({... , "ckpt_sched": sched.state_dict()})

State persisted in the checkpoint: cumulative wall-clock exposure and number
of unplanned restarts, so the MTBF posterior survives the failures it counts.
A clean shutdown should call ``mark_clean_exit()`` before the final save so the
next start is not counted as a failure.
"""
from __future__ import annotations

import time
from contextlib import contextmanager

from . import policy as po


class OnlineCheckpointScheduler:
    def __init__(self, prior_mtbf_h: float = 10.0, prior_events: float = 2.0,
                 init_cost_s: float = 30.0, weibull_shape: float | None = None,
                 min_interval_s: float = 60.0, max_interval_s: float = 6 * 3600.0,
                 cost_ema: float = 0.3, clock=time.monotonic):
        self.prior_mtbf_h, self.prior_events = prior_mtbf_h, prior_events
        self.cost_h = init_cost_s / 3600
        self.shape = weibull_shape
        self.min_h, self.max_h = min_interval_s / 3600, max_interval_s / 3600
        self.alpha = cost_ema
        self.clock = clock
        self.exposure_h = 0.0      # persisted
        self.n_failures = 0        # persisted
        self.clean_exit = False    # persisted
        self._t_start = self._t_last_ckpt = clock()
        self._exposure_at_start = 0.0

    # ---------------- persistence
    def state_dict(self) -> dict:
        return {"exposure_h": self._exposure_now(), "n_failures": self.n_failures,
                "cost_h": self.cost_h, "clean_exit": self.clean_exit}

    def load_state_dict(self, s: dict) -> None:
        self.exposure_h = self._exposure_at_start = float(s.get("exposure_h", 0.0))
        self.n_failures = int(s.get("n_failures", 0))
        self.cost_h = float(s.get("cost_h", self.cost_h))
        self.clean_exit = bool(s.get("clean_exit", False))
        self._t_start = self._t_last_ckpt = self.clock()

    def on_restart(self) -> None:
        """Call once after resuming from a checkpoint."""
        if not self.clean_exit:
            self.n_failures += 1
        self.clean_exit = False
        self._exposure_at_start = self._exposure_now()   # keep time accrued in this process
        self._t_start = self._t_last_ckpt = self.clock()

    def mark_clean_exit(self) -> None:
        self.clean_exit = True

    # ---------------- estimates
    def _exposure_now(self) -> float:
        return self._exposure_at_start + (self.clock() - self._t_start) / 3600

    @property
    def mtbf_h(self) -> float:
        a = self.prior_events * self.prior_mtbf_h
        return (a + self._exposure_now()) / (self.prior_events + self.n_failures)

    def interval_h(self) -> float:
        M, C = self.mtbf_h, self.cost_h
        if self.shape and abs(self.shape - 1) > 1e-9:
            from scipy.special import gamma
            age = (self.clock() - self._t_start) / 3600
            tau = po.weibull_hazard_adaptive(C, self.shape, M / gamma(1 + 1 / self.shape)).interval(age)
        else:
            tau = po.daly_interval(C, M)
        return min(max(tau, self.min_h), self.max_h)

    # ---------------- decisions
    def should_checkpoint(self) -> bool:
        return (self.clock() - self._t_last_ckpt) / 3600 >= self.interval_h()

    @contextmanager
    def timed_save(self):
        t0 = self.clock()
        yield
        dt = (self.clock() - t0) / 3600
        self.cost_h = (1 - self.alpha) * self.cost_h + self.alpha * dt
        self._t_last_ckpt = self.clock()
