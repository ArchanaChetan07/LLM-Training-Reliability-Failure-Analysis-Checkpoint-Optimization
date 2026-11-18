"""OnlineCheckpointScheduler: MTBF learning, persistence, cost measurement."""
import pytest

from llm_reliability.checkpointing import policy as po

class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_scheduler_learns_mtbf_and_persists_failures():
    from llm_reliability.checkpointing.scheduler import OnlineCheckpointScheduler
    clk = FakeClock()
    s = OnlineCheckpointScheduler(prior_mtbf_h=10, prior_events=2, init_cost_s=60, clock=clk)
    tau0 = s.interval_h()
    assert tau0 == pytest.approx(po.daly_interval(1 / 60, 10), rel=1e-6)
    clk.t = 3600 * 5                      # 5 h of exposure, then crash
    state = s.state_dict()
    s2 = OnlineCheckpointScheduler(prior_mtbf_h=10, prior_events=2, clock=clk)
    s2.load_state_dict(state)
    s2.on_restart()
    assert s2.n_failures == 1
    assert s2.mtbf_h == pytest.approx((20 + 5) / 3)
    assert s2.interval_h() < tau0          # more failures -> checkpoint more often


def test_scheduler_clean_exit_not_counted():
    from llm_reliability.checkpointing.scheduler import OnlineCheckpointScheduler
    clk = FakeClock()
    s = OnlineCheckpointScheduler(clock=clk)
    s.mark_clean_exit()
    st = s.state_dict()
    s.load_state_dict(st)
    s.on_restart()
    assert s.n_failures == 0


def test_scheduler_measures_checkpoint_cost():
    from llm_reliability.checkpointing.scheduler import OnlineCheckpointScheduler
    clk = FakeClock()
    s = OnlineCheckpointScheduler(init_cost_s=10, cost_ema=1.0, clock=clk)
    with s.timed_save():
        clk.t += 120
    assert s.cost_h == pytest.approx(120 / 3600)
    assert not s.should_checkpoint()
    clk.t += 3 * s.interval_h() * 3600  # interval grows slowly as exposure accrues
    assert s.should_checkpoint()


def test_scheduler_in_process_restart_keeps_exposure():
    from llm_reliability.checkpointing.scheduler import OnlineCheckpointScheduler
    clk = FakeClock()
    s = OnlineCheckpointScheduler(prior_mtbf_h=10, prior_events=2, clock=clk)
    clk.t = 3600 * 4
    s.on_restart()                       # elastic restart without reloading state
    clk.t = 3600 * 6
    assert s.state_dict()["exposure_h"] == pytest.approx(6.0)
    assert s.mtbf_h == pytest.approx((20 + 6) / 3)
