"""ckpt-calc: checkpoint-interval calculator grounded in measured failure rates.

Examples
--------
  ckpt-calc --gpus 1024 --ckpt-cost 60s --restart 30m --preset opt175b --current 80m
  ckpt-calc --gpus 4096 --ckpt-cost 2m --restart 20m --gpu-hours-per-failure 50000 --shape 0.7 --simulate
  ckpt-calc --gpus 512 --ckpt-cost 30s --failures 12 --hours 240 --current 1h
"""
from __future__ import annotations

import argparse
import re
import sys

import numpy as np

from .checkpointing import policy as po

# Measured in this study (results/metrics.json).  gpu_h_per_failure = exposure / events.
PRESETS = {
    "opt175b": {"gpu_h_per_failure": 26_631, "shape": 0.85,
                "desc": "OPT-175B final run, 992 A100, all unplanned interruptions (adjudicated logbook)"},
    "opt175b-hw": {"gpu_h_per_failure": 44_385, "shape": 1.0,   # hardware gaps: k = 1.19, not distinguishable from 1 (LRT p = 0.24)
                   "desc": "OPT-175B, hardware-attributed interruptions only"},
}

_UNITS = {"s": 1 / 3600, "sec": 1 / 3600, "m": 1 / 60, "min": 1 / 60, "h": 1.0, "hr": 1.0, "d": 24.0}


def parse_duration(s: str) -> float:
    m = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*([a-z]*)\s*", s.lower())
    if not m:
        raise argparse.ArgumentTypeError(f"bad duration {s!r} (use e.g. 45s, 5m, 1.5h)")
    v, u = float(m.group(1)), m.group(2) or "h"
    if u not in _UNITS:
        raise argparse.ArgumentTypeError(f"unknown unit {u!r}")
    return v * _UNITS[u]


def fmt(h: float) -> str:
    if h < 1 / 60:
        return f"{h*3600:.0f} s"
    if h < 2:
        return f"{h*60:.1f} min"
    return f"{h:.2f} h"


def mc_waste(policy, shape, M, C, R, horizon_mtbfs=400, n=8, seed=0):
    from scipy.special import gamma
    rng = np.random.default_rng(seed)
    scale = M / gamma(1 + 1 / shape)
    T = horizon_mtbfs * M
    w = [po.simulate(po.weibull_renewal_trace(shape, scale, T, rng), T, policy, C, R,
                     phase=rng.uniform()).waste_frac for _ in range(n)]
    return float(np.mean(w)), float(np.std(w) / np.sqrt(n))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ckpt-calc", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", type=int, required=True)
    ap.add_argument("--ckpt-cost", type=parse_duration, required=True, help="blocking time per checkpoint (C)")
    ap.add_argument("--restart", type=parse_duration, default=0.5, help="downtime per failure (R), default 30m")
    src = ap.add_argument_group("failure rate (pick one)")
    src.add_argument("--preset", choices=sorted(PRESETS))
    src.add_argument("--gpu-hours-per-failure", type=float)
    src.add_argument("--mtbf", type=parse_duration, help="system MTBF at this scale")
    src.add_argument("--failures", type=int, help="observed failures ...")
    src.add_argument("--hours", type=float, help="... over this many wall-clock hours at this scale")
    ap.add_argument("--shape", type=float, help="Weibull shape of time-to-failure after restart (1 = memoryless)")
    ap.add_argument("--current", type=parse_duration, help="your current fixed interval, to compare")
    ap.add_argument("--simulate", action="store_true", help="Monte-Carlo check under a Weibull failure process")
    a = ap.parse_args(argv)

    shape = a.shape
    if a.preset:
        gph = PRESETS[a.preset]["gpu_h_per_failure"]
        shape = shape or PRESETS[a.preset]["shape"]
        M = gph / a.gpus
        src_txt = PRESETS[a.preset]["desc"]
    elif a.gpu_hours_per_failure:
        M, src_txt = a.gpu_hours_per_failure / a.gpus, f"{a.gpu_hours_per_failure:,.0f} GPU-h per failure"
    elif a.mtbf:
        M, src_txt = a.mtbf, "given MTBF"
    elif a.failures is not None and a.hours:
        M = (a.hours + 2 * 10.0) / (a.failures + 2)   # same Gamma prior as the online policy
        src_txt = f"{a.failures} failures in {a.hours:g} h (posterior mean, prior 2 events @ 10 h)"
    else:
        ap.error("give --preset, --gpu-hours-per-failure, --mtbf, or --failures/--hours")

    C, R = a.ckpt_cost, a.restart
    ty, td = po.young_interval(C, M), po.daly_interval(C, M)
    out = sys.stdout.write
    out(f"\nSystem MTBF at {a.gpus} GPUs: {fmt(M)}   ({src_txt})\n")
    out(f"Checkpoint cost C = {fmt(C)}, restart R = {fmt(R)}, C/M = {C/M:.2e}\n\n")
    rows = [("Young (1974)", ty), ("Daly (2006)", td)]
    if a.current:
        rows.append(("Your current interval", a.current))
    out(f"{'policy':24s} {'interval':>10s} {'expected waste':>15s}  (exponential model)\n")
    w_daly = po.expected_waste_exact_exponential(td, C, R, M)
    for name, tau in rows:
        out(f"{name:24s} {fmt(tau):>10s} {po.expected_waste_exact_exponential(tau, C, R, M)*100:14.2f}%\n")
    if a.current:
        dw = po.expected_waste_exact_exponential(a.current, C, R, M) - w_daly
        out(f"\nSwitching to Daly recovers {dw*100:.2f}% of wall-clock "
            f"= {dw*24*a.gpus:,.0f} GPU-hours per day of training.\n")
        ratio = a.current / td
        out(f"Your interval is {ratio:.2f}x the optimum; waste near the optimum is flat "
            f"(2x off costs ~{(2 + 0.5) / 2 * 100 - 100:.0f}% more checkpoint-related waste).\n")

    if shape and abs(shape - 1) > 1e-9:
        from scipy.special import gamma
        scale = M / gamma(1 + 1 / shape)
        pa = po.weibull_hazard_adaptive(C, shape, scale)
        out(f"\nHazard-adaptive schedule (Weibull k={shape:g}): interval by time since restart\n")
        for age in (0.1, 0.5, 1, 3, 10, 30, 100):
            if age < 5 * M:
                out(f"  {fmt(age):>10s} after restart -> checkpoint every {fmt(pa.interval(age))}\n")
        if shape > 0.75:
            out("  note: for k > ~0.75 the adaptive schedule gains < 2% over Daly (regime map, fig 4).\n")
        if a.simulate:
            out("\nMonte-Carlo (Weibull renewal failures, 8 x 400 MTBF):\n")
            pols = [po.daly(C, M), pa] + ([po.fixed(a.current, "current")] if a.current else [])
            for p in pols:
                m, se = mc_waste(p, shape, M, C, R)
                out(f"  {p.name:24s} waste {m*100:6.2f}% +/- {se*100:.2f}\n")
    out("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
