"""End-to-end analysis: data -> survival estimates -> policy evaluation -> figures.

Usage:  python scripts/run_analysis.py [--quick]
Writes results/metrics.json, results/tables/*.csv, results/figures/*.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_reliability.data import acme, opt_logbook as ol, pai  # noqa: E402
from llm_reliability.stats import clustering as cl, survival as sv  # noqa: E402
from llm_reliability.checkpointing import policy as po  # noqa: E402

RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
OUT = ROOT / "results"
FIG, TAB = OUT / "figures", OUT / "tables"
SEED = 20261117

# Checkpoint-cost grid (hours).  OPT's blocking local save measured at ~1 s in
# the logbook (03:12:26 -> 03:12:27); the grid spans async-local (10 s) to a
# synchronous write of ~2.5 TB optimizer state to shared storage (10 min).
C_GRID = {"10 s": 10 / 3600, "1 min": 1 / 60, "5 min": 5 / 60, "10 min": 10 / 60}
C_REF = "1 min"
R_GRID = [0.25, 0.5, 1.0, 2.0]
R_REF = 0.5     # logbook: ~15 min checkpoint load + requeue/fast-forward; detection excluded
OPT_FIXED_H = 250 * 19.3 / 3600   # 250 updates x 19.3 s/update = 80.4 min


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        o = o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


# =========================================================================
# 1. OPT-175B
# =========================================================================
def opt_section(n_sim):
    events = ol.build_event_table(RAW / "opt" / "OPT175B_Logbook.pdf",
                                  ROOT / "data" / "annotations" / "opt_adjudication.csv")
    events.to_csv(PROC / "opt175b_logbook_events.csv", index=False)
    win = events[events.category.notna()]
    H = (ol.RUN_END - ol.RUN_START) / pd.Timedelta(hours=1)

    def hours(ts):
        return ((ts - ol.RUN_START) / pd.Timedelta(hours=1)).to_numpy()

    series = {
        "unplanned": hours(ol.interruption_times(events, ol.UNPLANNED)),
        "hardware": hours(ol.interruption_times(events, ("hardware",))),
        "hardware+hang": hours(ol.interruption_times(events, ("hardware", "hang"))),
    }
    res = {"span_h": H, "n_gpus": ol.N_GPUS, "n_entries_total": len(events),
           "n_entries_adjudicated": len(win),
           "category_counts_entries": win.category.value_counts().to_dict(),
           "category_counts_events": win.groupby("category").n_events.sum().to_dict(),
           "codebook_vs_manual_kappa": ol.cohen_kappa(win.category.astype(str), win.codebook_category),
           "codebook_vs_manual_accuracy": float((win.category == win.codebook_category).mean()),
           "tz_assumed_frac": float(events.tz_assumed.mean()),
           "series": {}}
    for name, f in series.items():
        d = len(f)
        lo, hi = sv.poisson_rate_ci(d, H)
        gaps = cl.interarrivals(f)
        wb = sv.fit_weibull(gaps, np.ones_like(gaps))
        entry = {
            "n_events": d, "mtbf_h": H / d, "mtbf_ci95_h": [1 / hi, 1 / lo],
            "gpu_h_per_failure": H * ol.N_GPUS / d,
            "gpu_h_per_failure_ci95": [ol.N_GPUS / hi, ol.N_GPUS / lo],
            "gap_weibull": wb.to_dict(),
            "laplace": cl.laplace_trend_test(f, 0, H),
        }
        if name == "unplanned":
            entry["clustering"] = cl.clustering_report(f, 0, H, n_sim=n_sim, seed=SEED)
        res["series"][name] = entry
    return events, series, res


# =========================================================================
# 2. Acme (Seren, Kalos)
# =========================================================================
EVENT_DEFS = {"crash": ("crash",), "crash+kill": ("crash", "operator_kill", "preempted")}
WARMUPS = [0.0, 0.5]


def acme_section(n_boot):
    out, chains_all, surv_tables = {}, {}, {}
    for c in ("seren", "kalos"):
        df = acme.load(RAW / "acme" / f"trace_{c}.csv", c)
        sens = {}
        for gap in (0.5, 2.0, 6.0):
            ch = acme.build_chains(df, gap_h=gap)
            s = acme.survival_table(ch, EVENT_DEFS["crash+kill"], 0.5)
            w = sv.fit_weibull(s.t, s.event)
            sens[f"gap_{gap}h"] = {"n_chains": int(ch.chain.nunique()), "kinds": ch.kind.value_counts().to_dict(),
                                   "weibull_shape": w.shape, "weibull_shape_se": w.shape_se}
        ch = acme.build_chains(df, gap_h=2.0)
        ch.to_csv(PROC / f"acme_{c}_pretrain_chains.csv", index=False)
        chains_all[c] = ch
        res = {"n_pretrain_jobs": int(len(ch)), "n_chains": int(ch.chain.nunique()),
               "kinds": ch.kind.value_counts().to_dict(),
               "gpu_h_total": float((ch.dur_h * ch.gpu_num).sum()),
               "chain_gap_sensitivity": sens, "fits": {}}
        for ename, ev in EVENT_DEFS.items():
            for wu in WARMUPS:
                s = acme.survival_table(ch, ev, wu)
                t = s.t.clip(lower=1 / 3600).to_numpy()
                w = sv.fit_weibull(t, s.event)
                d, gh = int(s.event.sum()), float(s.gpu_h.sum())
                lo, hi = sv.poisson_rate_ci(d, gh)
                key = f"{ename}|warmup_{wu}h"
                boot = sv.bootstrap(lambda tt, ee: sv.fit_weibull(tt, ee).shape, [t, s.event.to_numpy()],
                                    n_boot=n_boot, seed=SEED)
                scal = sv.fit_weibull_gpu_scaling(t, s.event, s.gpu_num, n_ref=512)
                res["fits"][key] = {
                    "n": len(s), "events": d, "weibull": w.to_dict(), "shape_boot95": [boot[0], boot[2]],
                    "gpu_h_per_failure": gh / max(d, 1), "gpu_h_per_failure_ci95": [1 / hi, 1 / lo if lo else None],
                    "gpu_scaling": scal.to_dict(),
                }
                surv_tables[(c, key)] = s
        out[c] = res
    return chains_all, surv_tables, out


# =========================================================================
# 3. PAI
# =========================================================================
def pai_section():
    cache = ROOT / "data" / "interim" / "pai_inst_end.pkl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    j = pai.load(RAW / "pai", cache=cache)
    raw = pd.read_csv(RAW / "pai" / "pai_job_table.csv", header=None, names=pai.JOB_COLS,
                      usecols=["job_name", "status", "end_time"])
    miss = raw.groupby("status").end_time.apply(lambda s: float(s.isna().mean())).to_dict()
    recovered = None
    if (RAW / "pai" / "pai_instance_table.csv").exists():
        ie = pai.instance_end_times(RAW / "pai", cache)
        fm = raw[(raw.status == "Failed") & raw.end_time.isna()]
        recovered = float(fm.job_name.isin(ie.index).mean())
    res = {"n_jobs_with_times": int(len(j)), "end_time_missing_by_status": miss,
           "failed_missing_end_recovered_from_instances": recovered,
           "end_source_counts": j.end_source.value_counts().to_dict(), "fits": {}}
    tables = {}
    for mg in (1, 8, 32):
        for wu in WARMUPS:
            s = pai.survival_table(j, mg, wu)
            t = s.t.clip(lower=1 / 3600).to_numpy()
            w = sv.fit_weibull(t, s.event)
            d, gh = int(s.event.sum()), float(s.gpu_h.sum())
            res["fits"][f"gpus>={mg}|warmup_{wu}h"] = {
                "n": len(s), "events": d, "weibull": w.to_dict(), "gpu_h_per_failure": gh / d,
                "median_gpus": float(s.gpus.median())}
            tables[(mg, wu)] = s
    return tables, res


# =========================================================================
# 4. Policy evaluation
# =========================================================================
PRIOR_MTBF_H = 10.0      # deliberately pessimistic fleet prior for the online policy
PRIOR_EVENTS = 2.0


def policy_set(C, M, wb_shape, wb_scale, prior=PRIOR_MTBF_H):
    return [
        po.fixed(OPT_FIXED_H, "OPT actual (80 min)"),
        po.fixed(1.0, "fixed 1 h"),
        po.fixed(3.0, "fixed 3 h"),
        po.young(C, M),
        po.daly(C, M),
        po.weibull_hazard_adaptive(C, wb_shape, wb_scale),
        po.online_daly(C, prior, PRIOR_EVENTS),
    ]


def policy_opt(series, res_opt, n_traces, quick):
    f = series["unplanned"]
    H = res_opt["span_h"]
    M_in = H / len(f)
    gaps = cl.interarrivals(f)
    wb = sv.fit_weibull(gaps, np.ones_like(gaps))
    rows = []
    # (a) in-sample replay on the real trace, phase-averaged
    for cname, C in C_GRID.items():
        for R in (R_GRID if not quick else [R_REF]):
            pols = policy_set(C, M_in, wb.shape, wb.scale)
            for p in pols:
                s = po.simulate_phases(f, H, p, C, R, n_phase=32 if quick else 64)
                rows.append({"eval": "replay_in_sample", "C": cname, "C_h": C, "R_h": R, **s})
            grid = np.geomspace(max(C * 2, 1 / 60), 12, 40 if quick else 70)
            ws = [po.simulate_phases(f, H, po.fixed(g), C, R, n_phase=16)["waste_mean"] for g in grid]
            rows.append({"eval": "replay_in_sample", "C": cname, "C_h": C, "R_h": R, "policy": "Oracle fixed (hindsight)",
                         "waste_mean": float(np.min(ws)), "tau_h": float(grid[int(np.argmin(ws))])})
    # (a2) prior sensitivity of the online policy (no oracle knowledge of M)
    for cname, C in C_GRID.items():
        for prior in (3.0, 10.0, 30.0, 100.0):
            s = po.simulate_phases(f, H, po.online_daly(C, prior, PRIOR_EVENTS), C, R_REF, n_phase=16 if quick else 32)
            rows.append({"eval": "online_prior_sensitivity", "C": cname, "C_h": C, "R_h": R_REF, **s})
    # (b) temporal hold-out: estimate M on first half, evaluate on second half
    half = H / 2
    f1, f2 = f[f < half], f[f >= half] - half
    M_train = half / len(f1)
    g1 = cl.interarrivals(f1)
    wb1 = sv.fit_weibull(g1, np.ones_like(g1))
    for cname, C in C_GRID.items():
        for p in policy_set(C, M_train, wb1.shape, wb1.scale):
            s = po.simulate_phases(f2, H - half, p, C, R_REF, n_phase=32 if quick else 64)
            rows.append({"eval": "holdout_2nd_half", "C": cname, "C_h": C, "R_h": R_REF, "M_train_h": M_train, **s})
    # (c) uncertainty: synthetic traces resampled from the empirical gap distribution
    rng = np.random.default_rng(SEED)
    traces = [po.empirical_renewal_trace(gaps, H, rng) for _ in range(n_traces)]
    for cname, C in C_GRID.items():
        pols = policy_set(C, M_in, wb.shape, wb.scale)
        phases = rng.uniform(size=len(traces))          # one phase per trace, shared by all policies
        sims = [[po.simulate(tr, H, p, C, R_REF, phase=ph) for p in pols] for tr, ph in zip(traces, phases)]
        W = np.array([[r.waste_frac for r in row] for row in sims])
        D = np.array([[r.downtime_h / H for r in row] for row in sims])   # waste no interval choice can avoid
        for i, p in enumerate(pols):
            rows.append({"eval": "bootstrap_traces", "C": cname, "C_h": C, "R_h": R_REF, "policy": p.name,
                         "waste_mean": float(W[:, i].mean()), "waste_p05": float(np.percentile(W[:, i], 5)),
                         "waste_p95": float(np.percentile(W[:, i], 95)),
                         **{k: v for k, v in p.meta.items() if isinstance(v, float)}})
        # paired differences (tuned Daly vs each fixed baseline)
        names = [p.name for p in pols]
        for tuned in ("Daly", "Online Daly"):
            ti = names.index(tuned)
            for j, p in enumerate(pols[:3]):
                dlt = W[:, j] - W[:, ti]
                ctrl = W[:, j] - D[:, j]                          # lost work + checkpoint overhead
                rows.append({"eval": "bootstrap_paired_gain", "tuned": tuned, "C": cname, "C_h": C, "R_h": R_REF,
                             "policy": p.name, "gain_pp_mean": float(dlt.mean() * 100),
                             "gain_pp_ci95": [float(np.percentile(dlt, 2.5) * 100), float(np.percentile(dlt, 97.5) * 100)],
                             "frac_controllable_waste_recovered": float(dlt.mean() / ctrl.mean()),
                             "gpu_hours_saved_mean": float(dlt.mean() * H * 992)})
    return pd.DataFrame(rows), {"M_in_h": M_in, "M_train_first_half_h": M_train,
                                "gap_weibull_shape": wb.shape, "gap_weibull_scale": wb.scale}


def policy_acme(chains_all, quick):
    """Cross-cluster transfer: fit a Weibull proportional-hazards model in GPU
    count on one cluster, evaluate policies on the other cluster's long runs.

    Every model-based policy gets the *same* size model, h(t|n) = (n/512)^beta h0(t),
    so scale_n = scale_512 * (n/512)^(-beta/k) and M_n = scale_n * Gamma(1+1/k).
    'Daly (per-GPU-h)' additionally shows the common practice of assuming beta = 1."""
    from scipy.special import gamma
    rows, fits = [], {}
    for c, ch in chains_all.items():
        s = acme.survival_table(ch, EVENT_DEFS["crash+kill"], 0.0)
        s = s[s.t > 1 / 60]  # exclude sub-minute launch failures (no work to lose)
        ph = sv.fit_weibull_gpu_scaling(s.t, s.event, s.gpu_num, n_ref=512)
        fits[c] = (ph, s.gpu_h.sum() / s.event.sum())
    for train, test in (("seren", "kalos"), ("kalos", "seren")):
        ph, gph_per_fail = fits[train]
        runs = acme.chain_failure_times(chains_all[test], EVENT_DEFS["crash+kill"], min_chain_h=72)
        for cname, C in C_GRID.items():
            agg = {}
            for r in runs:
                n = r["gpu_num"]
                scale_n = ph.scale_ref * (n / ph.n_ref) ** (-ph.beta / ph.shape)
                M_n = scale_n * gamma(1 + 1 / ph.shape)
                pols = [po.fixed(OPT_FIXED_H, "fixed 80 min"), po.fixed(1.0, "fixed 1 h"), po.fixed(3.0, "fixed 3 h"),
                        po.daly(C, gph_per_fail / n), po.daly(C, M_n), po.online_daly(C, M_n, PRIOR_EVENTS),
                        po.weibull_hazard_adaptive(C, ph.shape, scale_n)]
                pols[3].name, pols[4].name = "Daly (per-GPU-h, beta=1)", "Daly (fitted size model)"
                for p in pols:
                    sres = po.simulate_phases(r["failures_h"], r["span_h"], p, C, R_REF, n_phase=8 if quick else 16)
                    acc = agg.setdefault(p.name, [0.0, 0.0])
                    acc[0] += sres["waste_mean"] * r["span_h"]
                    acc[1] += r["span_h"]
            for name, (wsum, hsum) in agg.items():
                rows.append({"train": train, "test": test, "C": cname, "policy": name,
                             "waste_mean": wsum / hsum, "n_runs": len(runs), "run_hours": hsum})
    return pd.DataFrame(rows), {c: {**f[0].to_dict(), "gpu_h_per_failure": f[1]} for c, f in fits.items()}


def regime_map(quick):
    """Gain of hazard-adaptive over Daly as a function of Weibull shape and C/M.
    Time unit = MTBF (M=1). Renewal traces from a Weibull with mean 1."""
    from scipy.special import gamma
    shapes = [0.4, 0.5, 0.6, 0.7, 0.85, 1.0, 1.2]
    cms = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    T = 150.0 if quick else 300.0
    n_tr = 6 if quick else 16
    rng = np.random.default_rng(SEED)
    rows = []
    for k in shapes:
        scale = 1 / gamma(1 + 1 / k)
        traces = [po.weibull_renewal_trace(k, scale, T, rng) for _ in range(n_tr)]
        for cm in cms:
            C, R = cm, 0.02
            pd_ = po.daly(C, 1.0)
            pa = po.weibull_hazard_adaptive(C, k, scale)
            phases = rng.uniform(size=len(traces))
            wd = np.mean([po.simulate(tr, T, pd_, C, R, phase=ph).waste_frac for tr, ph in zip(traces, phases)])
            wa = np.mean([po.simulate(tr, T, pa, C, R, phase=ph).waste_frac for tr, ph in zip(traces, phases)])
            rows.append({"shape": k, "C_over_M": cm, "waste_daly": wd, "waste_adaptive": wa,
                         "gain_pp": (wd - wa) * 100, "gain_rel": (wd - wa) / wd})
    return pd.DataFrame(rows)


# =========================================================================
# 5. Figures
# =========================================================================
def figures(events, series, res_opt, surv_tables, pai_tables, pol, pol_meta, acme_meta, regime):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from llm_reliability import plotting as P
    P.apply()
    S = P.SERIES
    H = res_opt["span_h"]

    # --- Fig 1: OPT cumulative interruptions vs Poisson expectation
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    handles = []
    for i, (name, lab) in enumerate([("unplanned", "All unplanned"), ("hardware", "Hardware only")]):
        f_ = np.sort(series[name]) / 24
        ax.step(np.r_[0, f_, H / 24], np.r_[0, np.arange(1, len(f_) + 1), len(f_)], where="post", color=S[i], lw=2)
        ax.plot([0, H / 24], [0, len(f_)], color=S[i], lw=1, ls=(0, (3, 3)))
        handles.append(Line2D([], [], color=S[i], lw=2, label=f"{lab}: {len(f_)} events, MTBF {H/len(f_):.1f} h"))
    handles.append(Line2D([], [], color=P.MUTED, lw=1, ls=(0, (3, 3)), label="Constant-rate (Poisson) expectation"))
    ax.legend(handles=handles, loc="upper left")
    ax.set_xlabel("Days since Run 12.00 (2021-11-11)")
    ax.set_ylabel("Cumulative interruptions")
    ax.set_title("OPT-175B final run: interruptions arrive at a near-constant rate")
    ax.set_xlim(0, H / 24 + 0.5)
    ax.set_ylim(0, 58)
    fig.savefig(FIG / "fig1_opt_cumulative.png"); plt.close(fig)

    # --- Fig 2: Weibull probability plot of time-since-restart (slope = shape)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    curves = [(surv_tables[("seren", "crash+kill|warmup_0.0h")], "Acme Seren, pretrain (crash + kill)", S[0]),
              (surv_tables[("kalos", "crash+kill|warmup_0.0h")], "Acme Kalos, pretrain (crash + kill)", S[1]),
              (pai_tables[(8, 0.0)], "Alibaba PAI, jobs with >= 8 GPUs (failed)", S[2])]
    for s, lab, col in curves:
        t = s.t.clip(lower=1 / 3600).to_numpy()
        tt, Sv, _, _ = sv.kaplan_meier(t, s.event.to_numpy())
        m = (Sv > 0) & (Sv < 1) & (tt > 1 / 60) & (tt < np.percentile(t, 99.5))
        ax.plot(np.log10(tt[m]), np.log(-np.log(Sv[m])), color=col, lw=2, label=lab)
    f = series["unplanned"]
    g = cl.interarrivals(f)
    tt, Sv, _, _ = sv.kaplan_meier(g, np.ones_like(g))
    m = (Sv > 0) & (Sv < 1)
    ax.plot(np.log10(tt[m]), np.log(-np.log(Sv[m])), color=S[3], lw=0, marker="o", ms=5,
            label="OPT-175B, gaps between interruptions")
    x0 = np.log10(np.array([0.5, 300]))
    ax.plot(x0, (x0 - np.log10(20)) * np.log(10) + 0, color=P.MUTED, lw=1, ls=(0, (3, 3)),
            label="Slope 1 = exponential (Young/Daly assumption)")
    ax.set_xticks(np.log10([1 / 60, 0.1, 1, 10, 100]))
    ax.set_xticklabels(["1 min", "6 min", "1 h", "10 h", "100 h"])
    ax.set_xlabel("Time since (re)start, log scale")
    ax.set_ylabel("log(-log S(t))")
    ax.set_title("Hazard decreases with time since restart (slope < 1) in job traces")
    ax.legend(loc="upper left")
    fig.savefig(FIG / "fig2_weibull_plot.png"); plt.close(fig)

    # --- Fig 3: waste vs checkpoint interval on the OPT trace
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.7))
    for ax, cname in zip(axes, ["1 min", "10 min"]):
        C = C_GRID[cname]
        grid = np.geomspace(max(2 * C, 5 / 60), 8, 40)
        w = [po.simulate_phases(f, H, po.fixed(x), C, R_REF, n_phase=96)["waste_mean"] * 100 for x in grid]
        ax.plot(grid * 60, w, color=P.MUTED, lw=1.5)
        M = H / len(f)
        marks = [("OPT actual", OPT_FIXED_H, S[1]), ("fixed 3 h", 3.0, S[3]), ("Young", po.young_interval(C, M), S[2]),
                 ("Daly", po.daly_interval(C, M), S[0])]
        for lab, x, col in marks:
            y = po.simulate_phases(f, H, po.fixed(x), C, R_REF, n_phase=96)["waste_mean"] * 100
            ax.plot([x * 60], [y], marker="o", ms=8, color=col, mec=P.SURFACE, mew=2, lw=0,
                    label=f"{lab}: {x*60:.0f} min, {y:.2f}%")
        ax.set_xscale("log")
        ax.set_xticks([10, 30, 60, 120, 240, 480], ["10", "30", "60", "120", "240", "480"])
        ax.set_xlabel("Checkpoint interval (min, log scale)")
        ax.set_title(f"Checkpoint cost C = {cname}")
        ax.set_ylim(min(w) * 0.9, min(w) * 1.9)
        ax.legend(loc="upper center", fontsize=8)
    axes[0].set_ylabel("Wall-clock wasted (%)")
    fig.suptitle("Replay of the OPT-175B interruption trace (R = 30 min): waste is flat near the optimum",
                 x=0.01, ha="left", fontsize=11, fontweight="semibold", color=P.TEXT)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_opt_waste_curve.png"); plt.close(fig)

    # --- Fig 4: regime map (gain of hazard-adaptive over Daly)
    piv = regime.pivot(index="shape", columns="C_over_M", values="gain_rel") * 100
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("div", ["#eb6834", "#f3f2ee", "#2a78d6"])
    lim = max(1.0, np.nanmax(np.abs(piv.values)))
    im = ax.imshow(piv.values, cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto", origin="lower")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:+.0f}%", ha="center", va="center", fontsize=8, color=P.TEXT)
    ax.set_xticks(range(piv.shape[1]), [f"{c:g}" for c in piv.columns])
    tag = {0.85: "  (OPT-175B)", 0.5: "  (Acme Seren)", 0.6: "  (Acme Kalos)"}
    ax.set_yticks(range(piv.shape[0]), [f"{k:g}{tag.get(k, '')}" for k in piv.index])
    ax.set_xlabel("Checkpoint cost / MTBF (C/M)")
    ax.set_ylabel("Weibull shape k of failure gaps")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04)
    cb.set_label("Waste reduction vs Daly (% of Daly waste)")
    ax.set_title("Hazard-adaptive checkpointing only pays off when k << 1")
    fig.savefig(FIG / "fig4_regime_map.png"); plt.close(fig)



def regenerate_figures():
    """Rebuild figures from cached tables (no policy re-simulation of tables)."""
    events, series, res_opt = opt_section(200)
    chains, surv_tables, _ = acme_section(5)
    pai_tables, _ = pai_section()
    regime = pd.read_csv(TAB / "regime_map.csv")
    figures(events, series, res_opt, surv_tables, pai_tables, None, None, None, regime)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    for d in (PROC, FIG, TAB):
        d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_sim = 500 if a.quick else 5000
    n_boot = 50 if a.quick else 300
    n_traces = 40 if a.quick else 300

    print("[1/6] OPT logbook"); events, series, res_opt = opt_section(n_sim)
    print("[2/6] Acme");         chains, surv_tables, res_acme = acme_section(n_boot)
    print("[3/6] PAI");          pai_tables, res_pai = pai_section()
    print("[4/6] OPT policy");   pol, pol_meta = policy_opt(series, res_opt, n_traces, a.quick)
    pol.to_csv(TAB / "opt_policy_eval.csv", index=False)
    print("[5/6] Acme policy");  pol_acme, acme_meta = policy_acme(chains, a.quick)
    pol_acme.to_csv(TAB / "acme_policy_transfer.csv", index=False)
    print("[6/6] Regime map");   regime = regime_map(a.quick)
    regime.to_csv(TAB / "regime_map.csv", index=False)

    # predicted OPT MTBF from Acme per-GPU-hour rates (assumes linear scaling)
    pred = {c: {"gpu_h_per_failure": m["gpu_h_per_failure"],
                "pred_mtbf_992gpu_h": m["gpu_h_per_failure"] / ol.N_GPUS} for c, m in acme_meta.items()}
    metrics = {"opt": res_opt, "acme": res_acme, "pai": res_pai, "opt_policy_meta": pol_meta,
               "acme_transfer_fits": acme_meta, "acme_predicts_opt": pred,
               "config": {"C_grid_h": C_GRID, "R_grid_h": R_GRID, "R_ref_h": R_REF,
                          "opt_fixed_interval_h": OPT_FIXED_H, "seed": SEED, "quick": a.quick},
               "runtime_s": time.time() - t0}
    figures(events, series, res_opt, surv_tables, pai_tables, pol, pol_meta, acme_meta, regime)
    (OUT / "metrics.json").write_text(json.dumps(jsonable(metrics), indent=2))
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
