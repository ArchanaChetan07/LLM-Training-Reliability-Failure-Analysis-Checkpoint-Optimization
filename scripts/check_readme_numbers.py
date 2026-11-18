"""Cross-check that every quantitative claim in README.md matches results/.

Run after `make all`:  python scripts/check_readme_numbers.py   (exit code 1 on any mismatch)
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
m = json.loads((ROOT / "results" / "metrics.json").read_text())
README = (ROOT / "README.md").read_text(encoding="utf-8")
README_FLAT = README.replace("**", "")
pol = pd.read_csv(ROOT / "results" / "tables" / "opt_policy_eval.csv")
acm = pd.read_csv(ROOT / "results" / "tables" / "acme_policy_transfer.csv")
checks = []


def minus(s: str) -> str:
    return s.replace("[-", "[−").replace(", -", ", −").replace("(-", "(−")


def chk(name, text):
    checks.append((name, minus(text), minus(text) in README_FLAT))


o, s = m["opt"], m["opt"]["series"]
u, hw, hh = s["unplanned"], s["hardware"], s["hardware+hang"]
chk("n unplanned", f"{u['n_events']} unplanned interruptions")
chk("MTBF", f"MTBF {u['mtbf_h']:.1f} h (95% CI {u['mtbf_ci95_h'][0]:.1f}–{u['mtbf_ci95_h'][1]:.1f})")
chk("GPU-h", f"{u['gpu_h_per_failure']:,.0f}")
chk("hw MTBF", f"(MTBF {hw['mtbf_h']:.1f} h)")
chk("hw row", f"| {hw['mtbf_h']:.1f} [{hw['mtbf_ci95_h'][0]:.1f}, {hw['mtbf_ci95_h'][1]:.1f}] | {hw['gpu_h_per_failure']:,.0f} | "
              f"{hw['gap_weibull']['shape']:.2f} | {hw['gap_weibull']['lrt_vs_exp_p']:.2f} | {hw['laplace']['p_two_sided']:.2f} |")
chk("hw+hang row", f"| {hh['mtbf_h']:.1f} [{hh['mtbf_ci95_h'][0]:.1f}, {hh['mtbf_ci95_h'][1]:.1f}] | {hh['gpu_h_per_failure']:,.0f} | "
                   f"{hh['gap_weibull']['shape']:.2f} | {hh['gap_weibull']['lrt_vs_exp_p']:.2f} | {hh['laplace']['p_two_sided']:.2f} |")
w = u["gap_weibull"]
chk("unplanned row", f"| {u['mtbf_h']:.1f} [{u['mtbf_ci95_h'][0]:.1f}, {u['mtbf_ci95_h'][1]:.1f}] | {u['gpu_h_per_failure']:,.0f} | "
                     f"{w['shape']:.2f} ± {w['shape_se']:.2f} | {w['lrt_vs_exp_p']:.2f} | {u['laplace']['p_two_sided']:.2f} |")
c = u["clustering"]
chk("CV", f"CV {c['cv']:.2f} (*p* = {c['p_cv']:.2f})")
chk("dispersion", f"index {c['dispersion_24h']:.2f} (*p* = {c['p_dispersion_24h']:.2f})")
chk("aftershock", f"ratio {c['aftershock_6h']:.2f} (*p* = {c['p_aftershock_6h']:.2f})")
chk("shape p", f"(one-sided *p* = {c['p_weibull_shape_gaps']:.3f})")
chk("kappa", f"κ = {o['codebook_vs_manual_kappa']:.2f} ({o['codebook_vs_manual_accuracy']*100:.0f}% raw agreement)")
chk("tz", f"{o['tz_assumed_frac']*100:.0f}% carry no zone")
ce = o["category_counts_events"]
chk("categories", f"hardware {ce['hardware']} · hang {ce['hang']} · infra/software {ce['infra_software']} · unknown "
                  f"{ce['unknown']} (= 50 unplanned) · optimization rollbacks {ce['optimization']} · planned {ce['planned']}")
chk("entries", f"{o['n_entries_total']} entries; {o['n_entries_adjudicated']} in the final-run window")

for cl, lab in (("seren", "Seren"), ("kalos", "Kalos")):
    a = m["acme"][cl]
    for key, ev, wu in (("crash+kill|warmup_0.0h", "crash + kill", "none"),
                        ("crash+kill|warmup_0.5h", "crash + kill", "30 min"),
                        ("crash|warmup_0.5h", "crash only", "30 min")):
        f = a["fits"][key]
        k, b, g = f["weibull"]["shape"], f["shape_boot95"], f["gpu_scaling"]
        chk(f"{lab} {key}", f"| {lab} | {ev} | {wu} | {f['n']:,} | {k:.2f} [{b[0]:.2f}, {b[1]:.2f}] | "
                            f"{f['gpu_h_per_failure']:,.0f} | {g['beta']:.2f} ± {g['beta_se']:.2f} |")
sj, kj = m["acme"]["seren"], m["acme"]["kalos"]
chk("acme sizes", f"{sj['n_pretrain_jobs']:,} / {kj['n_pretrain_jobs']:,} Pretrain jobs ≥ 64 GPUs → {sj['n_chains']} / {kj['n_chains']} restart chains")
kinds = sj["kinds"]
missed = 1 - kinds["crash"] / (kinds["crash"] + kinds["operator_kill"] + kinds.get("preempted", 0))
chk("missed by FAILED-only", f"would miss {missed*100:.0f}%")
gs = sj["chain_gap_sensitivity"], kj["chain_gap_sensitivity"]
spread = max(max(v["weibull_shape"] for v in g.values()) - min(v["weibull_shape"] for v in g.values()) for g in gs)
checks.append(("chain-gap spread < 0.015", f"{spread:.4f}", spread < 0.015))

p = m["pai"]
chk("PAI k", f"*k* = {p['fits']['gpus>=8|warmup_0.5h']['weibull']['shape']:.2f} after warm-up, "
             f"{p['fits']['gpus>=8|warmup_0.0h']['weibull']['shape']:.2f} including start-up")
chk("PAI recovered", f"recovers only {p['failed_missing_end_recovered_from_instances']*100:.1f}%")

g = pol[pol["eval"] == "bootstrap_paired_gain"].copy()
g["ci"] = g.gain_pp_ci95.apply(json.loads)
row_of = {"10 s": None, "1 min": None, "5 min": None, "10 min": None}
for C in row_of:
    cells = []
    for tuned, base in (("Daly", "OPT actual (80 min)"), ("Daly", "fixed 1 h"), ("Daly", "fixed 3 h"),
                        ("Online Daly", "OPT actual (80 min)")):
        r = g[(g.tuned == tuned) & (g.C == C) & (g.policy == base)].iloc[0]
        cells.append(f"+{r.gain_pp_mean:.2f} [{r.ci[0]:.2f}, {r.ci[1]:.2f}]")
    chk(f"R3 row {C}", f"| {C} | " + " | ".join(cells) + " |")
for C in ("10 s", "1 min"):
    r = g[(g.tuned == "Daly") & (g.C == C) & (g.policy == "fixed 3 h")].iloc[0]
    checks.append((f"controllable frac 3h {C}", f"{r.frac_controllable_waste_recovered:.3f}", True))
h = pol[pol["eval"] == "holdout_2nd_half"].set_index(["C", "policy"]).waste_mean
chk("holdout", f"Daly {h[('10 min', 'Daly')]*100:.1f}% waste vs OPT actual {h[('10 min', 'OPT actual (80 min)')]*100:.1f}%")
ps = pol[pol["eval"] == "online_prior_sensitivity"].groupby("C").waste_mean.agg(lambda x: (x.max() - x.min()) * 100)
checks.append(("prior sensitivity ≤ 0.6 pp", f"{ps.max():.2f}", ps.max() <= 0.6 and "≤ 0.6 pp" in README))

a = acm.pivot_table(index=["test", "C"], columns="policy", values="waste_mean") * 100
cols = ["fixed 80 min", "fixed 1 h", "fixed 3 h", "Daly (fitted size model)", "Online Daly", "Weibull-adaptive"]
for (tc, C) in (("kalos", "1 min"), ("kalos", "10 min"), ("seren", "1 min"), ("seren", "10 min")):
    chk(f"R4 {tc} {C}", f"| {tc.capitalize()}, {C} | " + " | ".join(f"{a.loc[(tc, C), c]:.2f}" for c in cols) + " |")
best = a[["fixed 80 min", "fixed 1 h", "fixed 3 h"]].min(axis=1)
d = a["Weibull-adaptive"] - best
checks.append(("adaptive beats best fixed 5/8", int((d < 0).sum()), int((d < 0).sum()) == 5))
checks.append(("adaptive ties ≤0.02 in 2", int(((d >= 0) & (d <= 0.02)).sum()), int(((d >= 0) & (d <= 0.02)).sum()) == 2))
checks.append(("adaptive worst loss 0.22", f"{d.max():.2f}", f"{d.max():.2f}" == "0.22"))
for col in ("Daly (fitted size model)", "Daly (per-GPU-h, beta=1)"):
    n = int((a[col] > best).sum())
    checks.append((f"{col} loses 6/8", n, n == 6))
rel = (1 - a["Weibull-adaptive"] / a["Online Daly"]) * 100
checks.append(("adaptive vs online Daly 3–12%", f"{rel.min():.1f}–{rel.max():.1f}", round(rel.min()) == 3 and round(rel.max()) == 12))
r80 = ((1 - a["Weibull-adaptive"] / a["fixed 80 min"]) * 100).max()
checks.append(("adaptive vs fixed 80 up to 31%", f"{r80:.1f}", round(r80) == 31))
tf = m["acme_transfer_fits"]
chk("PH fits", f"(Seren: *k* = {tf['seren']['shape']:.2f}, β = {tf['seren']['beta']:.2f} ± {tf['seren']['beta_se']:.2f};\n"
               f"Kalos: *k* = {tf['kalos']['shape']:.2f}, β = {tf['kalos']['beta']:.2f} ± {tf['kalos']['beta_se']:.2f})")

bad = [c for c in checks if not c[2]]
for name, val, ok in checks:
    print(("ok   " if ok else "FAIL ") + name + ("" if ok else f"  -> {val}"))
print(f"\n{len(checks) - len(bad)}/{len(checks)} checks pass")
sys.exit(1 if bad else 0)
