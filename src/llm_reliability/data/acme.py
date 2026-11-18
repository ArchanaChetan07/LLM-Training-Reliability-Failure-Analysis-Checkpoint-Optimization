"""Shanghai AI Lab Acme traces (Hu et al., NSDI '24): Seren and Kalos clusters.

The scheduler trace records *jobs*, not failures.  An LLM pretraining run is a
chain of jobs: when a run dies (or hangs and is killed) the team resubmits it,
usually within minutes, at the same scale (or one node smaller after
excluding a bad node: e.g. 512 -> 504 GPUs).  We reconstruct those chains:

  consecutive Pretrain jobs of the same user whose GPU counts differ by
  <= ``size_tol`` (relative) and where the next job is *submitted* within
  ``gap_h`` of the previous job's end, belong to the same run.

Every non-final job in a chain ended in an interruption.  Its terminal state
tells us which kind:
  FAILED / NODE_FAIL  -> crash
  CANCELLED           -> operator kill (hang, slowdown, or planned change)
  PREEMPTED           -> scheduler preemption (unplanned for the job)
  COMPLETED/TIMEOUT   -> planned stop (time limit, phase change)
The final job of a chain is right-censored unless it FAILED.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CRASH = {"FAILED", "NODE_FAIL"}
PREEMPT = {"PREEMPTED"}


def load(path: Path, cluster: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ("submit_time", "start_time", "end_time"):
        df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    df["cluster"] = cluster
    df = df.dropna(subset=["start_time", "end_time"])
    df = df[df.end_time > df.start_time].copy()
    df["dur_h"] = (df.end_time - df.start_time).dt.total_seconds() / 3600
    return df


def build_chains(df: pd.DataFrame, job_type: str = "Pretrain", min_gpus: int = 64,
                 gap_h: float = 2.0, size_tol: float = 0.10) -> pd.DataFrame:
    d = df[(df["type"] == job_type) & (df.gpu_num >= min_gpus)].sort_values(
        ["user", "submit_time"]).copy()
    chain_ids = np.empty(len(d), dtype=object)
    next_cid = 0
    # open chains per user: list of (last_end, gpu_num, cid)
    open_chains: dict[str, list] = {}
    for i, r in enumerate(d.itertuples(index=False)):
        cands = open_chains.setdefault(r.user, [])
        best = None
        for j, (end, g, cid) in enumerate(cands):
            gap = (r.submit_time - end).total_seconds() / 3600
            if -0.05 <= gap <= gap_h and abs(r.gpu_num - g) <= size_tol * g:
                if best is None or end > cands[best][0]:
                    best = j
        if best is None:
            cid = f"{r.cluster}-{next_cid}"
            next_cid += 1
            cands.append((r.end_time, r.gpu_num, cid))
        else:
            cid = cands[best][2]
            cands[best] = (r.end_time, r.gpu_num, cid)
        chain_ids[i] = cid
    d["chain"] = chain_ids
    d = d.sort_values(["chain", "start_time"])
    d["seg"] = d.groupby("chain").cumcount()
    d["n_seg"] = d.groupby("chain")["seg"].transform("max") + 1
    d["is_last"] = d.seg == d.n_seg - 1
    d["interrupted"] = ~d.is_last | d.state.isin(CRASH)
    d["kind"] = np.select(
        [d.state.isin(CRASH), d.interrupted & (d.state == "CANCELLED"),
         d.interrupted & d.state.isin(PREEMPT), d.interrupted],
        ["crash", "operator_kill", "preempted", "planned"], default="censored")
    return d


def survival_table(chains: pd.DataFrame, events=("crash",), min_dur_h: float = 0.0
                   ) -> pd.DataFrame:
    """Per-job (time-since-restart) survival records.  ``events`` selects which
    interruption kinds count as failures; all others are right-censored."""
    s = chains[chains.dur_h > min_dur_h].copy()
    s["event"] = s.kind.isin(events)
    s["t"] = s.dur_h - min_dur_h  # conditional on surviving the warm-up
    s["gpu_h"] = s.dur_h * s.gpu_num
    return s[["cluster", "chain", "seg", "user", "gpu_num", "state", "kind", "t", "event",
              "dur_h", "gpu_h", "start_time", "end_time"]]


def chain_failure_times(chains: pd.DataFrame, events=("crash", "operator_kill", "preempted"),
                        min_chain_h: float = 72.0) -> list[dict]:
    """Wall-clock failure series for long runs (for policy replay)."""
    out = []
    for cid, g in chains.groupby("chain"):
        g = g.sort_values("start_time")
        t0, t1 = g.start_time.iloc[0], g.end_time.iloc[-1]
        span = (t1 - t0).total_seconds() / 3600
        if span < min_chain_h:
            continue
        ev = g[g.kind.isin(events)]
        times = ((ev.end_time - t0).dt.total_seconds() / 3600).to_numpy()
        busy = g.dur_h.sum()
        out.append({"chain": cid, "cluster": g.cluster.iloc[0], "gpu_num": int(g.gpu_num.median()),
                    "span_h": span, "busy_h": busy, "n_seg": len(g), "failures_h": times})
    return out
