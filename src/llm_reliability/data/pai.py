"""Alibaba PAI GPU trace (Weng et al., NSDI '22; cluster-trace-gpu-v2020).

Mixed ML workload (2020, mostly small jobs).  Used as the out-of-domain
contrast: does the decreasing-hazard shape found in LLM pretraining hold for a
general-purpose GPU cluster, and how does the per-GPU-hour failure rate compare?
Status 'Failed' = event; 'Terminated' = completed (censored); Running/Waiting
dropped (no end time).  GPU count = sum over tasks of plan_gpu/100 * inst_num.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

JOB_COLS = ["job_name", "inst_id", "user", "status", "start_time", "end_time"]
TASK_COLS = ["job_name", "task_name", "inst_num", "status", "start_time", "end_time",
             "plan_cpu", "plan_mem", "plan_gpu", "gpu_type"]


INST_COLS = ["job_name", "task_name", "inst_name", "worker_name", "inst_id", "status",
             "start_time", "end_time", "machine"]


def instance_end_times(raw_dir: Path, cache: Path | None = None) -> pd.Series:
    """Latest instance end_time per job (fills job-table gaps for failed jobs)."""
    if cache is not None and cache.exists():
        return pd.read_pickle(cache)
    parts = []
    for ch in pd.read_csv(raw_dir / "pai_instance_table.csv", header=None, names=INST_COLS,
                          usecols=["job_name", "end_time"], chunksize=5_000_000):
        parts.append(ch.dropna().groupby("job_name").end_time.max())
    s = pd.concat(parts).groupby(level=0).max()
    if cache is not None:
        s.to_pickle(cache)
    return s


def load(raw_dir: Path, use_instances: bool = True, cache: Path | None = None) -> pd.DataFrame:
    j = pd.read_csv(raw_dir / "pai_job_table.csv", header=None, names=JOB_COLS)
    j["end_source"] = np.where(j.end_time.notna(), "job", "missing")
    if use_instances and (raw_dir / "pai_instance_table.csv").exists():
        ie = instance_end_times(raw_dir, cache)
        fill = j.end_time.isna() & j.job_name.isin(ie.index)
        j.loc[fill, "end_time"] = j.loc[fill, "job_name"].map(ie)
        j.loc[fill, "end_source"] = "instance"
    t = pd.read_csv(raw_dir / "pai_task_table.csv", header=None, names=TASK_COLS,
                    usecols=["job_name", "inst_num", "plan_gpu", "gpu_type"])
    t["gpus"] = t.plan_gpu.fillna(0) / 100 * t.inst_num.fillna(1)
    g = t.groupby("job_name").agg(gpus=("gpus", "sum"),
                                  gpu_type=("gpu_type", lambda s: s.mode().iat[0] if s.notna().any() else None))
    j = j.merge(g, left_on="job_name", right_index=True, how="left")
    j = j[j.status.isin(["Failed", "Terminated"])].dropna(subset=["start_time", "end_time"])
    j = j[j.end_time > j.start_time].copy()
    j["dur_h"] = (j.end_time - j.start_time) / 3600
    j["event"] = j.status.eq("Failed")
    j["gpu_h"] = j.dur_h * j.gpus
    return j


def survival_table(j: pd.DataFrame, min_gpus: float = 8, min_dur_h: float = 0.0) -> pd.DataFrame:
    s = j[(j.gpus >= min_gpus) & (j.dur_h > min_dur_h)].copy()
    s["t"] = s.dur_h - min_dur_h
    return s
