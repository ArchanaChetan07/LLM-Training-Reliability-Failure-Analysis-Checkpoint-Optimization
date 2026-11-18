"""Logbook parsing, codebook, Acme restart-chain reconstruction, checksums."""
import pandas as pd
import pytest

from llm_reliability.data import acme, opt_logbook as ol

@pytest.mark.parametrize("header,expected", [
    ("2021-12-07 10:48am ET: ECC error", "2021-12-07 15:48"),
    ("2021-12-09 16:00 PT: [Susan] Run 12.49", "2021-12-10 00:00"),
    ("2021-12-27 13:19 CET [Mikel]", "2021-12-27 12:19"),
    ("2021-12-20 12:12 AM PT: [Punit] Node down", "2021-12-20 08:12"),
    ("[2021-11-07] Run 11.3: ECC Failure", "2021-11-07 17:00"),
])
def test_logbook_timestamp_parsing(header, expected):
    ts, _ = ol.parse_timestamp(header)
    assert ts.strftime("%Y-%m-%d %H:%M") == expected


def test_codebook_flags_hardware():
    c = ol.code_entry("2021-12-07 10:48am ET: ECC error, requeueing 12.46", "gpu is lost")
    assert c["primary_cause"] == "hw_gpu" and c["interruption"]


def _job(i, user, g, state, sub, start, end):
    return dict(job_id=i, user=user, gpu_num=g, type="Pretrain", state=state, cluster="x",
                submit_time=pd.Timestamp(sub, tz="UTC"), start_time=pd.Timestamp(start, tz="UTC"),
                end_time=pd.Timestamp(end, tz="UTC"))


def test_chain_reconstruction():
    df = pd.DataFrame([
        _job(1, "u", 512, "FAILED", "2023-01-01 00:00", "2023-01-01 00:01", "2023-01-01 10:00"),
        _job(2, "u", 504, "CANCELLED", "2023-01-01 10:05", "2023-01-01 10:06", "2023-01-02 10:00"),  # node excluded
        _job(3, "u", 504, "COMPLETED", "2023-01-02 10:30", "2023-01-02 10:31", "2023-01-03 00:00"),
        _job(4, "u", 128, "FAILED", "2023-01-01 05:00", "2023-01-01 05:01", "2023-01-01 06:00"),     # other run
        _job(5, "u", 512, "FAILED", "2023-01-05 00:00", "2023-01-05 00:01", "2023-01-05 03:00"),     # gap too big
    ])
    df["dur_h"] = (df.end_time - df.start_time).dt.total_seconds() / 3600
    ch = acme.build_chains(df, gap_h=2.0)
    assert ch.chain.nunique() == 3
    main = ch[ch.job_id.isin([1, 2, 3])]
    assert main.chain.nunique() == 1
    assert main.sort_values("start_time").kind.tolist() == ["crash", "operator_kill", "censored"]


def test_preempted_job_is_an_unplanned_interruption():
    df = pd.DataFrame([
        _job(1, "u", 256, "PREEMPTED", "2023-01-01 00:00", "2023-01-01 00:01", "2023-01-01 05:00"),
        _job(2, "u", 256, "COMPLETED", "2023-01-01 05:10", "2023-01-01 05:11", "2023-01-02 00:00"),
    ])
    df["dur_h"] = (df.end_time - df.start_time).dt.total_seconds() / 3600
    ch = acme.build_chains(df).sort_values("start_time")
    assert ch.kind.tolist() == ["preempted", "censored"]


def test_checksum_paths_resolve_from_data_raw():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "data" / "raw"
    for line in (root / "SHA256SUMS").read_text().splitlines():
        rel = line.split(maxsplit=1)[1].lstrip("*")
        assert not rel.startswith(".."), rel
