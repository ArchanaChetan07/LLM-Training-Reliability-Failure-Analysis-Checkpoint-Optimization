"""Turn Meta's OPT-175B logbook (PDF) into a coded event table.

Method
------
1. Extract text page-by-page with PyMuPDF.
2. Use the document's own Table of Contents (it lists every dated entry
   header) as the authoritative entry index.  This avoids confusing the
   thousands of timestamped *log lines* pasted into the body with entries.
3. Locate every TOC header in the body (in order) and slice the body text
   between consecutive headers -> one record per logbook entry.
4. Parse the timestamp + timezone (ET/PT/CET; unlabeled -> ET, the team's
   default; documented as an assumption) and convert to UTC.
5. Code each entry with a transparent, regex-based codebook (CODEBOOK below),
   applied to the header plus the first ``BODY_CHARS`` characters of the body.
6. Apply hand-adjudicated overrides from ``data/annotations/opt_overrides.csv``
   (entry_id -> corrected labels, with a free-text justification).

The coded table is a *lower bound* on interruptions: the paper reports that
many automatic restarts were handled by the monitor script without a logbook
entry.  Downstream analyses therefore include rate-inflation sensitivity.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

BODY_CHARS = 700

TZ_OFFSETS = {"ET": "-05:00", "EST": "-05:00", "PT": "-08:00", "PST": "-08:00",
              "CET": "+01:00", "UTC": "+00:00"}

_DATE_RE = re.compile(
    r"^\[?(?P<date>20(?:21|22)-\d{2}-\d{2})\]?[,:\-\s]*"
    r"(?:(?P<time>\d{1,2}(?::\d{2}){0,2})\s*(?P<ampm>am|pm|AM|PM)?)?\s*"
    r"(?P<tz>ET|EST|PT|PST|CET|UTC)?",
)

# ---------------------------------------------------------------------------
# Codebook.  Order matters only for documentation; an entry may carry several
# cause flags.  ``primary_cause`` is resolved by CAUSE_PRIORITY.
# ---------------------------------------------------------------------------
CODEBOOK: dict[str, str] = {
    # hardware / node-level faults (GPU, memory, interconnect, host)
    "hw_gpu": r"lost (?:a )?gpu|gpu is lost|gpu (?:failure|error|fell)|xid|"
              r"ecc|uncorrectable|cuda error|cublas|lost gpu",
    "hw_node": r"node(?:s)? down|machines? down|lost (?:a )?node|bad node|"
               r"node-\d+ (?:is )?(?:dead|down|unresponsive)|unresponsive|"
               r"provisioning error|port error|hardware",
    "hw_network": r"\bib issue|infiniband|nccl (?:error|timeout|failure)s?|"
                  r"nccl errors|bandwidth .* below threshold",
    "hang": r"\bhang(?:s|ing|ed|e|es)?\b|\bhung\b|stuck|file[- ]not[- ]modified",
    # human/optimization interventions (not hardware)
    "optim": r"loss scale|lower(?:ing)? (?:the )?lr|lr down|diverg|grad(?:ient)? norm|"
             r"spik|\bclip\b|sgd|\badam\b|weight decay|ppl trending|"
             r"skip(?:ping)? batches|ablation",
    # cluster operations / tooling
    "ops": r"maintenance|auto-?recovery|autorecover|monitor\.py|time ?limit|"
           r"blob upload|omicron|onboarding|tensorboard",
}
CAUSE_PRIORITY = ["hw_gpu", "hw_network", "hw_node", "hang", "optim", "ops"]
HARDWARE_CAUSES = {"hw_gpu", "hw_network", "hw_node", "hang"}

INTERRUPTION_RE = re.compile(
    r"requeu|restart|relaunch|resum|launch(?:ing)? (?:of )?(?:run )?1[12]\.|"
    r"\brun 1[12]\.\d+|scancel|resubmit|recover|kick off|job (?:was )?put on hold",
    re.I,
)

_COMPILED = {k: re.compile(v, re.I) for k, v in CODEBOOK.items()}


@dataclass
class Entry:
    entry_id: int
    header: str
    ts_utc: pd.Timestamp | None
    tz_assumed: bool
    body: str


def extract_text(pdf_path: Path) -> list[str]:
    import fitz  # PyMuPDF

    with fitz.open(pdf_path) as doc:
        return [p.get_text() for p in doc]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def parse_timestamp(header: str) -> tuple[pd.Timestamp | None, bool]:
    m = _DATE_RE.match(header.strip())
    if not m:
        return None, False
    date, time, ampm, tz = m.group("date", "time", "ampm", "tz")
    hh, mm = 12, 0  # undated-time entries -> local noon
    if time:
        parts = [int(x) for x in time.split(":")]
        hh = parts[0]
        mm = parts[1] if len(parts) > 1 else 0
        if ampm:
            ampm = ampm.lower()
            if ampm == "pm" and hh < 12:
                hh += 12
            if ampm == "am" and hh == 12:
                hh = 0
        if hh > 23:
            hh, mm = 12, 0
    assumed = tz is None
    off = TZ_OFFSETS[tz or "ET"]
    ts = pd.Timestamp(f"{date}T{hh:02d}:{mm:02d}:00{off}").tz_convert("UTC")
    return ts, assumed


def toc_headers(lines: list[str]) -> list[str]:
    """Dated headers from the (last) Table of Contents, in document order."""
    idx = min(i for i, l in enumerate(lines) if l.strip() == "Table of Contents/Index")
    out, started = [], False
    for raw in lines[idx + 1:]:
        l = raw.strip()
        if l.startswith("<<<PAGE"):
            continue
        if l == "175B Log":
            started = True
            continue
        if not started:
            continue
        if l.startswith("Kitchen sink"):
            break
        if _DATE_RE.match(l) and _DATE_RE.match(l).group("date"):
            out.append(l)
        elif out and not _DATE_RE.match(l) and len(l) < 40 and l.islower():
            out[-1] += " " + l  # wrapped header continuation (e.g. "calling")
    return out


def segment(pages: list[str]) -> list[Entry]:
    full = "\n".join(f"<<<PAGE {i+1}>>>\n{p}" for i, p in enumerate(pages))
    lines = full.splitlines()
    headers = toc_headers(lines)
    toc_start = min(i for i, l in enumerate(lines) if l.strip() == "Table of Contents/Index")
    body_lines = lines[:toc_start]
    norm_body = [_norm(l) for l in body_lines]

    positions: list[int] = []
    cursor = 0
    for h in headers:
        key = _norm(h)[:28]
        hit = next((i for i in range(cursor, len(norm_body))
                    if norm_body[i].startswith(key)), None)
        positions.append(hit if hit is not None else -1)
        if hit is not None:
            cursor = hit + 1

    entries: list[Entry] = []
    found = [(h, p) for h, p in zip(headers, positions) if p >= 0]
    for k, (h, p) in enumerate(found):
        end = found[k + 1][1] if k + 1 < len(found) else toc_start
        body = "\n".join(l for l in body_lines[p + 1:end] if not l.startswith("<<<PAGE"))
        ts, assumed = parse_timestamp(h)
        entries.append(Entry(k, h, ts, assumed, body))
    return entries


def code_entry(header: str, body: str) -> dict:
    text = header + "\n" + body[:BODY_CHARS]
    flags = {k: bool(r.search(text)) for k, r in _COMPILED.items()}
    # header-level signals are stronger than body mentions
    head_flags = {k: bool(r.search(header)) for k, r in _COMPILED.items()}
    primary = next((c for c in CAUSE_PRIORITY if head_flags[c]), None) or \
        next((c for c in CAUSE_PRIORITY if flags[c]), "none")
    return {
        **{f"cause_{k}": v for k, v in flags.items()},
        "primary_cause": primary,
        "interruption": bool(INTERRUPTION_RE.search(text)),
    }


UNPLANNED = ("hardware", "hang", "infra_software", "unknown")
CATEGORIES = UNPLANNED + ("optimization", "planned", "none")

# Final 175B run: Run 12.00 (first entry) to "done training" (2022-01-06 15:46 ET).
RUN_START = pd.Timestamp("2021-11-11T22:30:00Z")
RUN_END = pd.Timestamp("2022-01-06T20:46:44Z")
N_GPUS = 992  # 124 nodes x 8 A100-80GB (ngpu992 in every run directory)


def codebook_category(row) -> str:
    """Map the automatic codebook output onto the adjudication label set."""
    if not row["interruption"]:
        return "none"
    c = row["primary_cause"]
    if c in {"hw_gpu", "hw_node", "hw_network"}:
        return "hardware"
    return {"hang": "hang", "optim": "optimization", "ops": "infra_software"}.get(c, "unknown")


def build_event_table(pdf_path: Path, adjudication_csv: Path | None = None) -> pd.DataFrame:
    """One row per logbook entry with codebook labels and (if given) manual labels."""
    entries = segment(extract_text(pdf_path))
    rows = []
    for e in entries:
        rows.append({
            "entry_id": e.entry_id, "ts_utc": e.ts_utc, "tz_assumed": e.tz_assumed,
            "header": e.header, **code_entry(e.header, e.body),
            "body_excerpt": _norm(e.body)[:300],
        })
    df = pd.DataFrame(rows)
    df["codebook_category"] = df.apply(codebook_category, axis=1)
    df["category"] = pd.NA
    df["n_events"] = 0
    df["event_ts"] = df["ts_utc"]
    df["note"] = ""
    if adjudication_csv is not None:
        adj = pd.read_csv(adjudication_csv, comment="#", dtype=str, keep_default_na=False)
        for _, r in adj.iterrows():
            m = df.header == r.header_prefix  # exact match wins over prefix
            if not m.any():
                m = df.header.str.startswith(r.header_prefix)
            if m.sum() != 1:
                raise ValueError(f"prefix must match exactly one entry, got {m.sum()}: {r.header_prefix!r}")
            if r.category not in CATEGORIES:
                raise ValueError(f"bad category {r.category!r}")
            df.loc[m, "category"] = r.category
            df.loc[m, "n_events"] = int(r.n_events or 0)
            df.loc[m, "note"] = r.note
            if r.event_ts:
                df.loc[m, "event_ts"] = pd.Timestamp(r.event_ts).tz_convert("UTC")
        in_window = df.ts_utc.between(RUN_START - pd.Timedelta(hours=1), RUN_END)
        missing = df.loc[in_window & df.category.isna(), "header"].tolist()
        if missing:
            raise ValueError(f"unadjudicated entries in the run window: {missing}")
    return df.sort_values("event_ts", kind="stable").reset_index(drop=True)


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    labels = sorted(set(a) | set(b))
    po = float((a.values == b.values).mean())
    pe = sum((a == l).mean() * (b == l).mean() for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def interruption_times(events: pd.DataFrame, categories=UNPLANNED,
                       spread_hours: float = 4.0) -> pd.DatetimeIndex:
    """Expand adjudicated entries into individual interruption timestamps.

    Entries representing k>1 events (e.g. "autorecovery kicked in 3 times in the
    last few hours") are spread evenly over ``spread_hours`` centred on event_ts.
    """
    out = []
    sel = events[events.category.isin(categories) & (events.n_events > 0)]
    for _, r in sel.iterrows():
        k = int(r.n_events)
        if k == 1:
            out.append(r.event_ts)
        else:
            offs = [spread_hours * (i / (k - 1) - 0.5) for i in range(k)]
            out.extend(r.event_ts + pd.Timedelta(hours=o) for o in offs)
    ts = pd.DatetimeIndex(sorted(out))
    return ts[(ts >= RUN_START) & (ts <= RUN_END)]
