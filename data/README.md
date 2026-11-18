# Data provenance

| Source | File(s) | Origin | License / terms |
|---|---|---|---|
| OPT-175B logbook | `raw/opt/OPT175B_Logbook.pdf` (114 pp) | facebookresearch/metaseq `projects/OPT/chronicles` | MIT (metaseq repo) |
| Acme Seren / Kalos job traces | `raw/acme/trace_{seren,kalos}.csv` | InternLM/AcmeTrace (Hu et al., NSDI '24); Seren via HF `Qinghao/AcmeTrace` (LFS) | see AcmeTrace repo |
| Alibaba PAI GPU trace v2020 | `raw/pai/pai_*_table.csv` | alibaba/clusterdata `cluster-trace-gpu-v2020` (Weng et al., NSDI '22) | CC BY 4.0 |

`raw/SHA256SUMS` pins the exact bytes analysed. `make data` re-downloads and verifies.

## Derived
* `annotations/opt_adjudication.csv`: **hand-coded** label for every logbook entry in the final-run
  window (Run 12.00 → completion), with a one-line justification and, where the entry states it,
  the actual failure time. This file is the primary human input of the study; the regex codebook
  in `src/llm_reliability/data/opt_logbook.py` is kept only to report agreement (Cohen's κ).
* `processed/opt175b_logbook_events.csv`: all 131 dated entries, codebook + manual labels.
* `processed/acme_{cluster}_pretrain_chains.csv`: Pretrain jobs grouped into restart chains.

Operator names appear in the source logbook; derived tables keep the original headers for
traceability but no analysis depends on who wrote an entry.
