#!/usr/bin/env bash
# Fetch the three public sources into data/raw and verify checksums.
set -euo pipefail
cd "$(dirname "$0")/../data/raw"
mkdir -p opt acme pai
M=https://raw.githubusercontent.com/facebookresearch/metaseq/main/projects/OPT/chronicles
curl -sSL -o opt/OPT175B_Logbook.pdf https://github.com/facebookresearch/metaseq/raw/main/projects/OPT/chronicles/OPT175B_Logbook.pdf
for f in 10_percent_update 27_percent_update 56_percent_update final_update README; do curl -sSL -o opt/$f.md $M/$f.md; done
curl -sSL -o acme/trace_kalos.csv https://raw.githubusercontent.com/InternLM/AcmeTrace/main/data/job_trace/trace_kalos.csv
curl -sSL -o acme/trace_seren.csv https://huggingface.co/datasets/Qinghao/AcmeTrace/resolve/main/data/job_trace/trace_seren.csv  # Git-LFS on GitHub
B=https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces
for t in pai_job_table pai_task_table pai_machine_spec pai_instance_table; do
  curl -sSL -o pai/$t.tar.gz $B/$t.tar.gz && tar -xzf pai/$t.tar.gz -C pai
done
(cd . && sha256sum -c SHA256SUMS --ignore-missing)
