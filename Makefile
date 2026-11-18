PY ?= python
export PYTHONPATH := src

.PHONY: data test quick all check calc
data:            ## download raw traces (~1 GB download, ~3 GB extracted)
	bash scripts/download_data.sh
test:
	$(PY) -m pytest -q tests
quick:           ## ~2 min smoke run of the full pipeline
	$(PY) scripts/run_analysis.py --quick
all:             ## full-precision run (writes results/), then verify README numbers
	$(PY) scripts/run_analysis.py
	$(PY) scripts/check_readme_numbers.py
check:           ## README numbers == results/
	$(PY) scripts/check_readme_numbers.py
calc:
	$(PY) -m llm_reliability.cli --gpus 992 --ckpt-cost 1m --restart 30m --preset opt175b --current 80m
