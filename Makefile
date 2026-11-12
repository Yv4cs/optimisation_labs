PYTHON ?= python
EXP ?= configs/experiments/t4_qwen3b_fp16.yaml

.PHONY: help setup setup-dev freeze env test smoke bench

help:  ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

setup:  ## GPU box (Kaggle/RunPod): runtime deps + this package
	$(PYTHON) -m pip install -r requirements-kaggle.txt
	$(PYTHON) -m pip install -e .

setup-dev:  ## laptop without GPU: package + tests only
	$(PYTHON) -m pip install -e ".[dev]"

freeze:  ## record exact installed versions for this environment
	mkdir -p env
	$(PYTHON) -m pip freeze --exclude-editable > env/lock-kaggle.txt

env:  ## print GPU, driver, CUDA and runtime versions
	$(PYTHON) -m infer_lab.env_info

test:  ## unit tests (no GPU needed)
	$(PYTHON) -m pytest -q

smoke:  ## 1 concurrency level, 1 repeat, few requests; logs to results/smoke/
	$(PYTHON) -m infer_lab.runner --exp $(EXP) --smoke

bench:  ## full sweep for EXP; appends to results/runs.jsonl
	$(PYTHON) -m infer_lab.runner --exp $(EXP)
