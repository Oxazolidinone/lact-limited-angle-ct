# KLTN pipeline: LEARN deep unrolling for limited-angle CT. Every target calls the `lact` CLI (see `lact --help`).
# Per-machine settings go in configs/local.yaml (loaded automatically), or pass CFG="-c file.yaml".
# Data variants (Poisson noise, DeepLesion): add a file under configs/variant, e.g.
#   make pipeline DEV=cuda CFG="-c configs/variant/mayo_poisson_1e6.yaml"
# Environment: .venv in the project directory if present (python3 -m venv .venv), otherwise python from PATH.
VENV ?= .venv
BIN   = $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/,)
PY   ?= $(BIN)python
LACT ?= $(PY) -m lact
CFG  ?=
DEV  ?= auto
SET  ?=
GEOMS_MAIN  = fan_a90 fan_a120 fan_a150
GEOMS_CHECK = fan_full fan_short
GEOMS_HALF  = fan_a90_half fan_a120_half fan_a150_half
SETARG = $(if $(SET),-s $(SET),)
C = $(CFG) $(SETARG) --device $(DEV)

.PHONY: install dev test test-unit test-regression lint typecheck arch check-all inspect data operators check simulate tune-tv \
        pipeline pilot status source-zip download-mayo download-deeplesion e2 select-lr e3 e4 e5 e6 freeze eval cross cost tables figures analysis demo smoke clean-smoke

install:
	$(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e .
dev:
	$(PY) -m pip install -e ".[dev]"

# ---------------------------------------------------------------- code quality
test:
	$(PY) -m pytest -q
test-unit:            # fast, needs neither ASTRA nor data
	$(PY) -m pytest -q tests/unit
test-regression:      # pins outputs of finished runs; needs ../ket-qua or LACT_RESULTS_DIR (skips otherwise)
	$(PY) -m pytest -q tests/regression
lint:
	$(BIN)ruff check src tests scripts experiments
typecheck:
	$(BIN)mypy
arch:                 # check the dependency rules between layers
	$(BIN)lint-imports
check-all: lint typecheck arch test

# ---------------------------------------------------------------- data and operators (weeks 1-2)
download-mayo:        # download Mayo 2016 (full_3mm) from AAPM's public Box into the parent folder of data.dicom_root
	$(LACT) data download-mayo $(C)
download-deeplesion:  # second dataset: download zips 1-2 (~8 GiB) from NIH's Box, then split by patient
	$(LACT) data download-deeplesion -c configs/data/deeplesion.yaml $(SETARG)
	$(LACT) data split-deeplesion -c configs/data/deeplesion.yaml $(SETARG)
inspect:
	$(LACT) data inspect $(C)
data:
	$(LACT) data preprocess $(C)
operators:
	$(LACT) ops build $(C) -g $(GEOMS_MAIN) $(GEOMS_CHECK) $(GEOMS_HALF)
check:                # E0 + E1
	$(LACT) ops check $(C)
	$(LACT) ops density $(C)
simulate:
	$(LACT) data simulate $(C) -g $(GEOMS_MAIN)
tune-tv:
	$(LACT) eval tune-tv $(C) -g $(GEOMS_MAIN)
pilot:                # Gate 1: measure sec/step, estimate GPU-hours
	$(LACT) train pilot $(C)
status:               # progress of the training runs (heartbeat/summary)
	$(LACT) train status $(C)
source-zip:           # dist/kltn-lact.zip to move the source code to the server
	$(PY) scripts/pack_source.py

WORKERS ?= 1
PIPELINE_ARGS ?=
pipeline:             # runs the steps in dependency order; e.g. make pipeline DEV=cuda (reads configs/local.yaml)
	$(LACT) pipeline run $(C) --workers $(WORKERS) $(PIPELINE_ARGS)

# ---------------------------------------------------------------- training (weeks 5-8)
e2:
	$(LACT) train grid configs/experiment/e2_sweep.yaml $(C) --run
select-lr:
	$(LACT) report select-lr $(C) --exp e2_sweep
e3 e4 e5: %:
	$(LACT) train grid $(firstword $(wildcard configs/experiment/$@_*.yaml)) $(C) --run

# ---------------------------------------------------------------- evaluation and analysis (weeks 9-10)
freeze:
	$(LACT) eval freeze $(C) --note "configuration frozen before opening the test split"
eval:
	$(LACT) eval runs $(C) --exp e3_main e4_matched e5_a150 --split test
	$(LACT) eval baselines $(C) -g $(GEOMS_MAIN) --split test
e6 cross:
	$(LACT) eval cross configs/experiment/e6_crossgeo.yaml $(C) --split test
cost:
	$(LACT) eval cost $(C) --exp e3_main e4_matched
tables:
	$(LACT) report tables $(C) --split test
figures:
	$(LACT) report figures $(C) --split test
analysis:
	$(LACT) report analysis $(C) --split test
demo:
	$(LACT) demo $(CFG)

# ---------------------------------------------------------------- end-to-end smoke run on fake data
SMOKE = -c configs/env/smoke_fake.yaml
SMOKE_TRAIN = model.K=3 train.max_steps=6 train.limit_val=3
smoke:
	$(LACT) data fake --out data_raw/fake_mayo --n-slices 6 --size 128
	$(LACT) data preprocess $(SMOKE) --overwrite
	$(LACT) ops build $(SMOKE) -g $(GEOMS_MAIN) $(GEOMS_CHECK) $(GEOMS_HALF)
	$(LACT) ops check $(SMOKE)
	$(LACT) data simulate $(SMOKE) -g $(GEOMS_MAIN) --overwrite
	$(LACT) ops density $(SMOKE) --n-val 3 --tv-iters 30
	$(LACT) eval tune-tv $(SMOKE)
	$(LACT) train grid configs/experiment/e2_sweep.yaml $(SMOKE) -s $(SMOKE_TRAIN) --run
	$(LACT) report select-lr $(SMOKE)
	$(LACT) train grid configs/experiment/e3_main.yaml $(SMOKE) -s $(SMOKE_TRAIN) --run
	$(LACT) train grid configs/experiment/e4_matched.yaml $(SMOKE) -s $(SMOKE_TRAIN) --run
	$(LACT) train grid configs/experiment/e5_a150.yaml $(SMOKE) -s $(SMOKE_TRAIN) --run
	$(LACT) eval freeze $(SMOKE) --note smoke
	$(LACT) eval runs $(SMOKE) --exp e3_main e4_matched e5_a150 --split test
	$(LACT) eval baselines $(SMOKE) -g $(GEOMS_MAIN) --split test
	$(LACT) eval cross configs/experiment/e6_crossgeo.yaml $(SMOKE) --split test
	$(LACT) eval cost $(SMOKE) --exp e3_main e4_matched --iters 5 --warmup 1
	$(LACT) report tables $(SMOKE) --split test --n-boot 300
	$(LACT) report figures $(SMOKE) --split test
	$(LACT) report analysis $(SMOKE) --split test --n-erf 4 --n-stage 3

clean-smoke:
	rm -rf outputs_smoke data_raw/fake_mayo
