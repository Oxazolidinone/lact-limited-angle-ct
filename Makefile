# LACT: LEARN-DenseNet with an L-BFGS data step and the compared methods, for limited-angle CT.
# Every target calls the `lact` CLI (see `lact --help`); per-machine settings go in configs/local.yaml.
#   make data-joint && make train-joint && make eval-joint        # Mayo + DeepLesion
#   make data-detect && make train-detect && make eval-detect     # measured 2DeteCT
#   make smoke                                                    # the same steps on synthetic data, in minutes
# Environment: .venv in the project directory if present (python3 -m venv .venv), otherwise python from PATH.
VENV ?= .venv
BIN   = $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/,)
PY   ?= $(BIN)python
LACT ?= $(PY) -m lact
DEV  ?= auto
GPUS ?=
GPUARG = $(if $(GPUS),--gpus $(GPUS),)
JOINT  = -c configs/variant/joint.yaml
DETECT = -c configs/variant/detect.yaml
E = configs/experiment
GRIDS_JOINT  = $(E)/base_mayo.yaml $(E)/base_dl.yaml $(E)/lbfgs_mayo.yaml $(E)/lbfgs_dl.yaml $(E)/cmp_mayo.yaml \
               $(E)/cmp_dl.yaml $(E)/dpma1_mayo.yaml $(E)/dpma1_dl.yaml $(E)/dpma_mayo.yaml $(E)/dpma_dl.yaml
GRIDS_DETECT = $(E)/base_dt.yaml $(E)/lbfgs_dt.yaml $(E)/cmp_dt.yaml $(E)/dpma1_dt.yaml $(E)/dpma_dt.yaml
EXPS_JOINT   = base_mayo base_dl lbfgs_mayo lbfgs_dl cmp_mayo cmp_dl dpma_mayo dpma_dl
EXPS_DETECT  = base_dt lbfgs_dt cmp_dt dpma_dt

.PHONY: install dev test test-unit test-regression lint typecheck arch check-all \
        data-joint data-detect train-joint train-detect eval-joint eval-detect smoke clean-smoke

install:
	$(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e .
dev:
	$(PY) -m pip install -e ".[dev]"

# ---------------------------------------------------------------- code quality
test:
	$(PY) -m pytest -q
test-unit:            # fast, needs neither ASTRA nor data
	$(PY) -m pytest -q tests/unit
test-regression:      # pins outputs of finished runs; needs LACT_RESULTS_DIR (skipped otherwise)
	$(PY) -m pytest -q tests/regression
lint:
	$(BIN)ruff check src tests
typecheck:
	$(BIN)mypy
arch:                 # check the dependency rules between layers
	$(BIN)lint-imports
check-all: lint typecheck arch test

# ---------------------------------------------------------------- data: Mayo + DeepLesion (joint store), 2DeteCT
data-joint:
	$(LACT) data download-mayo $(JOINT)
	$(LACT) data download-deeplesion $(JOINT) --key-slices
	$(LACT) data preprocess $(JOINT) configs/data/split_joint.yaml
	$(LACT) ops build $(JOINT) -g fan_a90 fan_a120
	$(LACT) ops build $(JOINT) -g fan_a240 --sizes 256
	$(LACT) data simulate $(JOINT) configs/data/split_joint.yaml -g fan_a90 fan_a120 --device $(DEV)
	$(LACT) eval tune-tv $(JOINT) configs/data/split_joint.yaml -g fan_a90 --device $(DEV)
	$(LACT) eval tune-tv $(JOINT) configs/data/split_joint.yaml -g fan_a120 --device $(DEV)
data-detect:
	$(LACT) data download-2detect -c configs/env/server.yaml configs/data/detect.yaml
	$(LACT) data preprocess $(DETECT) configs/data/split_detect.yaml
	$(LACT) ops build $(DETECT) -g dt_a90 dt_a120 dt_a90_to240 dt_a120_to240 --sizes 256
	$(LACT) data simulate $(DETECT) configs/data/split_detect.yaml -g dt_a90 dt_a120 --device $(DEV)
	$(LACT) eval tune-tv $(DETECT) configs/data/split_detect.yaml -g dt_a90 --device $(DEV)
	$(LACT) eval tune-tv $(DETECT) configs/data/split_detect.yaml -g dt_a120 --device $(DEV)

# ---------------------------------------------------------------- training (resumable; DONE runs are skipped)
train-joint:
	$(LACT) train grid $(GRIDS_JOINT) $(JOINT) --run $(GPUARG)
train-detect:
	$(LACT) train grid $(GRIDS_DETECT) $(DETECT) --run $(GPUARG)

# ---------------------------------------------------------------- test evaluation (per-slice metrics)
eval-joint:
	$(LACT) eval freeze $(JOINT) --note "configuration frozen before opening the test split"
	$(LACT) eval runs $(JOINT) configs/data/split_joint.yaml --exp $(EXPS_JOINT) --split test --device $(DEV)
	$(LACT) eval baselines $(JOINT) configs/data/split_joint.yaml --split test -g fan_a90 fan_a120 --device $(DEV)
eval-detect:
	$(LACT) eval freeze $(DETECT) --note "configuration frozen before opening the test split"
	$(LACT) eval runs $(DETECT) configs/data/split_detect.yaml --exp $(EXPS_DETECT) --split test --device $(DEV)
	$(LACT) eval baselines $(DETECT) configs/data/split_detect.yaml --split test -g dt_a90 dt_a120 --device $(DEV)

# ---------------------------------------------------------------- end-to-end run on synthetic data (not for results)
SMOKE = -c configs/env/smoke_fake.yaml
SMOKE_TRAIN = model.K=3 train.max_steps=6 train.limit_val=3
smoke:
	$(LACT) data fake --out data_raw/fake_mayo --n-slices 6 --size 128
	$(LACT) data preprocess $(SMOKE) --overwrite
	$(LACT) ops build $(SMOKE) -g fan_a90
	$(LACT) data simulate $(SMOKE) -g fan_a90 --overwrite
	$(LACT) eval tune-tv $(SMOKE) -g fan_a90 --device $(DEV)
	$(LACT) train run $(SMOKE) -m learn_densenet -g fan_a90 -s $(SMOKE_TRAIN) exp=smoke --device $(DEV)
	$(LACT) train run $(SMOKE) -m learn_densenet -g fan_a90 -s $(SMOKE_TRAIN) exp=smoke_lbfgs \
	        model.update=lbfgs model.lbfgs_damping=0.5 --device $(DEV)
	$(LACT) eval freeze $(SMOKE) --note smoke
	$(LACT) eval runs $(SMOKE) --exp smoke smoke_lbfgs --split test --device $(DEV)
	$(LACT) eval baselines $(SMOKE) -g fan_a90 --split test --device $(DEV)
clean-smoke:
	rm -rf outputs_smoke data_raw/fake_mayo
