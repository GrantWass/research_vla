.PHONY: help test test-verbose setup check clean doctor inventory policies install-adapter setup-policy dry-run eval smoke smoke-all robot-check teleop
.DEFAULT: help

# Repo venv when present (robot deps), else system python.
PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

# Sim: make dry-run POLICY=turbovla TASK=stack_bowls
POLICY ?= pi05
TASK ?=
CKPT ?=
MODE ?= eval
# Robot: make robot-check ARM=2
ARM ?= 1

help:
	@echo "make test                Unit tests (CPU only; no GPU, Isaac or robot)"
	@echo "make setup | check       Clone upstream repos at pinned commits | verify the pins"
	@echo ""
	@echo "Robot (docs/robot.md)"
	@echo "make robot-check [ARM=1|2]  Read-only joint/camera check -> verify/piper_ARM/"
	@echo "make teleop [SECONDS=300]   Far arm leads, close arm follows"
	@echo ""
	@echo "Sim (docs/sim.md)"
	@echo "make policies            List registered policies (policies/*.conf)"
	@echo "make setup-policy POLICY=<p>  Policy env + checkpoint"
	@echo "make dry-run POLICY=<p> TASK=<t> [CKPT=<c>]  Resolve an eval command, no sim"
	@echo "make eval POLICY=<p> TASK=<t> [CKPT=<c>]     One eval"
	@echo "make smoke POLICY=<p> [TASK=<t>]             One smoke episode"
	@echo "make smoke-all [TASK=<t>]                    Smoke every policy, pass/fail table"
	@echo "make doctor | inventory  RoboDojo health check | runnable tasks"
	@echo "make install-adapter POLICY=<p>  Copy our adapter into XPolicyLab"
	@echo "make clean               Remove pyc/pycache files"

test:
	$(PY) -m unittest discover -s tests

test-verbose:
	$(PY) -m unittest discover -s tests -v

setup:
	bash setup.sh

check:
	bash setup.sh --check

# ---- robot

robot-check:
	$(PY) -m real_robot.verify --stage robot --config real_robot/configs/piper_$(ARM).json --out verify/piper_$(ARM)

SECONDS ?= 300
teleop:
	$(PY) -m real_robot.teleop --leader can_piper2 --follower-config real_robot/configs/piper_1.json --seconds $(SECONDS)

# ---- sim

doctor:
	cd RoboDojo && bash scripts/robodojo.sh doctor --skip-isaac --skip-conda --skip-policy

inventory:
	cd RoboDojo && python3 scripts/internal/task_inventory.py --format json --check | python3 -c "import json,sys; [print(t['name']) for t in json.load(sys.stdin)['tasks'] if t.get('runnable')]"

policies:
	bash scripts/run_eval.sh --list

install-adapter:
ifndef POLICY
	$(error Usage: make install-adapter POLICY=<policy>  (try: make policies))
endif
	bash scripts/install_adapter.sh $(POLICY)

setup-policy:
ifndef POLICY
	$(error Usage: make setup-policy POLICY=<demo|openvla|pi05|turbovla>)
endif
	bash scripts/setup_policy.sh $(POLICY) $(ARGS)

dry-run:
ifndef TASK
	$(error Usage: make dry-run POLICY=<policy> TASK=<task_name>  (try: make inventory))
endif
	bash scripts/run_eval.sh --policy $(POLICY) --task $(TASK) $(if $(CKPT),--ckpt $(CKPT)) --dry-run

eval:
ifndef TASK
	$(error Usage: make eval POLICY=<policy> TASK=<task_name> [CKPT=<ckpt>])
endif
	bash scripts/run_eval.sh --policy $(POLICY) --task $(TASK) $(if $(CKPT),--ckpt $(CKPT)) --mode $(MODE)

smoke:
	bash scripts/run_eval.sh --policy $(POLICY) $(if $(TASK),--task $(TASK)) $(if $(CKPT),--ckpt $(CKPT)) --mode smoke --fail-fast

# Every registered policy through Isaac Sim once; keeps going past failures.
SMOKE_POLICIES ?= demo openvla pi05 turbovla
smoke-all:
	@task=$${TASK:-stack_bowls}; fail=0; rows=""; \
	for p in $(SMOKE_POLICIES); do \
	  echo "=== smoke $$p ($$task) ==="; \
	  if bash scripts/run_eval.sh --policy $$p --task $$task --mode smoke --fail-fast; \
	  then rows="$$rows\n  PASS  $$p"; else rows="$$rows\n  FAIL  $$p"; fail=1; fi; \
	done; \
	printf "smoke-all ($$task):$$rows\n"; exit $$fail

clean:
	find . -name "*.pyc" -not -path "./RoboDojo/*" -not -path "./turbovla/*" -not -path "./.venv/*" | xargs rm -f 2>/dev/null; \
	find tests templates policies adapters scripts real_robot -name "__pycache__" | xargs rm -rf 2>/dev/null; true
