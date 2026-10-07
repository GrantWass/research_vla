.PHONY: help test test-verbose setup check clean doctor inventory policies install-adapter setup-policy dry-run eval smoke smoke-all robot-venv robot-check teleop park status snapshot home move cans leader-restore policy policy-live record
.DEFAULT: help

# Repo venv when present (robot deps), else system python.
PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
UV ?= $(or $(shell command -v uv),$(HOME)/.local/bin/uv)

# Sim: make dry-run POLICY=turbovla TASK=stack_bowls
POLICY ?= pi05
TASK ?=
CKPT ?=
MODE ?= eval
# Robot: make robot-check ARM=2, make park ARMS="1 2"
ARM ?= 1
ARMS ?= 1 2
# LeRobot venv (pi0.5 policies, demo recording): ROBOT_PC_SETUP in docs/robot.md
LEROBOT_PY ?= $(HOME)/lerobot_robot_piper/.venv/bin/python
# make policy: LeRobot checkpoint, instruction, steps; LIVE=1 moves the arm
CKPT_HF ?= richardshkim/pi05_piper_banana_v2_full
INSTR ?= Put banana into basket.
STEPS ?= 300
SPEED ?= 0.5
EXEC ?= 8
SMOOTH ?= 2

help:
	@echo "make test                Unit tests (CPU only; no GPU, Isaac or robot)"
	@echo "make setup | check       Clone upstream repos at pinned commits | verify the pins"
	@echo ""
	@echo "Robot (docs/robot.md)"
	@echo "make robot-venv          Create .venv from requirements/robot.txt"
	@echo "make status [ARMS=\"1 2\"]    Read-only: joints, end pose, motors, base vs home"
	@echo "make snapshot [ARMS=...]      Save every camera's frame -> verify/snapshots/<time>/"
	@echo "make home [ARMS=...]          Fold + face the table, motors stay on (asks first)"
	@echo "make park [ARMS=...]          Fold + face the table + gentle power-down (asks first)"
	@echo "make move ARM=1 TO=j1,..,j6 | BY=d1,..,d6 [GRIP=mm]  Ease one arm (deg; asks first)"
	@echo "make cans                     CAN links up? | make leader-restore: far arm out of teach mode"
	@echo "make policy [CKPT_HF=.. INSTR=.. STEPS=300]  Fast pi0.5 dry run on the close arm"
	@echo "make policy-live [...]        Same, moving the arm (asks YES; hand on e-stop)"
	@echo "make teleop [SECONDS=300]     Far arm leads, close arm follows"
	@echo "make record DIR=demos/x INSTR=\"...\" [EPISODES=10]  Teleop + record LeRobot demos"
	@echo "make robot-check [ARM=1|2]    Read-only joint/camera check -> verify/piper_ARM/"
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

robot-venv:
	$(UV) venv --python 3.10 .venv
	$(UV) pip install --python .venv/bin/python -r requirements/robot.txt
	$(UV) pip install --python .venv/bin/python --no-deps pyorbbecsdk2==2.1.2

robot-check:
	$(PY) -m real_robot.verify --stage robot --config real_robot/configs/piper_$(ARM).json --out verify/piper_$(ARM)

SECONDS ?= 300
teleop:
	$(PY) -m real_robot.teleop --leader can_piper2 --follower-config real_robot/configs/piper_1.json --seconds $(SECONDS)

# Arm commands (real_robot/arm.py); arm 1 = close, 2 = far.
status:
	$(PY) -m real_robot.arm status --arms $(ARMS)

snapshot:
	$(PY) -m real_robot.arm snapshot --arms $(ARMS)

home:
	$(PY) -m real_robot.arm home --arms $(ARMS)

park:
	$(PY) -m real_robot.arm park --arms $(ARMS)

move:
	$(PY) -m real_robot.arm move --arm $(ARM) $(if $(TO),--to $(TO),--by $(BY)) $(if $(GRIP),--grip $(GRIP))

cans:
	@ip -br link | grep can || echo "no CAN links: plug in the adapters (docs/robot.md)"

leader-restore:
	$(PY) -m real_robot.teleop --leader can_piper2 --restore-leader

# pi0.5 on the close arm: child process, CUDA graphs, async replanning (~3/s)
# with fixed sampling noise and 4-step blending, at half the trained rate,
# commands streamed at 100 Hz through a 2 Hz smoothing filter; docs/robot.md.
POLICY_FLAGS = --backend piper --config real_robot/configs/piper_1_banana.json \
	--lerobot $(CKPT_HF) --end-pose-state --base-offset-deg 90 --instruction "$(INSTR)" \
	--max-steps $(STEPS) --execute-steps $(EXEC) --speed $(SPEED) \
	--policy-process --compile reduce-overhead --async --blend-steps 4 --noise-seed 0 --smooth-hz $(SMOOTH)
policy:
	$(LEROBOT_PY) -m real_robot.runner $(POLICY_FLAGS)

policy-live:
	$(LEROBOT_PY) -m real_robot.runner $(POLICY_FLAGS) --live

EPISODES ?= 10
record:
	@test -n "$(DIR)" || (echo "make record DIR=demos/<name> INSTR=\"<instruction>\""; exit 2)
	$(LEROBOT_PY) -m real_robot.teleop --record $(DIR) --task "$(INSTR)" --episodes $(EPISODES)

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
