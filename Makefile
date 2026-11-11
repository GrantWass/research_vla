.PHONY: policy-server policy-server-stop help test test-verbose setup check clean doctor inventory policies install-adapter setup-policy dry-run eval smoke smoke-all robot-venv robot-check teleop quest-setup quest-check quest-teleop park status snapshot home move cans leader-restore policy policy-live record record-check convert merge finetune finetune-job offline-eval align jev-server jev-live
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
CONFIG ?= real_robot/configs/piper_1_banana.json

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
	@echo "make policy-server            Keep the pi0.5 policy loaded (run once, in its own terminal)"
	@echo "make policy [CKPT_HF=.. INSTR=.. STEPS=300 CONFIG=..]  pi0.5 dry run on the close arm"
	@echo "make policy-live [...]        Same, moving the arm (asks YES; hand on e-stop)"
	@echo "make offline-eval DATA=demos/x_pi05 CKPTS=\"new=.. old=..\"  Checkpoints vs recorded action chunks (no robot)"
	@echo "make align EP=20 [ALIGN_DATA=..]  Place cube/bowl like demo EP (live camera over its first frame)"
	@echo "make teleop [SECONDS=300]     Far arm leads, close arm follows"
	@echo "make record DIR=demos/x INSTR=\"...\"  Record demos (own terminal): homes arms; Space start, Enter save, r discard, q quit"
	@echo "make finetune [DATA=demos/x STEPS=3000]  Convert + fine-tune pi0.5 (action expert only, fits 24 GB)"
	@echo "make finetune-job [DATA=.. EPOCHS=1 BATCH=8]  Same, detached: log + outputs/finetune_report.md"
	@echo "make convert DIR=demos/x OUT=demos/x_pi05  Recorded dataset -> pi0.5 Piper training format"
	@echo "make record-check [SKIP_LEADER=1]  Pre-flight for record: CAN, follower, cameras, leader, dataset (follower never moves)"
	@echo "make robot-check [ARM=1|2]    Read-only joint/camera check -> verify/piper_ARM/"
	@echo "make quest-setup              Install Meta Quest teleop (conda ROS 2, no sudo)"
	@echo "make quest-check [ARM=1|2]    Read-only: Quest stack's arm driver, motors stay off"
	@echo "make quest-teleop [ARM=1|2] [QSPEED=60] [QSMOOTH=3]  Quest teleop (press+release A, B stops; then park)"
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
# The model flags are shared by the policy server and in-process runs.
NOISE_SEED ?= 0
MODEL_FLAGS = --lerobot $(CKPT_HF) --base-offset-deg 90 $(if $(NOISE_SEED),--noise-seed $(NOISE_SEED)) \
	$(if $(DENOISE),--denoise-steps $(DENOISE)) $(if $(BEST_OF),--best-of $(BEST_OF))
POLICY_SOCK ?= $(HOME)/.cache/real_robot/policy.sock
# With a policy server running (make policy-server), runs connect to it and
# start in ~1 s; otherwise they load and compile the model (~95 s).
POLICY_SRC = $(if $(wildcard $(POLICY_SOCK)),--policy-server $(POLICY_SOCK),--policy-process --compile reduce-overhead)
POLICY_FLAGS = --backend piper --config $(CONFIG) \
	$(MODEL_FLAGS) $(POLICY_SRC) --end-pose-state --instruction "$(INSTR)" \
	--max-steps $(STEPS) --execute-steps $(EXEC) --speed $(SPEED) \
	--async --blend-steps 4 --smooth-hz $(SMOOTH) $(if $(BLEND_FROM),--blend-from $(BLEND_FROM)) $(if $(SPEED_PCT),--speed-percent $(SPEED_PCT))

# Load + compile the policy once and keep it on the GPU (leave it running in
# a terminal; Ctrl-C or make policy-server-stop ends it).
policy-server:
	$(LEROBOT_PY) -m real_robot.lerobot_policy $(MODEL_FLAGS) --compile reduce-overhead --socket $(POLICY_SOCK)

policy-server-stop:
	$(LEROBOT_PY) -c "from real_robot.lerobot_policy import PolicyClient; PolicyClient('$(POLICY_SOCK)').call('shutdown')"

policy:
	$(LEROBOT_PY) -m real_robot.runner $(POLICY_FLAGS)

# PARK=1: press d when the task is done (or when STEPS run out) to park the arm
policy-live:
	$(LEROBOT_PY) -m real_robot.runner $(POLICY_FLAGS) --live $(if $(PARK),--park-when-done)

# Checkpoints vs the recorded actions of a converted dataset; no robot (real_robot/offline_eval.py)
OFFLINE_OUT ?= outputs/offline_eval
offline-eval:
	@test -n "$(DATA)" -a -n "$(CKPTS)" || (echo 'make offline-eval DATA=demos/<name>_pi05 CKPTS="new=outputs/<run>/checkpoints/last/pretrained_model old=..." [OFFLINE_OUT=..]'; exit 2)
	$(LEROBOT_PY) -m real_robot.offline_eval --dataset $(DATA) $(foreach c,$(CKPTS),--ckpt $(c)) --out $(OFFLINE_OUT)

# Set the scene up like recorded episode EP: live base camera over its first frame (real_robot/align_scene.py)
ALIGN_DATA ?= demos/rubiks_bowl_v5_straight_pi05
EP ?= 0
align:
	$(LEROBOT_PY) -m real_robot.align_scene --dataset $(ALIGN_DATA) --episode $(EP) --config $(CONFIG)

# Meta Quest teleop (AgileX QuestArmTeleop in conda env "vt"; real_robot/quest/).
QUEST_ENV = source $(HOME)/miniforge3/etc/profile.d/conda.sh && conda activate vt && \
	source $(HOME)/QuestArmTeleop/install/setup.bash && export PATH=$(HOME)/opt/platform-tools:$$PATH
QUEST_LAUNCH = ros2 launch real_robot/quest/teleop_quest_piper.launch.py arm:=$(ARM) rviz:=$(QRVIZ) mode:=$(QMODE) forward_yaw_deg:=$(QFORWARD) scale:=$(QSCALE) floor_mm:=$(QFLOOR)
QSPEED ?= 60
QSMOOTH ?= 3.0
QRVIZ ?= false  # RViz windows add lag (2026-10-08 traces)
QMODE ?= room  # room: 1:1 hand -> gripper in the room; controller: upstream's mapping
QFORWARD ?= -93.3  # base-frame yaw (deg) of your "forward" in room mode
QSCALE ?= 1.0
QFLOOR ?= 0  # mm above the arm's base plane (the table is ~15 mm below it): gripper/wrist floor
quest-setup:
	bash real_robot/quest/setup.sh

quest-check:
	bash -c '$(QUEST_ENV) && (timeout -s INT 8 $(QUEST_LAUNCH) check:=true rviz:=false >/dev/null 2>&1 &) && sleep 5 && \
		timeout 3 ros2 topic echo --once --field position /feedback/joint_states && \
		timeout 3 ros2 topic echo --once --field pose.position /feedback/tcp_pose; sleep 4; ros2 daemon stop >/dev/null'

quest-teleop:
	bash -c '$(QUEST_ENV) && $(QUEST_LAUNCH) speed_percent:=$(QSPEED) smooth_hz:=$(QSMOOTH) $(if $(TRACE),trace:=$(TRACE))'

record-check:
	$(LEROBOT_PY) -m real_robot.record_check $(if $(SKIP_LEADER),--skip-leader)

EPISODES ?= 0
record:
	@test -n "$(DIR)" || (echo "make record DIR=demos/<name> INSTR=\"<instruction>\""; exit 2)
	$(LEROBOT_PY) -m real_robot.teleop --record $(DIR) --task "$(INSTR)" --episodes $(EPISODES)

# Merge recorded datasets (e.g. batches of a long session) into one (real_robot/merge_datasets.py)
merge:
	@test -n "$(OUT)" -a -n "$(SRCS)" || (echo 'make merge OUT=demos/<name> SRCS="demos/<a> demos/<b> ..."'; exit 2)
	$(LEROBOT_PY) -m real_robot.merge_datasets $(OUT) $(SRCS)

# Fine-tune the pi0.5 checkpoint on DATA (default demos/rubiks_bowl_v5); scripts/finetune_pi05.sh
finetune:
	bash scripts/finetune_pi05.sh

# Detached fine-tune job (survives logout); report in outputs/finetune_report.md. EPOCHS=1 by default.
finetune-job:
	bash scripts/finetune_job.sh

# Recorded dataset -> pi0.5 Piper training format (real_robot/convert_dataset.py)
convert:
	@test -n "$(DIR)" -a -n "$(OUT)" || (echo "make convert DIR=demos/<name> OUT=demos/<name>_pi05"; exit 2)
	$(LEROBOT_PY) -m real_robot.convert_dataset $(DIR) $(OUT)

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

# JEV verifier + retry around pi0.5 (docs/jev.md). JEV server in its own terminal:
jev-server:
	bash scripts/serve_jev.sh

# make jev-live CKPT_HF=<ckpt> INSTR="..." JEV=on|off ATTEMPTS=3 [JEV_FLAGS=--jev-dry]
JEV ?= on
ATTEMPTS ?= 3
jev-live:
	$(LEROBOT_PY) -m real_robot.jev_guard $(POLICY_FLAGS) --live --jev $(JEV) --attempts $(ATTEMPTS) $(if $(PARK),--park-when-done) $(JEV_FLAGS)
