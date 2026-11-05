# Meta Quest teleop of one Piper arm. Run from this folder:
#   make setup | check | ready | teleop | park      (see README.md)
# Defaults are the settings that worked for us; override any on the command line,
# e.g. make teleop SMOOTH=2 SPEED=40.
.PHONY: setup check ready teleop park test

CAN ?=
SPEED ?= 60
SMOOTH ?= 3.0
MODE ?= room
FORWARD ?= -93.3
SCALE ?= 1.0
FLOOR ?= 0
RVIZ ?= false

HERE := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ENV = source $(HOME)/miniforge3/etc/profile.d/conda.sh && conda activate vt && \
	source $(HOME)/QuestArmTeleop/install/setup.bash && export PATH=$(HOME)/opt/platform-tools:$$PATH
LAUNCH = ros2 launch $(HERE)teleop.launch.py $(if $(CAN),can_port:=$(CAN)) rviz:=$(RVIZ) mode:=$(MODE) \
	forward_yaw_deg:=$(FORWARD) scale:=$(SCALE) floor_mm:=$(FLOOR)

setup:
	bash $(HERE)setup.sh

# Read-only: the driver with the motors off prints joints and TCP pose.
check:
	bash -c '$(ENV) && ($(LAUNCH) check:=true >/dev/null 2>&1 &) && sleep 5 && \
		timeout 3 ros2 topic echo --once --field position /feedback/joint_states && \
		timeout 3 ros2 topic echo --once --field pose.position /feedback/tcp_pose; \
		pkill -INT -f teleop.launch.py; sleep 4; pkill -KILL -f agx_arm_ctrl_single; ros2 daemon stop >/dev/null'

ready:
	bash -c '$(ENV) && python $(HERE)arm_tool.py ready $(if $(CAN),--can $(CAN))'

teleop:
	bash -c '$(ENV) && $(LAUNCH) speed_percent:=$(SPEED) smooth_hz:=$(SMOOTH) $(if $(TRACE),trace:=$(TRACE))'

park:
	bash -c '$(ENV) && python $(HERE)arm_tool.py park $(if $(CAN),--can $(CAN))'

test:
	python3 -m unittest discover -s $(HERE)tests
