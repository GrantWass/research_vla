.PHONY: help test test-verbose setup check clean doctor inventory dry-run eval smoke policies install-adapter finetune-help
.DEFAULT: help

# Model selector: make dry-run POLICY=turbovla TASK=stack_bowls
POLICY ?= openvla
TASK ?=
CKPT ?=
MODE ?= eval

help:
	@echo "make test          Run CPU-only smoke suite (system python, no GPU/Isaac)"
	@echo "make test-verbose  Same, verbose output"
	@echo "make setup         Clone upstream repos at pinned commits (see setup.sh)"
	@echo "make check         Verify checkouts match pins (no network writes)"
	@echo "make doctor        RoboDojo health check (skips Isaac/conda/policy: Mac-safe)"
	@echo "make inventory     List runnable RoboDojo tasks (no Isaac)"
	@echo "make policies      List registered VLA policies (policies/*.conf)"
	@echo "make dry-run POLICY=<p> TASK=<task>  Resolve one eval command without launching sim"
	@echo "make eval POLICY=<p> TASK=<task> [CKPT=<c>]     Run one eval (GPU box)"
	@echo "make smoke POLICY=<p> [TASK=<t>] [CKPT=<c>]     Run smoke (GPU box)"
	@echo "make install-adapter POLICY=<p>  Copy this-repo adapter into XPolicyLab"
	@echo "make finetune-help Show OpenVLA LoRA fine-tune template usage"
	@echo "make clean         Remove pyc/pycache files"

test:
	python3 -m unittest discover -s tests

test-verbose:
	python3 -m unittest discover -s tests -v

setup:
	bash setup.sh

check:
	bash setup.sh --check

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

finetune-help:
	@echo "OpenVLA LoRA:   see templates/openvla_finetune/README.md"
	@echo "  cd openvla && bash ../templates/openvla_finetune/finetune_lora.sh"
	@echo "TurboVLA:       see templates/turbovla_finetune/README.md"
	@echo "  bash scripts/install_turbovla_training.sh"
	@echo "  cd turbovla && ROBODOJO_DATA_ROOT=/data/... bash ../templates/turbovla_finetune/train.sh"
	@echo "pi0.5:          upstream adapter (GPU box, uv-managed, no template)"
	@echo "  cd RoboDojo/XPolicyLab/policy/Pi_05 && bash install.sh"
	@echo "  bash process_data.sh RoboDojo cotrain arx_x5 joint"
	@echo "  bash train.sh RoboDojo cotrain arx_x5 joint 0 0"

clean:
	find . -name "*.pyc" -not -path "./openvla/*" -not -path "./RoboDojo/*" -not -path "./turbovla/*" | xargs rm -f 2>/dev/null; \
	find tests templates policies adapters scripts -name "__pycache__" | xargs rm -rf 2>/dev/null; true
