.PHONY: help test test-verbose setup check clean doctor inventory dry-run finetune-help
.DEFAULT: help

help:
	@echo "make test          Run CPU-only smoke suite (system python, no GPU/Isaac)"
	@echo "make test-verbose  Same, verbose output"
	@echo "make setup         Clone upstream repos at pinned commits (see setup.sh)"
	@echo "make check         Verify checkouts match pins (no network writes)"
	@echo "make doctor        RoboDojo health check (skips Isaac/conda/policy: Mac-safe)"
	@echo "make inventory     List runnable RoboDojo tasks (no Isaac)"
	@echo "make dry-run TASK=<task>  Resolve one eval command without launching sim"
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

dry-run:
ifndef TASK
	$(error Usage: make dry-run TASK=<task_name>  (try: make inventory))
endif
	cd RoboDojo && bash scripts/robodojo.sh eval \
	  --policy-dir XPolicyLab/policy/demo_policy \
	  --task $(TASK) --ckpt demo --policy-env demo-env --dry-run

finetune-help:
	@echo "See templates/openvla_finetune/README.md for the full checklist."
	@echo "GPU-box quick start:"
	@echo "  cd openvla && bash ../templates/openvla_finetune/finetune_lora.sh"

clean:
	find . -name "*.pyc" -not -path "./openvla/*" -not -path "./RoboDojo/*" | xargs rm -f 2>/dev/null; \
	find tests templates -name "__pycache__" | xargs rm -rf 2>/dev/null; true
