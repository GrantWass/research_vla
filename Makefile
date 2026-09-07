.PHONY: help test test-verbose setup check clean
.DEFAULT: help

help:
	@echo "make test          Run CPU-only smoke suite (system python, no GPU/Isaac)"
	@echo "make test-verbose  Same, verbose output"
	@echo "make setup         Clone upstream repos at pinned commits (see setup.sh)"
	@echo "make check         Verify checkouts match pins (no network writes)"
	@echo "make clean         Remove pyc/pycache files"

test:
	python3 -m unittest discover -s tests

test-verbose:
	python3 -m unittest discover -s tests -v

setup:
	bash setup.sh

check:
	bash setup.sh --check

clean:
	find . -name "*.pyc" -not -path "./openvla/*" -not -path "./RoboDojo/*" | xargs rm -f 2>/dev/null; \
	find tests templates -name "__pycache__" | xargs rm -rf 2>/dev/null; true
