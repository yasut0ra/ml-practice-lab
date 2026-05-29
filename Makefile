SHELL := /bin/bash

PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: setup run check answers clean

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run:
	$(PYTHON) scripts/run_all.py

check:
	$(PYTHON) -m compileall scripts answers
	$(PYTHON) scripts/run_all.py --fast

answers:
	for file in answers/*.py; do \
		echo "== $$file =="; \
		$(PYTHON) $$file; \
	done

clean:
	rm -rf .venv __pycache__ scripts/__pycache__ answers/__pycache__ .pytest_cache outputs
