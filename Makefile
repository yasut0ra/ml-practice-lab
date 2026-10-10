SHELL := /bin/bash

PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: setup run check answers visuals clean

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run:
	$(PYTHON) scripts/run_all.py

check:
	$(PYTHON) -m compileall -q scripts answers tests
	$(PYTHON) scripts/check_materials.py
	$(PYTHON) -m unittest discover -s tests -v
	$(PYTHON) scripts/run_all.py

answers:
	$(PYTHON) scripts/run_all.py --answers

visuals:
	$(PYTHON) scripts/visualize.py

clean:
	rm -rf .venv __pycache__ scripts/__pycache__ answers/__pycache__ .pytest_cache outputs
