PY = .venv/bin/python

venv:
	python3 -m venv .venv
	.venv/bin/pip install -e ".[dev]"

test:
	.venv/bin/pytest -q

data:
	$(PY) scripts/fetch_data.py
	$(PY) scripts/generate_instructions.py

run:
	$(PY) -m typo_study.runner config/experiment.yaml

analyze:
	$(PY) -m typo_study.analysis config/experiment.yaml

.PHONY: venv test data run analyze
