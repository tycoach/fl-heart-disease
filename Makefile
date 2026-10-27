.PHONY: setup data audit siloed centralized federated compare all test lint clean

setup:
	pip install -e ".[dev]"

data:
	python scripts/download_data.py

audit:
	python scripts/audit_data.py

siloed:
	python scripts/run_siloed.py

centralized:
	python scripts/run_centralized.py

federated:
	python scripts/run_federated.py

compare:
	python scripts/compare_results.py

all: siloed centralized federated compare

test:
	pytest -q

lint:
	ruff check src scripts tests

clean:
	rm -rf results/metrics/* results/figures/* results/logs*/*
