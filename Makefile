# Makefile for Module 1 - Osprey Backend

.PHONY: test smoke backfill alerts run lint fmt help

help:
	@echo "Available targets:"
	@echo "  make run       - Run Streamlit app (app/main.py)"
	@echo "  make test      - Run pytest tests"
	@echo "  make smoke     - Run smoke tests on sample files"
	@echo "  make backfill  - Backfill MongoDB documents with missing metrics"
	@echo "  make alerts    - Run alerts watch (monitors quality metrics and sends notifications)"
	@echo "  make lint      - Run ruff linter"
	@echo "  make fmt       - Format code with black"

run:
	@streamlit run app/main.py

test:
	@pytest -q

smoke:
	@python scripts/smoke_run.py

backfill:
	@python scripts/backfill_min_metrics.py

alerts:
	@python scripts/alerts_watch.py

lint:
	@ruff check .

fmt:
	@black .

