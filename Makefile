# Makefile for Docling project

.PHONY: test smoke backfill alerts help

help:
	@echo "Available targets:"
	@echo "  make test      - Run pytest tests"
	@echo "  make smoke     - Run smoke tests on sample files"
	@echo "  make backfill  - Backfill MongoDB documents with missing metrics"
	@echo "  make alerts    - Run alerts watch (monitors quality metrics and sends notifications)"

test:
	@pytest -q

smoke:
	@python scripts/smoke_run.py

backfill:
	@python scripts/backfill_min_metrics.py

alerts:
	@python scripts/alerts_watch.py

