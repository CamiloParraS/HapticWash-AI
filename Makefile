.PHONY: lint test validate-schema stub-model ingest report-corpus train-step eval-step export golden

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest

validate-schema:
	uv run python -m haptic_ai.cli validate-schema tests/fixtures/canonical_100rows.csv

stub-model:
	uv run --extra ml python scripts/make_stub_model.py

ingest:
	uv run python -m haptic_ai.cli ingest --all

report-corpus:
	uv run python -m haptic_ai.cli report-corpus

train-step:
	uv run python -m haptic_ai.cli train --config configs/step_cnn.yaml

eval-step:
	uv run python -m haptic_ai.cli evaluate --config configs/step_cnn.yaml

export:
	uv run python -m haptic_ai.cli export --config configs/step_cnn.yaml

golden:
	uv run python -m haptic_ai.cli golden --config configs/step_cnn.yaml
