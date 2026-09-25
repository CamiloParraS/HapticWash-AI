.PHONY: lint test validate-schema stub-model ingest report-corpus eval-trees eval-step export

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

eval-trees:
	uv run python -m haptic_ai.cli evaluate --config configs/step_trees.yaml

# eval-step and export need TensorFlow: Linux only (D2), run in WSL or CI.
eval-step:
	uv run --extra ml python -m haptic_ai.cli evaluate --config configs/step_cnn.yaml

export:
	uv run --extra ml python -m haptic_ai.cli export --config configs/step_cnn.yaml
