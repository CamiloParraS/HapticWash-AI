.PHONY: ingest train-step eval-step export golden test

ingest:
	uv run python -m haptic_ai.cli ingest --all

train-step:
	uv run python -m haptic_ai.cli train --config configs/step_cnn.yaml

eval-step:
	uv run python -m haptic_ai.cli evaluate --config configs/step_cnn.yaml

export:
	uv run python -m haptic_ai.cli export --config configs/step_cnn.yaml

golden:
	uv run python -m haptic_ai.cli golden --config configs/step_cnn.yaml

test:
	uv run pytest
