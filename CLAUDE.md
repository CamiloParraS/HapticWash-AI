# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

The Python data + model pipeline for HapticWash. Nothing here runs on the watch. It turns
public handwashing IMU datasets into one canonical corpus, trains a WHO-step classifier, and
exports a quantized LiteRT model. The Wear OS app lives in a separate repo (`HapticWash - Main`)
and consumes the release artifacts: `model.tflite`, `model_meta.json`, `golden/*.npy`.

Specs: `docs/SPEC-AI.md` (this repo) and `docs/HapticWash-SPEC.md` (shared contracts). Code
comments cite them as `SPEC x.y` / `§x.y`. Decisions are logged as `D<n>` in `docs/DECISIONS.md`.
Add a new dated `D<n>` entry there when you make a pipeline, dataset, or dependency decision.

## Commands

Uses `uv`, Python 3.11. There is no `make` on the Windows dev box, so run the underlying commands.

```sh
uv sync                                   # core deps
uv run ruff check . && uv run ruff format --check .   # lint (line length 100)
uv run pytest                             # all tests
uv run pytest tests/test_preprocess.py::test_name     # single test
uv run python -m haptic_ai.cli validate-schema tests/fixtures/canonical_100rows.csv
uv run python -m haptic_ai.cli ingest --all           # build data/processed/
uv run python -m haptic_ai.cli report-corpus          # reports/M1_corpus.{json,md} + figures
python scripts/fetch_data.py [key ...]    # download datasets into data/ (default: uwash, zhang_who)
```

`train` / `evaluate` / `export` / `golden` CLI subcommands take `--config configs/step_cnn.yaml`.

**TensorFlow (`ml` extra) is Linux-only** (D2). Model export and `scripts/make_stub_model.py`
run on CI (`.github/workflows/ci.yml`) or in WSL, not on native Windows.

## Architecture

- `haptic_ai/schema.py`: the canonical sensor table (SPEC 8.1). `LABELS` order is frozen and
  defines the model output tensor. `-1` means unlabelled. Nominal rate is 50 Hz. `subject_id`
  must be `<dataset_key>_<local_id>` with the key from `DATASET_KEYS`. `validate` rejects bad
  frames and never coerces them.
- `haptic_ai/ingest/<dataset>.py`: one `DatasetAdapter` per dataset. `ingest()` returns a
  canonical DataFrame (resampled, SI units, raw acc with gravity).
- `haptic_ai/corpus.py`: runs the adapters in `ADAPTERS`, splits sessions at gaps > 100 ms,
  validates, and writes Parquet partitioned by `subject_id` to `data/processed/`. The corpus
  stays raw on purpose (D16).
- `haptic_ai/preprocess.py`: per-session array functions (gravity-align, bandpass, mirroring)
  that M2 applies before windowing (`windows.py`, §8.7). Anything changed here must match the
  Kotlin implementation on the watch (§9.2).
- `features.py`, `models/` (`cnn1d`, `tree`), `evaluate.py`, `export.py`: training, eval, and
  LiteRT export. `docs/model_meta.schema.json` is the contract for `model_meta.json`.

## Data rules

- `data/` is gitignored and datasets are never committed or redistributed (`docs/DATASETS.md`).
- Split (D15): train on `uwash` with leave-one-subject-out. `zhang_who` is an external test set
  only and must never be used for training or model selection (CC-BY-NC-ND licence, D4).
- `ablutomania` has no step labels and is not ingested yet. `harage` was dropped (D7).
- Label mapping and known data issues: `docs/label_mapping.md`.
