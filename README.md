# HapticWash-AI

Data + model pipeline for HapticWash: turn public handwashing IMU datasets into one
canonical corpus, train a WHO-step classifier, and export a quantized LiteRT model for
the Wear OS app. Spec: [`docs/HapticWash-SPEC.md`](docs/HapticWash-SPEC.md).

## Setup

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.11 (uv fetches it).

```sh
uv sync                 # core: schema, ingest, preprocess, evaluate
uv sync --all-extras    # + TensorFlow (Linux only; needed for model export)
```

The `ml` extra (TensorFlow) is Linux-only — see `docs/DECISIONS.md` D2. Model export
runs on CI or WSL. `make` targets: `validate-schema`, `lint`, `test`, `stub-model`.

## Datasets

Not redistributed. `data/` is gitignored; licences and citations in
[`docs/DATASETS.md`](docs/DATASETS.md). ⚠️ `zhang_who` is CC-BY-NC-ND-4.0 — see
`docs/DECISIONS.md` D4 before commercial use.

- **zhang_who** — Zhang, Y. (2022). *Replication Data for: handwashing steps with IMU
  signals* (V1.0). KU Leuven RDR. https://doi.org/10.48804/XHPPC7
- **ablutomania** — Scholl, P. & Wahl, F. (2021). Ablutomania-Set.
- **ocdetect** — OCDetect, Zenodo record 13924901.
- **harage** — Mallol-Ragolta, A. et al. harAGE.
