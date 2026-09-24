# HapticWash-AI

> **This is the AI repo** (Python, never runs on the watch). The Wear OS app lives in the
> separate **`HapticWash - Main`** repo and consumes this repo's release artifacts
> (`model.tflite` + `model_meta.json` + `golden/*.npy`).

Data + model pipeline for HapticWash: turn public handwashing IMU datasets into one
canonical corpus, train a WHO-step classifier, and export a quantized LiteRT model for
the Wear OS app.

## Setup

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.11 (uv fetches it).

```sh
uv sync                 # core: schema, ingest, preprocess, evaluate
uv sync --all-extras    # + TensorFlow (Linux only; needed for model export)
```

## Datasets

Not redistributed. licences and citations in
[`docs/DATASETS.md`](docs/DATASETS.md).

- **zhang_who** — Zhang, Y. (2022). _Replication Data for: handwashing steps with IMU
  signals_ (V1.0). KU Leuven RDR. https://doi.org/10.48804/XHPPC7
- **uwash** — Wang, F. et al. (2025). _You Can Wash Hands Better: Accurate Daily
  Handwashing Assessment with a Smartwatch._ IEEE TMC. arXiv:2112.06657.
  https://github.com/aiotgroup/UWash
- **ablutomania** — Scholl, P. & Wahl, F. (2021). Ablutomania-Set.
- **ocdetect** — OCDetect, Zenodo record 13924901.
- **harage** — Mallol-Ragolta, A. et al. harAGE.
