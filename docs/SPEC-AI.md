# HapticWash-AI — AI Pipeline Specification

**Repo:** `HapticWash-AI` (Python, data + model pipeline)
**Scope:** datasets → canonical corpus → training → honest evaluation → quantized LiteRT
model + `model_meta.json` + golden files. **Nothing in this repo runs on the watch.**

> Shared contracts (label set §1.3, CSV §8.1, `model_meta.json` §8.2, tensor §8.3,
> preprocessing §8.7, parity §9.2, budgets §4.3, agent rules §10) live in the
> [Core spec](HapticWash-SPEC.md), which wins on conflict. The Wear OS app has its own
> spec (`HapticWash - Main/docs/SPEC-APP.md`). § numbers are global.

---

## 2. Tech Stack

### 2.2 `HapticWash-AI` — model pipeline

| Concern | Choice |
|---|---|
| Language | Python 3.11 |
| Env | `uv` (preferred) or conda; lockfile committed |
| Numerics | numpy, scipy, pandas |
| Classical ML | scikit-learn, imbalanced-learn |
| Deep ML | PyTorch (export via `ai-edge-torch`) **or** Keras (export via `tf.lite`) — pick one at M2 and record the decision in `docs/DECISIONS.md` |
| Features | hand-rolled in `haptic_ai/features.py`; `tsfresh` allowed for exploration only, never in the shipped path |
| Plots | matplotlib |
| Tests | pytest |
| Orchestration | `make` targets + a `configs/*.yaml` per experiment. No workflow engine. |
| Tracking | JSON metric reports committed to `reports/`. No MLflow server. |

**Datasets are never committed.** `data/raw/` is gitignored. `scripts/fetch_data.py`
downloads them and verifies checksums.

## 3. Repository Layout

```
HapticWash-AI/                       # model pipeline
├── haptic_ai/
│   ├── ingest/                      # one adapter module per source dataset
│   ├── schema.py                    # canonical CSV schema + validators
│   ├── preprocess.py                # resample, filter, gravity-align, mirror
│   ├── features.py                  # feature extraction (mirrors app implementation)
│   ├── windows.py                   # windowing + label assignment
│   ├── models/                      # baseline_tree.py, cnn1d.py
│   ├── evaluate.py                  # LOSO CV, metrics, confusion matrices
│   └── export.py                    # quantize → tflite + emit model_meta.json
├── configs/
├── scripts/fetch_data.py
├── tests/
├── reports/                         # committed metric JSON + figures
├── artifacts/                       # gitignored; release outputs
└── docs/DECISIONS.md
```

## 4. Architectural Constraints

### 4.4 Determinism

Every training run pins seeds for `random`, `numpy`, and the DL framework, and records
them in the run's report JSON. A rerun of the same config must reproduce reported metrics
to within ±0.005 macro-F1. Non-reproducible results are treated as bugs.

---

## 5. Data Sources

Fetched by `scripts/fetch_data.py`. **Verify each licence before use and record it in
`docs/DATASETS.md`. Do not redistribute any dataset in either repository.** Cite all of
them in both READMEs.

| Key | Source | Role |
|---|---|---|
| `zhang_who` | Zhang et al., "Replication Data for: handwashing steps with IMU signals", KU Leuven RDR, DOI `10.48804/XHPPC7` | **Primary** — WHO step labels, 10 subjects, Byteflies 100 Hz. Trains the step classifier. |
| `uwash` | Wang et al., UWash, IEEE TMC 2025 / arXiv `2112.06657`; data via `github.com/aiotgroup/UWash` | **Primary** — per-sample WHO step labels, 51 subjects × 5 locations, Samsung Gear Sport 50 Hz acc+gyro. Trains the step classifier (D14). |
| `ablutomania` | Ablutomania-Set (Scholl & Wahl, 2021) | Handwashing plus deliberate confounders (e.g. rinsing a cup). Hard negatives. |
| `ocdetect` | OCDetect, Zenodo record `13924901`; pipeline at `github.com/OCDetect/OCDetect-pipeline` | 22 participants, ~2600 h of all-day Wear OS recordings at 50 Hz acc+gyro, ~2930 labelled washes. Realistic NULL/background for spotting. **Ingested at M5, not M1** (D8). |
| `own_phone` | Self-collected, phone strapped to forearm | Augmentation and confounder collection **only**. Never used for evaluation. |
| `own_watch` | Borrowed Wear OS device (M6) | **Held-out test set. Never used for training or model selection.** |

Each dataset needs an adapter in `haptic_ai/ingest/` producing the canonical schema in
§8.1, plus an explicit label-mapping table committed at `docs/label_mapping.md`. Where a
source label cannot be mapped confidently to §1.3, map it to `WASH_OTHER` and record why.
Never invent a mapping to increase class counts.

---

## 6. Milestone Requirements

AI-side milestones. M0 and M6 are in the Core spec; M3/M4 in the App spec.

---

### M1 — Unified corpus

**Objective.** Turn the three public datasets needed for step classification (`zhang_who`,
`uwash`, `ablutomania` — D8, D14) into one queryable, validated, canonical corpus. Nothing downstream is trustworthy until this is right, so do not start
M2 early.

**Deliverables**
- One ingest adapter per M1 dataset (`zhang_who`, `uwash`, `ablutomania`).
- Per-dataset axis-frame table (axis order, signs, wrist) in `docs/label_mapping.md`,
  mapped to the Android sensor frame at ingest (D10).
- `docs/label_mapping.md` — explicit source-label → canonical-label table with rationale
  for every mapping, including every deliberate drop.
- `haptic_ai/preprocess.py`, applied in this order (§8.7 defines each step exactly):
  1. resample to 50 Hz with `scipy.signal.resample_poly` (anti-aliased; never plain
     decimation),
  2. unit normalization (m/s², rad/s),
  3. gravity-alignment rotation (needs gravity, so it runs **before** the bandpass),
  4. Butterworth bandpass 1–18 Hz (order 3), **causal**,
  5. left/right wrist mirroring augmentation.
- `reports/M1_corpus.json` + `reports/M1_corpus.md`: per-dataset and per-subject sample
  counts, per-label duration, wrist distribution, sample-rate jitter, gap statistics.
- Corpus written to `data/processed/` in Parquet, partitioned by `subject_id`.

**Acceptance criteria**
- Every row in `data/processed/` passes `schema.validate`.
- Zero duplicate `(subject_id, session_id, timestamp_ns)` triples.
- Resampled series have no gap > 100 ms; gaps above that are recorded as session breaks,
  not interpolated.
- Every subject id is globally unique across datasets (prefix with the dataset key).
- Gravity-alignment unit test: a synthetic signal rotated by a known quaternion recovers
  the original within 1e-3.
- Mirroring unit test: mirroring twice is the identity.
- Causality unit test: changing samples after index *t* does not change the filtered
  output at or before *t*.
- Window-label unit test covering the purity rule in §8.7.

**Verification**
- `make ingest && make report-corpus`
- `pytest tests/test_preprocess.py tests/test_ingest_*.py`
- Human review of `reports/M1_corpus.md` — plots of at least three washes per dataset,
  eyeballed against the descriptions in the source papers.

---

### M2 — Step classifier + honest evaluation

**Objective.** Produce a shippable step classifier and, more importantly, a defensible
number for how well it generalizes to people it has never seen.

**Deliverables**
- `haptic_ai/features.py` — time-domain (mean, std, min, max, RMS, energy, zero-crossing
  rate, correlation between axes) and frequency-domain (dominant frequency, spectral
  entropy, band power in 1–3 / 3–6 / 6–12 Hz) features, per axis, per window.
- Baseline model: Random Forest and Gradient Boosting on those features.
- Comparison model: small 1D CNN.
- `haptic_ai/evaluate.py` implementing **leave-one-subject-out cross-validation**, with a
  dummy classifier baseline, per-subject macro-F1, mean ± std, and confusion matrices.
- Causal post-processing: moving average over the last *k* window predictions
  (start with k=5); report metrics with and without it.
- Window-size sweep: {2 s, 3 s, 5 s} at 50 % overlap. Record the winner and why.
- `haptic_ai/export.py` — int8 quantization, TFLite export, `model_meta.json` emission.
- Release `v0.1.0` of `HapticWash-AI` with `model.tflite` + `model_meta.json`.
- `golden/inputs.npy`, `golden/outputs.npy`, `golden/raw.npy` exported with the release,
  per §9.2 (the app's parity tests consume them).
- `reports/M2_step_classifier.md` with all of the above plus per-class F1.

**Acceptance criteria**
- LOSO macro-F1 on labels 1–5, **target ≥ 0.65**, stretch ≥ 0.75. If below target, the
  milestone still passes provided the report documents the gap, the per-class breakdown,
  and at least three tried-and-rejected mitigations. **Reporting a real low number is a
  pass; reporting an inflated one is a failure.**
- Metrics are reported per subject with standard deviation, never as a single pooled
  accuracy.
- The dummy-classifier baseline is reported alongside every headline metric.
- No subject appears in both train and test folds. Enforced by an automated assertion,
  not by inspection.
- Quantized model within the §4.3 size budget; post-quantization macro-F1 drop ≤ 0.02
  versus float.
- `model_meta.json` validates against the schema and its `labels` array exactly matches
  §1.3 ordering.

**Verification**
- `make train-step && make eval-step`
- `pytest tests/test_evaluate.py` — includes a test that deliberately leaks a subject
  across folds and asserts the guard raises.
- `pytest tests/test_export.py` — asserts tensor shapes, dtype, size budget, and that
  reference inputs produce outputs matching the pre-export model within tolerance.

---

### M5-AI — Spotting model *(optional for v1)*

**Objective.** Train a binary wash-vs-NULL model with a false-positive rate low enough to
not be annoying. The on-watch runtime (gating, duty-cycling, toggle) is M5-App.

Calibrate expectations before starting: on realistic all-day data, published work reports
a maximum per-subject F1 around 0.77 with a mean near 0.33 against a chance level of
roughly 0.03. A mean F1 of 0.4 here is a genuinely good result, not a failure.

**Deliverables**
- Binary wash-vs-NULL model trained primarily on `ocdetect`, with `ablutomania`
  confounders, 5 s windows, 50 % overlap.
- Explicit class-imbalance handling: compare random undersampling of the majority class,
  SMOTE, and class-weighted loss. Report all three.
- Threshold selection tuned for **precision**, not F1, since a false positive interrupts
  the user.
- Confounder analysis table: which non-wash activities trigger false positives, ranked.
- `ocdetect` ingest adapter (moved here from M1 — D8); `ablutomania` `longterm-2020.zip`
  fetched only now.
- Spotting model exported with its own `model_meta.json` (`task: "spotting"`).

**Acceptance criteria**
- LOSO median per-subject F1 ≥ 0.40 **and** ≥ 3× the dummy-classifier chance level.
- Estimated false positives ≤ 1 per 8 hours of wear on held-out `ocdetect` subjects.
- Confounder table committed with at least the top 10 false-positive-inducing contexts.

**Verification**
- `make train-spotting && make eval-spotting`

---

## 9. Verification

### 9.4 Evaluation integrity rules

- Any metric quoted anywhere (README, report, slides, `model_meta.json`) must trace to a
  committed run report containing its config and git SHA.
- Pooled accuracy on imbalanced data is banned as a headline metric.
- Every headline metric is accompanied by the dummy-classifier chance level.
- `own_watch` and any borrowed-device data are evaluation-only, enforced in code.
- Phone-collected (`own_phone`) data may be used for training and augmentation, never for
  reported evaluation. Also enforced in code.

---
