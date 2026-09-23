# HapticWash — Engineering Specification

**Status:** v1.0 (planning)
**Owner:** Camilo Parra S.
**Repos:** `HapticWash` (Wear OS app) · `HapticWash-AI` (data + model pipeline)
**Audience:** human developer + AI coding agents

> **Agents: read `AGENT RULES` (§10) before making any change.** This document is the
> source of truth. If an instruction elsewhere conflicts with this file, this file wins.
> If this file is wrong or ambiguous, open an issue instead of guessing.

---

## 1. Project Overview

### 1.1 What it is

HapticWash is an on-device Wear OS application that observes wrist motion during
handwashing and reports which WHO handwashing gestures appear to have been performed,
which appear to have been missed, and keeps a local history of recent washes.

It is a **hygiene coach**, not a medical device and not a compliance-verification system.
No output may be phrased as a guarantee that hands are clean.

### 1.2 Problem decomposition

The project contains two distinct machine learning problems. Conflating them is the
main failure mode of prior art in this space.

| | **Spotting** | **Step classification** |
|---|---|---|
| Question | Is a handwash happening at all? | Which gesture is this window? |
| Class balance | ~1:99 (wash vs. all daily activity) | roughly balanced within a wash |
| Difficulty | Hard. Confounded by dishwashing, tooth brushing, laundry, kneading, hair brushing | Moderate |
| Required for | M5 (always-on mode) | M2 (core feature) |

**Consequence for scope:** v1 ships **manual mode** — the user taps "Start wash" and the
step classifier runs for the duration of the session. Always-on spotting is added in M5
behind a settings toggle and is explicitly labelled experimental. Do not block v1 on
spotting.

### 1.3 Gesture label set (canonical)

| id | name | user-facing label |
|---|---|---|
| 0 | `NULL` | (not shown) |
| 1 | `PALM_TO_PALM` | Palms |
| 2 | `BACK_OF_HAND` | Back of hands |
| 3 | `INTERLACED_FINGERS` | Between fingers |
| 4 | `THUMBS` | Thumbs |
| 5 | `FINGERTIPS` | Fingertips |
| 6 | `WASH_OTHER` | Washing |

`WASH_OTHER` covers wetting, soaping, rinsing, and any wash motion not matching 1–5. It
exists so the classifier is never forced to assign a WHO step to a non-step moment. It is
counted as "wash in progress" but contributes to no step's coverage.

**This ordering is frozen.** It defines the output tensor index order in §8.3.

### 1.4 Non-goals

- Apple watchOS support.
- Cloud sync, accounts, multi-user, or any server component.
- Camera, microphone, or audio sensing of any kind.
- Real-time step-by-step *instruction* ("now do your thumbs"). v1 reports coverage
  **after** the wash. Live prompting changes user behaviour and invalidates the model's
  training distribution.
- Any LLM inference on the watch.

---

## 2. Tech Stack

### 2.1 `HapticWash` — Wear OS app

| Concern | Choice | Notes |
|---|---|---|
| Language | Kotlin | JVM target 17 |
| Min SDK | 30 (Wear OS 3) | Target latest stable |
| UI | Jetpack Compose for Wear OS (`androidx.wear.compose`) | Not phone Compose Material |
| Lists | `ScalingLazyColumn` / `TransformingLazyColumn` | Never plain `LazyColumn` |
| Async | Coroutines + `Flow` | |
| Sensors | `SensorManager`, `TYPE_ACCELEROMETER` + `TYPE_GYROSCOPE` | 50 Hz, see §8.1 |
| Background | Foreground `Service`, type `dataSync` | Notification required |
| Inference | LiteRT (TensorFlow Lite) Android runtime | CPU only, pin exact version |
| Storage | Room (sessions + step coverage) | Local only |
| Prefs | DataStore (Preferences) | wrist side, toggles |
| DI | Hilt (optional; manual DI acceptable if it keeps the graph small) | |
| Tests | JUnit5 + Turbine (unit), AndroidX Test + Espresso (instrumented) | |

**Permissions:** `FOREGROUND_SERVICE`, `FOREGROUND_SERVICE_DATA_SYNC`,
`POST_NOTIFICATIONS`, `VIBRATE`, `WAKE_LOCK` (partial wake lock held only for the
duration of an active session, so sensor delivery continues with the screen off;
confirm it is needed on the emulator at M3 — D12).
Accelerometer and gyroscope at 50 Hz require **no** runtime permission.
`HIGH_SAMPLING_RATE_SENSORS` is **not** needed (only applies above 200 Hz).
Do **not** request `BODY_SENSORS`, `ACTIVITY_RECOGNITION`, `INTERNET`, or location.
`INTERNET` absence is a hard architectural guarantee — see §4.2.

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

### 2.3 The "AI layer", concretely

There are three tiers, and they must not be confused.

**Tier 1 — On-watch inference (required, offline, no LLM).**
A quantized int8 LiteRT classifier. Budget: **≤ 150 KB** model file, **≤ 15 ms** per
window on a mid-range watch CPU, single-threaded. Input is a fixed-shape window of
filtered IMU data or derived features; output is a probability vector over the 7 labels
in §1.3. This is the only model that ships.
<!-- 
**Tier 2 — Development-time agents (LLM, online, not shipped).**
Coding agents (Claude Code or similar) writing code against this spec. They touch source,
never runtime. Their output is reviewed like any other contribution. -->

**Tier 3 — Optional phone-side natural-language summaries (LLM, online, stretch goal, M6+).**
A weekly "here's how your handwashing looked" paragraph, generated on the paired phone by
sending **aggregate statistics only** (counts, mean durations, per-step coverage
percentages) to an API. Governed by §4.2. Off by default, requires explicit opt-in, and
the app must be fully functional with it permanently disabled. **Do not implement before
M6.** No raw IMU data ever leaves the device, under any circumstance.

---

## 3. Repository Layout

```
HapticWash/                          # Wear OS app
├── wear/src/main/java/com/hapticwash/
│   ├── sensing/                     # MotionSource interface, Real + Replay impls
│   ├── pipeline/                    # filtering, windowing, normalization
│   ├── inference/                   # LiteRT wrapper, model_meta parsing
│   ├── session/                     # coverage state machine, session lifecycle
│   ├── data/                        # Room entities, DAOs, repository
│   └── ui/                          # Compose screens
├── wear/src/main/assets/model/      # model.tflite + model_meta.json (from AI repo release)
├── wear/src/androidTest/            # instrumented: parity tests, replay E2E
├── wear/src/test/                   # unit: state machine, windowing, filters
└── SPEC.md                          # this file (symlink or copy; keep in sync)

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

---

## 4. Architectural Constraints

### 4.1 Wear OS UI/UX constraints

These are non-negotiable and apply to every screen.

1. **Glanceable.** Any screen must be comprehensible in under 5 seconds. If a screen
   needs a paragraph, it is the wrong screen.
2. **Round-safe.** Assume a circular display. No content in the corners. Use
   `ScalingLazyColumn` so first and last items scale toward the bezel. Always include
   `TimeText` at the top of a scrollable screen.
3. **Touch targets ≥ 48dp.** Primary action per screen: exactly one.
4. **One-tap start.** Starting a wash must be a single tap from the app's main screen.
5. **No text entry, no dialogs with more than two options, no nested navigation deeper
   than two levels.**
6. **Haptics over sound.** Feedback is vibration. Use short, distinct patterns:
   session start (one short pulse), session end (two short pulses), missed-step
   notice (one long pulse). Never vibrate more than once per 3 seconds.
7. **Ambient mode.** During an active wash the app must handle ambient/always-on
   transitions without stopping sensor collection or crashing.
8. **Battery honesty.** Continuous 50 Hz accelerometer + gyroscope is expensive. Manual
   mode must stop the sensor listener the moment a session ends. Always-on mode (M5)
   must show an explicit battery warning at enable time.
9. **No result is presented as certainty.** Coverage UI language is "looks like",
   "may have been brief", "not detected". Never "you missed" / "you failed" / "clean".
10. **Accessibility.** All interactive elements carry `contentDescription`. Do not encode
    step status by colour alone; pair colour with an icon or text.

### 4.2 Offline / online AI execution policy

**Hard rules, enforceable by test.**

- **R1.** The Wear OS app declares no `INTERNET` permission. An instrumented test asserts
  this by reading the merged manifest. Any PR that adds it fails review automatically.
- **R2.** All inference required for the core feature runs on-watch, offline, in-process,
  on CPU. No dependency on a phone connection, network, or companion app.
- **R3.** Raw or windowed IMU samples never leave the watch. Not to a phone, not to a
  server, not to logs shipped off-device.
- **R4.** Tier 3 summaries (§2.3), if ever built, live in a **separate phone module**,
  are opt-in, transmit only aggregate counts and durations, and cache their last result
  so the feature is inert rather than broken when offline.
- **R5.** No model weights are downloaded at runtime. The model is bundled in `assets/`
  and versioned with the app build.
- **R6.** If model loading fails for any reason, the app degrades to a **timer-only
  mode** (count the wash, show duration, show no step coverage) and displays a single
  non-blocking notice. It must never crash and never show fabricated coverage.

### 4.3 Performance budgets

| Metric | Budget | Verified in |
|---|---|---|
| Model file size | ≤ 150 KB | M2 |
| Inference latency, one window | ≤ 15 ms | M3 |
| End-to-end window latency (sample → coverage update) | ≤ 250 ms | M3 |
| App cold start to "Start wash" tappable | ≤ 1.5 s | M4 |
| Sustained heap during a session | ≤ 40 MB | M4 |
| Battery drain, manual mode, 10 washes/day | ≤ 3 %/day | M6 (borrowed device) |

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
| `zhang_who` | Zhang et al., "Replication Data for: handwashing steps with IMU signals", KU Leuven RDR, DOI `10.48804/XHPPC7` | **Primary** — the only public set with WHO step labels. Trains the step classifier. |
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

Each milestone lists **Objective → Deliverables → Acceptance criteria → Verification**.
A milestone is not done until every acceptance criterion has a passing automated check or
a committed report. Milestones are sequential; M5 may be dropped without invalidating v1.

---

### M0 — Bootstrap and contracts

**Objective.** Freeze the interfaces between the two repos before any modelling or UI work
begins, so the app and the pipeline can proceed in parallel without drift.

**Deliverables**
- Both repos initialised: build files, linting (ktlint / ruff), CI running tests on push.
- `haptic_ai/schema.py` implementing §8.1 with a `validate(df)` function.
- `docs/DATASETS.md` with per-dataset licence and citation.
- A **stub** `model.tflite` (random weights, correct shapes) plus a valid
  `model_meta.json`, published as release `v0.0.1-stub` of `HapticWash-AI`.
- `docs/DECISIONS.md` started.

**Acceptance criteria**
- `make validate-schema` passes on a hand-written 100-row fixture.
- The stub release artifacts load in a Python test asserting shapes match §8.3.
- CI green in both repos.

**Verification**
- `pytest tests/test_schema.py`
- Manual: confirm the stub `model_meta.json` validates against `docs/model_meta.schema.json`.

---

### M1 — Unified corpus

**Objective.** Turn the two public datasets needed for step classification (`zhang_who`,
`ablutomania` — D8) into one queryable, validated, canonical corpus. Nothing downstream is trustworthy until this is right, so do not start
M2 early.

**Deliverables**
- One ingest adapter per M1 dataset (`zhang_who`, `ablutomania`).
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

### M3 — App skeleton, replay harness, parity

**Objective.** Get the full production pipeline running end-to-end on an emulator against
real, labelled human data — and prove the model deployed on the device computes the same
thing as the model evaluated in Python.

This milestone is the answer to "I don't own a Wear OS device." The Wear OS emulator can
only simulate accelerometer and gyroscope through Extended Controls → Virtual Sensors →
Device Pose sliders, which cannot produce a realistic 50 Hz handwash trace. The replay
harness replaces it.

**Deliverables**
- `MotionSource` interface (§8.4) with `RealSensorSource` and `ReplaySource`.
  Everything downstream is source-agnostic.
- `ReplaySource` reads a canonical CSV pushed via `adb push` and emits samples honouring
  original inter-sample timing, with an optional speed multiplier for tests.
- Broadcast-intent trigger for replay (§8.5) so replay is demoable from the command line.
- Kotlin implementations of filtering, windowing, normalization, and feature extraction
  that mirror `HapticWash-AI` exactly.
- LiteRT inference wrapper reading `model_meta.json` at load time. Hardcoded shapes,
  label orders, or normalization constants in Kotlin are a **spec violation**.
- Foreground service collecting at 50 Hz.
- R8 keep rule for LiteRT in the release build (`-keep class org.tensorflow.** { *; }`);
  a release-build smoke test loads the model.
- **Golden-file parity test** (§9.2).
- Minimal UI: start / stop / raw predicted label. No polish yet.

**Acceptance criteria**
- `adb`-triggered replay of a held-out session from `zhang_who` runs end to end on the
  emulator and produces a per-window prediction stream.
- Parity: for ≥ 50 golden windows, on-device probability vectors match Python outputs
  within **1e-2 absolute** per class, and top-1 labels match on **≥ 98 %** of windows.
- Measured inference latency ≤ 15 ms/window (report the median and p95 from the emulator;
  treat as indicative, confirm on hardware at M6).
- Manifest test asserting `INTERNET` is absent (R1).
- Model-load-failure test: with a corrupt `model.tflite`, the app enters timer-only mode
  and does not crash (R6).

**Verification**
- `./gradlew :wear:testDebugUnitTest`
- `./gradlew :wear:connectedDebugAndroidTest` on a Wear OS AVD
- `scripts/replay_demo.sh <session.csv>` — documented, one command, works from a clean
  checkout.

---

### M4 — Coverage state machine, feedback, history

**Objective.** Turn a stream of window predictions into something a person can act on.

**Deliverables**
- `CoverageStateMachine` (§8.6): consumes smoothed predictions, accumulates seconds per
  step, emits per-step status `NOT_DETECTED | BRIEF | OK`.
- Thresholds in a single configuration object, not scattered constants. Defaults:
  `OK` ≥ 4 s cumulative, `BRIEF` ≥ 1.5 s, otherwise `NOT_DETECTED`; minimum session
  duration 10 s for coverage to be shown at all.
- Compose screens: Start, In-progress (elapsed time + subtle progress), Summary
  (per-step status), History (last 20 washes), Settings (wrist side, haptics, always-on
  toggle placeholder).
- Room persistence of sessions and per-step durations.
- Haptic patterns per §4.1 item 6.
- Wrist-side setting wired through to the mirroring transform.

**Acceptance criteria**
- State machine unit tests covering: all steps covered; one step entirely missing; a step
  detected in two non-contiguous bursts summing above threshold; a 4-second session
  (must show no coverage); a session with 40 % `WASH_OTHER`; an empty prediction stream.
- Replaying a labelled session yields step statuses consistent with its ground-truth step
  durations for ≥ 4 of 5 steps, on ≥ 8 of 10 held-out sessions.
- History survives process death and device reboot.
- UI audit against every item in §4.1, recorded as a checklist in `docs/UX_AUDIT.md`
  with a screenshot of each screen on a round AVD.
- No user-facing string asserts cleanliness or uses failure language. Enforced by a lint
  test over the strings resource against a banned-word list.

**Verification**
- `./gradlew :wear:testDebugUnitTest --tests '*CoverageStateMachine*'`
- `scripts/replay_batch.sh` over 10 held-out sessions, producing a comparison report.
- Manual UX audit checklist.

---

### M5 — Always-on spotting *(optional for v1)*

**Objective.** Detect that a wash is happening without the user tapping anything, at a
false-positive rate low enough to not be annoying.

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
- Second model in the app assets; gated inference (spotting model runs continuously at
  low duty cycle; step model runs only after spotting fires).
- Duty-cycling strategy documented and implemented (e.g. spotting inference on 1 of every
  N windows until a positive, then dense).
- Battery-impact estimate.

**Acceptance criteria**
- LOSO median per-subject F1 ≥ 0.40 **and** ≥ 3× the dummy-classifier chance level.
- Estimated false positives ≤ 1 per 8 hours of wear on held-out `ocdetect` subjects.
- Confounder table committed with at least the top 10 false-positive-inducing contexts.
- Feature toggle defaults **off**, with the §4.1 item 8 battery warning at enable time.

**Verification**
- `make train-spotting && make eval-spotting`
- Replay of a full held-out `ocdetect` day through the app; count spurious session starts.

---

### M6 — Hardware validation and limitations report

**Objective.** Close the loop on real hardware, and document precisely what remains
unverified. This section is what distinguishes the project from a hackathon build.

**Deliverables**
- A borrowed Wear OS device session (target ≥ 2 hours): 10–20 self-labelled washes plus
  ordinary background activity, recorded through the app's own collection path.
- `own_watch` recordings kept as a **held-out test set**, never used for training or model
  selection. Enforce with a guard in the data loader that refuses to load `own_watch`
  outside evaluation mode.
- Measured on-hardware inference latency, cold start, and battery drain against §4.3.
- `docs/LIMITATIONS.md` — a table of every claim in the README with its evidence tier:
  *verified on hardware* / *verified by replay of public data* / *verified by unit test* /
  *unverified*.
- README rewritten with results, honest metrics, dataset citations, and a demo GIF of the
  replay harness.

**Acceptance criteria**
- Held-out on-watch step macro-F1 reported. **Any** number is acceptable; omitting it is
  not.
- Every §4.3 budget either met or documented with the measured value and a note.
- `docs/LIMITATIONS.md` contains no claim tagged "verified on hardware" that lacks a
  corresponding measurement in `reports/`.

**Verification**
- `make eval-holdout --dataset own_watch`
- Human review of `docs/LIMITATIONS.md` against `reports/`.

---

## 7. Milestone Dependency Graph

```
M0 ──> M1 ──> M2 ──> M3 ──> M4 ──> M6
                            │
                            └──> M5 (optional, before M6)
```

M3 may begin in parallel with M2 using the M0 stub model, since the stub has correct
shapes. This is the recommended parallelization for agent work.

---

## 8. API and Data Contracts

Changing anything in this section is a breaking change requiring a version bump in both
repos and an entry in `docs/DECISIONS.md`.

### 8.1 Canonical sensor CSV schema

Used for processed data, replay files, and on-device export.

| column | type | unit | notes |
|---|---|---|---|
| `timestamp_ns` | int64 | ns | monotonic since session start, not epoch |
| `ax`,`ay`,`az` | float32 | m/s² | includes gravity; raw accelerometer |
| `gx`,`gy`,`gz` | float32 | rad/s | gyroscope |
| `label` | int8 | — | canonical id per §1.3; `-1` = unlabelled |
| `subject_id` | string | — | `<dataset_key>_<local_id>`, globally unique |
| `session_id` | string | — | unique within subject |
| `wrist` | string | — | `left` \| `right` \| `unknown` |
| `source` | string | — | dataset key per §5 |

Rules: 50 Hz nominal. Header row required. UTF-8, LF line endings, `.` decimal separator.
Column order is fixed as listed. A file failing `schema.validate` is rejected, never
coerced.

### 8.2 `model_meta.json`

```json
{
  "spec_version": "1.0",
  "model_id": "step-classifier",
  "model_version": "0.1.0",
  "task": "step_classification",
  "sample_rate_hz": 50,
  "window_size_s": 3.0,
  "window_stride_s": 1.5,
  "channels": ["ax", "ay", "az", "gx", "gy", "gz"],
  "preprocessing": {
    "bandpass": { "low_hz": 1.0, "high_hz": 18.0, "order": 3 },
    "gravity_align": true,
    "gravity_lowpass_hz": 0.3,
    "normalization": "zscore_per_channel",
    "norm_mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "norm_std":  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
  },
  "input": { "shape": [1, 150, 6], "dtype": "float32" },
  "output": { "shape": [1, 7], "dtype": "float32", "activation": "softmax" },
  "labels": ["NULL","PALM_TO_PALM","BACK_OF_HAND","INTERLACED_FINGERS","THUMBS","FINGERTIPS","WASH_OTHER"],
  "postprocessing": { "moving_average_k": 5 },
  "training": {
    "datasets": ["zhang_who", "ablutomania"],
    "loso_macro_f1": 0.00,
    "loso_macro_f1_std": 0.00,
    "commit": "<git sha>"
  }
}
```

The app **must** read `sample_rate_hz`, `window_size_s`, `window_stride_s`, all of
`preprocessing`, `labels`, and `postprocessing` from this file at load time. If
`spec_version` is unrecognized, the app enters timer-only mode (R6) rather than guessing.

### 8.3 Model tensor contract

- Input: `float32[1, W, 6]` where `W = window_size_s * sample_rate_hz`, channels ordered
  as in `channels`, already bandpassed, gravity-aligned, and z-scored using `norm_mean` /
  `norm_std`.
- Output: `float32[1, 7]`, softmax, index order **exactly** the `labels` array, which
  matches §1.3.
- Quantization: int8 weights with float32 in/out (dynamic-range or full-integer with
  float wrappers). The Kotlin side never handles quantization parameters directly.

### 8.4 `MotionSource`

```kotlin
data class MotionSample(
    val timestampNs: Long,
    val ax: Float, val ay: Float, val az: Float,
    val gx: Float, val gy: Float, val gz: Float,
)

interface MotionSource {
    /** Cold flow. Collecting starts acquisition; cancelling stops it. */
    fun stream(): Flow<MotionSample>
    val nominalRateHz: Int
}
```

`RealSensorSource` registers accelerometer and gyroscope at `samplingPeriodUs = 20_000`
and pairs them by nearest timestamp, dropping unpaired samples. `ReplaySource` reads a
canonical CSV and emits with original inter-sample delays, scaled by `speed`.

No code outside `sensing/` may reference `SensorManager` or `SensorEvent`.

### 8.5 Replay trigger intent

```
adb shell am broadcast \
  -a com.hapticwash.REPLAY \
  --es path /sdcard/Download/session_042.csv \
  --ef speed 1.0 \
  com.hapticwash/.sensing.ReplayReceiver
```

Registered in debug builds only. Must not be present in the release manifest; a manifest
test asserts this.

### 8.6 Session and coverage contract

```kotlin
enum class StepStatus { NOT_DETECTED, BRIEF, OK }

data class StepCoverage(val label: Int, val seconds: Float, val status: StepStatus)

data class WashSession(
    val id: Long,
    val startedAtEpochMs: Long,
    val durationS: Float,
    val coverage: List<StepCoverage>,   // labels 1..5, always length 5, always in §1.3 order
    val washOtherS: Float,
    val modelVersion: String,           // from model_meta.json
    val source: String,                 // "sensor" | "replay"
)
```

`coverage` is always length 5 in fixed order, even when a step has zero seconds. The UI
never has to handle a missing entry. `source` is persisted so replay-generated history is
distinguishable from real washes.

### 8.7 Preprocessing and windowing definitions

Both repos implement exactly this; the golden-file preprocessing parity test (§9.2 item
4) is the check.

- **Streaming, causal.** Every filter runs over the continuous session stream, carrying
  its state across windows. Windows are cut from the already-filtered stream; nothing is
  filtered per window. Python uses `scipy.signal.sosfilt` (with `butter(..., output="sos")`),
  never `filtfilt`/`sosfiltfilt`, because the watch can only filter causally. Filter
  state starts at zero at session start (`sosfilt_zi` is not used).
- **Gravity alignment.** Gravity `g` is estimated per sample by a causal 2nd-order
  Butterworth low-pass at `gravity_lowpass_hz` on the raw accelerometer. Each sample's
  accelerometer and gyroscope vectors are rotated by the minimal rotation (Rodrigues)
  taking `g/|g|` to `+z`. Heading (rotation about `z`) is left free.
- **Order.** resample → units → gravity-align → bandpass → z-score (`norm_mean`,
  `norm_std`).
- **Window labels.** A window takes the majority label of its samples if that label
  covers ≥ 60 % of them; otherwise it is labelled `-1` and excluded from training and
  evaluation.

---

## 9. Verification and Testing Criteria

### 9.1 Global gates (every PR)

- Unit tests pass in both repos.
- Linters clean.
- No new dependency without an entry in `docs/DECISIONS.md`.
- Manifest assertions (no `INTERNET`; no replay receiver in release).
- Banned-string lint on user-facing text (§4.1 item 9).
- Reproducibility: any changed training config reruns to the same metrics within ±0.005.

### 9.2 Golden-file parity test (the central correctness check)

The only way to know the shipped model behaves like the evaluated model without owning
a device.

1. `HapticWash-AI` exports `golden/inputs.npy` (≥ 50 preprocessed windows drawn from
   held-out subjects) and `golden/outputs.npy` (float model probabilities), committed to
   the release alongside the model.
2. `HapticWash` includes both as test assets.
3. An instrumented test runs each window through the on-device LiteRT interpreter and
   asserts per-class absolute difference ≤ 1e-2 and top-1 agreement ≥ 98 %.
4. Additionally, a **preprocessing** parity test: raw windows from `golden/raw.npy` are
   run through the Kotlin filter/align/normalize chain and compared to Python's output,
   tolerance 1e-3. This catches the more likely bug — divergent preprocessing, not
   divergent inference.

If parity fails, the Kotlin side is presumed wrong until proven otherwise.

### 9.3 Replay end-to-end test

For each of 10 held-out labelled sessions: replay through the full app pipeline, write
the resulting `WashSession` to a JSON report, and compare per-step seconds against ground
truth. Committed to `reports/` as a table. This is the primary demo artifact and should be
the GIF in the README.

### 9.4 Evaluation integrity rules

- Any metric quoted anywhere (README, report, slides, `model_meta.json`) must trace to a
  committed run report containing its config and git SHA.
- Pooled accuracy on imbalanced data is banned as a headline metric.
- Every headline metric is accompanied by the dummy-classifier chance level.
- `own_watch` and any borrowed-device data are evaluation-only, enforced in code.
- Phone-collected (`own_phone`) data may be used for training and augmentation, never for
  reported evaluation. Also enforced in code.

---

## 10. AGENT RULES

1. **Do not invent data.** If a dataset field, label, or mapping is unclear, stop and open
   an issue. Never fabricate a plausible-looking mapping, metric, or citation.
2. **Do not report a metric you did not compute.** Placeholders must be literal `0.00`
   with a `TODO`, never a realistic-looking guess.
3. **Do not add network calls to the app.** See §4.2 R1–R5. This includes analytics,
   crash reporting, and remote config.
4. **Do not hardcode values that belong in `model_meta.json`.** Window size, label order,
   normalization constants, and sample rate are all read at runtime.
5. **Do not change §1.3, §8.1, §8.2, or §8.3** without an explicit instruction and a
   `docs/DECISIONS.md` entry.
6. **Keep the two feature implementations in lockstep.** Any change to
   `haptic_ai/features.py` requires the matching Kotlin change plus a regenerated golden
   file in the same PR.
7. **Prefer the simple model.** Do not introduce a transformer, an attention mechanism, or
   a pretrained backbone before the tree baseline is measured and documented.
8. **Do not weaken a test to make it pass.** If an acceptance threshold is unreachable,
   report the real number and document why; that is a passing outcome under §6 M2.
9. **One milestone per branch.** Branch naming: `m2/step-classifier`.
10. **Cite your sources in code comments** when implementing something from a paper,
    including the paper and section.

---

## 11. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Step classification does not generalize across people | High | Honest LOSO reporting; conservative coverage thresholds; `WASH_OTHER` escape hatch; frame output as estimates |
| `zhang_who` step labels do not map cleanly to §1.3 | Medium | Explicit mapping doc; map ambiguous cases to `WASH_OTHER`; consider collapsing to 3 coarse steps as a documented fallback |
| No device ever becomes available | Medium | Replay harness + parity tests carry the project; `docs/LIMITATIONS.md` states the gap plainly. Do not fake hardware results. |
| Dataset licence prohibits intended use | Medium | Check at M0, before any modelling work depends on it |
| Always-on drains battery unacceptably | High | M5 is optional, off by default, duty-cycled, warned |
| Scope creep into watchOS, cloud, or LLM features | Medium | §1.4 non-goals; §4.2 hard rules |

---

## 12. Open Questions

- [ ] Confirm the exact licence and citation requirement for each dataset in §5.
- [ ] Decide PyTorch vs. Keras at M2 and record it.
- [ ] Confirm `zhang_who` step label granularity matches §1.3 or requires collapsing.
- [ ] Identify a specific borrowable Wear OS device and target date for M6.
- [ ] Decide whether Tier 3 phone summaries are in scope at all.
