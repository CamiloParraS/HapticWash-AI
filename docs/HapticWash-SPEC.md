# HapticWash — Core Specification (shared contracts)

**Status:** v1.0 (planning)
**Owner:** Camilo Parra S.
**Audience:** human developer + AI coding agents

> **HapticWash is two repos with one contract between them.** This file holds only what
> both sides must agree on. Everything else lives in the side-specific spec:
>
> | Spec | Location | Owns |
> |---|---|---|
> | **Core** (this file) | `HapticWash-AI/docs/HapticWash-SPEC.md` | What the product is, label set, data/model contracts, parity, M0/M6, rules, risks |
> | **AI** | [`HapticWash-AI/docs/SPEC-AI.md`](SPEC-AI.md) | Datasets, preprocessing, training, evaluation, export (M1, M2, M5-AI) |
> | **App** | `HapticWash - Main/docs/SPEC-APP.md` | Wear OS app: sensing, on-device inference, UI, storage, privacy (M3, M4, M5-App) |
>
> Section numbers (§) are **global** and unchanged from the original single spec, so
> `SPEC 8.1` in code and `docs/DECISIONS.md` still points to exactly one place. The
> section index below says which file each § lives in.
>
> **Agents: read `AGENT RULES` (§10) before making any change.** If specs conflict, this
> file wins. If a spec is wrong or ambiguous, open an issue instead of guessing.

---

## Section index

| § | Topic | File |
|---|---|---|
| 1 | Project overview, label set, non-goals | Core |
| 2.1 | App tech stack | App |
| 2.2 | AI pipeline tech stack | AI |
| 2.3 | The "AI layer" — where AI ends and the app begins | Core |
| 3 | Repository layout | Core (boundary) · AI · App |
| 4.1, 4.2 | Wear OS UX constraints; offline/online policy | App |
| 4.3 | Performance budgets | Core |
| 4.4 | Determinism | AI |
| 5 | Data sources | AI |
| 6 — M0, M6 | Bootstrap; hardware validation | Core |
| 6 — M1, M2, M5-AI | Corpus; step classifier; spotting model | AI |
| 6 — M3, M4, M5-App | App skeleton + parity; coverage/UI; always-on runtime | App |
| 7 | Milestone dependency graph | Core |
| 8.1, 8.2, 8.3, 8.7 | CSV schema, `model_meta.json`, tensor contract, preprocessing | Core |
| 8.4, 8.5, 8.6 | `MotionSource`, replay intent, session/coverage | App |
| 9.1, 9.2 | Global gates; golden-file parity | Core |
| 9.3 | Replay end-to-end test | App |
| 9.4 | Evaluation integrity rules | AI |
| 10, 11, 12 | Agent rules, risks, open questions | Core |

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

§2.1 (app) is in the App spec, §2.2 (pipeline) in the AI spec.

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
HapticWash/                   # workspace folder, not a repo
├── HapticWash - Main/        # APP — Kotlin, Wear OS (+ phone module)  → docs/SPEC-APP.md
└── HapticWash-AI/            # AI  — Python, data + model pipeline     → docs/SPEC-AI.md
```

**The only things that cross the boundary** are `HapticWash-AI` release artifacts, copied
into the app's `wear/src/main/assets/model/` and test assets:

| Artifact | Produced by | Consumed by | Contract |
|---|---|---|---|
| `model.tflite` | AI (`export.py`) | App (LiteRT) | §8.3 |
| `model_meta.json` | AI (`export.py`) | App (read at load time) | §8.2 |
| `golden/inputs.npy`, `outputs.npy`, `raw.npy` | AI | App (parity tests) | §9.2 |
| Canonical CSV sessions | AI (`ingest/`) | App (`ReplaySource`) | §8.1 |

Nothing else is shared. The app never runs Python; the pipeline never reads Kotlin.
Per-repo layouts: §3 in each side spec.

---

## 4. Architectural Constraints

§4.1–4.2 are in the App spec, §4.4 in the AI spec.

### 4.3 Performance budgets

| Metric | Budget | Verified in |
|---|---|---|
| Model file size | ≤ 150 KB | M2 |
| Inference latency, one window | ≤ 15 ms | M3 |
| End-to-end window latency (sample → coverage update) | ≤ 250 ms | M3 |
| App cold start to "Start wash" tappable | ≤ 1.5 s | M4 |
| Sustained heap during a session | ≤ 40 MB | M4 |
| Battery drain, manual mode, 10 washes/day | ≤ 3 %/day | M6 (borrowed device) |


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

M1, M2, M5-AI → AI spec. M3, M4, M5-App → App spec.

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
 core │     AI       │     APP      │ core
──────┼──────────────┼──────────────┼──────
 M0 ──┼─> M1 ──> M2 ─┼─> M3 ──> M4 ─┼─> M6
      │              │         │    │
      │   M5-AI ─────┼──> M5-App    │   (M5 optional, before M6)
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

### 8.4 – 8.6

App-internal contracts (`MotionSource`, replay intent, session/coverage) — see the App spec.

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

### 9.3 – 9.4

Replay end-to-end test (§9.3) → App spec. Evaluation integrity rules (§9.4) → AI spec.

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
