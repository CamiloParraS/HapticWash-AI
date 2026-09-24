# Decisions

Append-only log of choices that constrain the project. Referenced by SPEC §9.1
(no new dependency without an entry here) and §8 (contract changes).

---

## D1 — 2026-09-09 — M0 bootstrap

Repo brought to SPEC §M0: canonical `haptic_ai/schema.py` + `validate()`, 100-row
fixture + `make validate-schema`, `docs/model_meta.schema.json`, stub-model generator
(`scripts/make_stub_model.py`), `docs/DATASETS.md`, this file, ruff config, GitHub
Actions CI.

## D2 — 2026-09-09 — TensorFlow is an optional, Linux-only extra

`tensorflow` moved from core `dependencies` to `[project.optional-dependencies].ml`
with marker `sys_platform == 'linux'`.

- **Why:** SPEC §2.2 defers the DL framework choice (PyTorch vs Keras) to M2, so TF is
  not a committed core dependency yet. Its transitive dep
  `tensorflow-io-gcs-filesystem` ships **no Windows wheels**, which made `uv sync` fail
  outright on the maintainer's Windows machine.
- **Consequence:** schema, ingest, preprocessing, and evaluation code must run without
  TF. Model export and the M0 stub model run on Linux — CI or WSL:
  `uv run --extra ml python scripts/make_stub_model.py`. The committed `.venv` was
  Linux-only and has been removed; `uv sync` rebuilds it per-platform.
- **Revisit at M2** when the framework is chosen (PyTorch via `ai-edge-torch` would
  drop the TF-on-Windows problem entirely).

## D3 — 2026-09-09 — Added `jsonschema` (dev)

For validating `model_meta.json` against `docs/model_meta.schema.json` (SPEC §M0
verification, §8.2). Dev-only, pure Python.

## D4 — 2026-09-09 — CLOSED — `zhang_who` licence is CC-BY-NC-ND-4.0

Verified on the KU Leuven RDR citation page. `zhang_who` is the **primary** dataset
(only public source with WHO step labels), so this gates the project.

- **NonCommercial:** acceptable while HapticWash is a non-commercial portfolio
  project; would be violated by a paid/monetised release.
- **NoDerivatives:** ambiguous for ML. Position taken pending review: model _weights_
  trained from the data are not treated as a derivative work; the processed corpus
  **is** a derivative and is never redistributed (SPEC §2.2/§5 already require this).
- **Action:** owner to confirm (a) the project stays non-commercial, or seek a
  commercial-use grant from the authors; (b) accept the ND interpretation above or get
  written permission. Record the outcome here. **Do not begin M2 work that depends on
  `zhang_who` until closed.**

### Decision

Proceed with `zhang_who` as the primary training dataset for the M2 step classifier, subject to strict architectural boundaries and compliance guardrails.

### Rationale

#### 1. NoDerivatives (ND) Compliance

- **No Data Redistribution:** CC BY-NC-ND 4.0 prohibits distributing modified versions of the _dataset itself_ (e.g., publishing re-sampled, filtered, or merged Parquet/CSV files derived from `zhang_who`). Per SPEC §5, raw and processed data are explicitly gitignored (`data/raw/` and `data/processed/`) and never committed, hosted, or redistributed in any repository.
- **Model Weights are Functional Artifacts:** Under standard open-source ML interpretation, quantized neural network parameters (`.tflite` weights) extracted via gradient descent represent learned statistical patterns and numerical functions, not copyrightable derivative works of the underlying raw IMU files.
- **Feature Processing Separation:** Code routines in `haptic_ai/preprocess.py` and `haptic_ai/features.py` are independent transformations executed locally by the end user via `scripts/fetch_data.py`, not distributed derivatives of the source files.

#### 2. NonCommercial (NC) Compliance

- **Scope Alignment:** HapticWash is strictly an open-source, non-commercial research and portfolio project with no paid tiers, ads, or monetization mechanisms (§1.1, §1.4).
- **Open Source License:** The HapticWash software codebase is licensed under a compatible open-source license (e.g., MIT/Apache-2.0), while acknowledging that the shipped default model weights in `v0.1.0` carry academic/non-commercial lineage derived from `zhang_who`.

* Licences for `ablutomania`, `ocdetect`, `harage` still TODO in `docs/DATASETS.md`.

## D5 — 2026-09-09 — `zhang_who` step granularity: collapse, don't expand (closes SPEC §12 Q3)

`zhang_who` PART 1 `FRAME/FrameACM_*.xls` has per-step annotation with `Action` codes
`0, 0.5, 1, 2.1, 2.2, 3, 4, 5.1, 5.2, 6.1, 6.2, 7` — **finer** than canonical §1.3,
not coarser. Provisional mapping (finalised with rationale in `docs/label_mapping.md`
at M1):

| zhang `Action`                                       | canonical §1.3           |
| ---------------------------------------------------- | ------------------------ |
| 0 (wet), 0.5 (soap), 4 (backs of fingers), 7 (rinse) | `WASH_OTHER` (6)         |
| 1                                                    | `PALM_TO_PALM` (1)       |
| 2.1, 2.2                                             | `BACK_OF_HAND` (2)       |
| 3                                                    | `INTERLACED_FINGERS` (3) |
| 5.1, 5.2                                             | `THUMBS` (4)             |
| 6.1, 6.2                                             | `FINGERTIPS` (5)         |

`Action 4` (WHO "backs of fingers to opposing palms, fingers interlocked") has no
canonical equivalent → `WASH_OTHER`, not invented into a step. The 3-coarse-step
fallback in SPEC §11 is not needed on granularity grounds.

## D6 — 2026-09-09 — `docs/` is lowercase

SPEC paths use `docs/`. The initial commit had `Docs/`; renamed. New docs land in
`docs/`.

## D7 — 2026-09-23 — Drop `harage` from the corpus

No public download (the paper says data is available from the authors on request), no
licence to verify, accelerometer only at 25 Hz (can't fill `gx,gy,gz` in §8.1), no step
labels, and washing was simulated without running water. Removed from §5. Revisit only by
emailing the authors if M2 is short of subjects.

## D8 — 2026-09-23 — M1 corpus = `zhang_who` + `ablutomania`; `ocdetect` moves to M5

`ocdetect` (31.6 GB, all-day, no step labels) only serves spotting, and nothing in M2 uses
it. M1 ingests `zhang_who` and `ablutomania` only. From `ablutomania`, fetch just
`handwashing-2019.zip`, `handwashing-2020.zip`, `hwseminar-2019.zip` (~4.4 GB); skip
`longterm-2020.zip` (35.7 GB) until M5. `ocdetect` ingest becomes an M5 deliverable.

## D9 — 2026-09-23 — Step labels come from 10 subjects; plan for it

`zhang_who` PART 1 is the only public source of WHO step labels: 10 participants, so LOSO
has 10 folds and per-subject macro-F1 will vary widely. The report shows every fold, not
just mean ± std.

Optional lever, highest-value data work available: `ablutomania` includes sink video, so
its *natural* washes could be hand-annotated with WHO steps, adding up to 22 subjects.
Needs an annotation protocol written up in `docs/label_mapping.md` before any labelling.
Owner to decide at M2 if the 10-subject result falls below target.

## D10 — 2026-09-23 — Sensor domain gap is a known risk

`zhang_who` was recorded with Byteflies sensors at 100 Hz, not a Wear OS watch. The
replay harness (M3) cannot reveal this gap; only `own_watch` (M6) can. Mitigations:
- M1: per-dataset axis-frame table (axis order, signs, which wrist) in
  `docs/label_mapping.md`, mapped to Android's sensor frame at ingest.
- M2: training-only augmentation — small rotation (±15°, not arbitrary SO(3); large
  rotations fight gravity alignment), amplitude scaling, Gaussian jitter. Report metrics
  with and without it.
- M6: held-out on-watch macro-F1 is the real answer; report it whatever it is.

## D11 — 2026-09-23 — Microphone / water sound stays out of v1

Detecting running water via the watch mic is a proven signal (Apple Watch combines it with
motion), but it doesn't help v1: manual mode already knows a wash started, sound can't
tell steps apart, and the tap is often off during WHO scrubbing. It also needs
`RECORD_AUDIO` and contradicts §1.4. **§1.4 unchanged.** Recorded as an M5 candidate
only: an on-device audio classifier (e.g. YAMNet, which has tap/sink classes) as a
confirmation gate after IMU spotting fires, audio never stored. Adopting it requires
amending §1.4 and §2.1 permissions in a new decision.

## D12 — 2026-09-23 — Preprocessing contract pinned down (§8.7, §8.2, M1, M3, §2.1)

Gaps found before M1 that would have made the §9.2 parity test fail or left behaviour
undefined:

- **Causal, streaming filters.** `sosfilt` over the continuous stream with state carried
  across windows; never `filtfilt`. The watch can only filter causally.
- **Gravity-align before bandpass.** The 1 Hz high-pass removes gravity, so alignment
  must come first. Algorithm defined in §8.7 (causal 0.3 Hz low-pass gravity estimate,
  minimal rotation to `+z`). New `model_meta.json` field
  `preprocessing.gravity_lowpass_hz`, added to the JSON schema and stub generator.
- **Window labels:** majority label if ≥ 60 % of samples, else `-1` and excluded.
- **Resampling:** `resample_poly` (anti-aliased) for `zhang_who` 100 → 50 Hz.
- **`WAKE_LOCK`** added to §2.1 for the session duration only; to be confirmed on the
  emulator at M3.
- **R8 keep rule** for LiteRT added to M3.
- §5 / M1 updated for D7 and D8.

`spec_version` stays `1.0`: no release carrying the old contract has been published
(`v0.0.1-stub` is still unreleased), so there is nothing to be incompatible with. After
the first release, contract changes bump it.

## D13 — 2026-09-23 — Spec split into core / AI / app

`docs/HapticWash-SPEC.md` now holds only the shared contracts (§1, §2.3, §3 boundary,
§4.3, M0, M6, §7, §8.1–8.3, §8.7, §9.1–9.2, §10–12). AI-only sections moved to
`docs/SPEC-AI.md`; app-only sections to `HapticWash - Main/docs/SPEC-APP.md`.

- **§ numbers are unchanged and global**, so existing `SPEC x.y` references in code and in
  D1–D12 still resolve; the core spec's section index maps each § to its file.
- M5 split into **M5-AI** (spotting model, `ocdetect` ingest) and **M5-App** (gating,
  duty-cycling, toggle). Content unchanged.
- M2 gains an explicit deliverable: export `golden/*.npy` with the release. §9.2 already
  required it; M2 just didn't list it.
- The app links to the core spec instead of keeping a copy (old §3 "symlink or copy").

## D14 — 2026-09-23 — Add `uwash` to the M1 corpus as a second primary step source

Found via the UWash paper (Wang et al., IEEE TMC 2025, arXiv `2112.06657`). It is the
largest public source of WHO step labels: **51 subjects at 5 locations**, recorded at
50 Hz on a real wrist smartwatch (Samsung Gear Sport), with **per-sample** labels. It has
5× the subjects of `zhang_who` and is closer to the Wear OS domain. Downloaded and
checksummed. Provenance, file format and quirks are in `docs/DATASETS.md`.

- **Corpus:** M1 = `zhang_who` + `uwash` + `ablutomania` (amends D8). `uwash` needs no
  resampling.
- **D9 superseded in part:** labelled subjects go from 10 to 61, so LOSO is no longer
  thin. Hand-annotating `ablutomania` drops from "highest-value data work" to unneeded
  unless M2 shows a gap. The per-fold report stays.
- **D10:** `uwash` narrows the sensor gap (a real watch at 50 Hz) but doesn't close it
  (Tizen, not Wear OS). It needs its own row in the axis-frame table. Gyro units look
  like °/s (Android uses rad/s); confirm this at ingest.
- **D4:** MIT covers the data (owner confirmed 2026-09-24), so a model trained without
  `zhang_who` avoids the CC-BY-NC-ND question entirely.
- **Split:** LOSO over the labelled subjects. `uwash` subject ids follow the schema rule
  `<key>_<local_id>`: `uwash_canteen_3`. The location prefix also allows a
  leave-one-location-out check.

Provisional label mapping (finalised in `docs/label_mapping.md` at M1, consistent with
D5):

| uwash `label` | gesture | canonical §1.3 |
|---|---|---|
| 1 | palm to palm | `PALM_TO_PALM` (1) |
| 2, 3 | back of hand (R over L / L over R) | `BACK_OF_HAND` (2) |
| 4 | palm to palm, fingers interlaced | `INTERLACED_FINGERS` (3) |
| 5 | backs of fingers, interlocked | `WASH_OTHER` (6) — same as `zhang_who` Action 4 (D5) |
| 6, 7 | thumbs (L / R) | `THUMBS` (4) |
| 8, 9 | fingertips (L / R) | `FINGERTIPS` (5) |
| 0 | everything else | `NULL` (0); edge zone → `-1` (see below) |

**Label 0 (corrected 2026-09-24).** Measured on the data: *inside* a wash the nine
gestures are contiguous. Label-0 runs between gestures are 12–40 s (median 18 s, 271 runs
≈ 5 per file); these are the gaps *between* washes. So label 0 is walking plus wetting,
soaping, rinsing and drying, with no boundary between them. Relabelling it
`WASH_OTHER` would teach the model that walking is washing.
Rule: label 0 → `NULL`, except the **edge zone** (N s immediately before the first
gesture and after the last gesture of each wash), which is set to `UNLABELLED` (`-1`)
and excluded. That way wet/soap/rinse is never trained as `NULL`, which would conflict
with `zhang_who`'s `WASH_OTHER` (D5). Start with N = 5 s. It is a tuning knob: check it
against the M1 plots.

## D15 — 2026-09-24 — Train on `uwash`, test on `zhang_who` (closes D4)

- **Train:** `uwash` only, leave-one-subject-out: each fold trains on 50 subjects and
  tests on the one left out, rotated over all 51. Mean ± std and every fold are reported.
- **External test:** `zhang_who` (all 10 subjects) is never used for training or model
  selection. It is a cross-device check: Byteflies at 100 Hz vs the Gear Sport at 50 Hz (D10).
- **Licence:** `zhang_who` is Creative Commons Attribution-NonCommercial-NoDerivatives 4.0.
  Using it for evaluation only means no shipped weights are trained on it, so the ND
  question in D4 no longer applies. NC still holds: evaluation stays non-commercial.
- **`ablutomania`:** it has no step labels, so it can't train or test the step classifier.
  Its roles are confounders and NULL, which only the spotting model needs.
