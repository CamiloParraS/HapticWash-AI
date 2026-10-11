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

## D16 — 2026-09-24 — M1 corpus build choices

- **Dependencies:** `xlrd` (zhang_who `FrameACM_*.xls` is BIFF) and `openpyxl`
  (`FrameACM_30.xls` is actually xlsx; PART 2 `annotation.xlsx`). Both pure Python.
- **The corpus stays raw.** §8.1 says acc includes gravity, so `data/processed/` holds
  only resample + units + gap splits. Gravity-align, bandpass and mirroring live in
  `haptic_ai/preprocess.py` as per-session array functions, and M2 applies them to each
  continuous session before windowing (§8.7).
- **Gap split:** any gap > 100 ms starts a new session `<session>_t<ms>` (see D18), and time restarts
  at 0. Applied to every dataset. `uwash` is not re-gridded (D14), so its jitter goes
  straight to M2: dt p1 / p99 = 1 / 34 ms.
- **zhang_who unannotated samples → `-1`**, not `NULL`: before, between and after steps
  the subject is at the sink.
- **Gravity-align singularity:** when gravity is exactly −z, use a 180° rotation about x.
  The Kotlin side must do the same (§9.2).
- **`ablutomania` deferred:** not downloaded, no step labels (D15), not used by M2. M1 ships
  `zhang_who` + `uwash`. The adapter lands when the files are fetched, at the latest
  with M5.
- Known data issues (zhang gyro X clipping, uwash jitter) are in `docs/label_mapping.md`.

## D17 — 2026-09-24 — `zhang_who` gyro scale is 131 LSB/(°/s), not the readme's 16.384

The readme's 16.384 (±2000 °/s) made zhang gyro 3.5–8× uwash during the same steps, with
|gyro| held near 38 rad/s for seconds. The raw data clips at int16 full scale, so the
scale sets the range. Two checks point to ±250 °/s (131, an 8× smaller value):

- **Gravity rotation:** the rate the gravity direction turns in acc, fitted against gyro,
  gives scale factors of 0.07–0.14 with 16.384 (≈ 1/8). uwash gives 0.93.
- **Clipping rate:** 1.7 % of zhang per-axis samples sit at ±32767. uwash exceeds 250 °/s
  on 2.9 % and 500 °/s on 0.6 %. A ±250 °/s range matches; ±500 °/s (65.5) would clip
  about 3× less than observed.

Acc stays at 4096 LSB/g (±8 g). The clipped samples are kept (`docs/label_mapping.md`).

## D18 — 2026-09-24 — M1 review fixes: uwash re-grid and shared files, zhang wet/soap/rinse

From the human review of the M1 plots.

- **uwash re-gridded to 50 Hz, dropouts ≤ 200 ms interpolated.** The raw timestamps are
  jittery (dt p1 = 1 ms) and have 2,125 gaps over 100 ms. They are real dropouts, not
  batching: after a 100–150 ms gap the next second holds ~46 samples, not 50. Splitting at
  every one gave 2,176 sessions with a median of 2.9 s, shorter than the M2 windows
  (2–5 s). Now each file is linearly interpolated onto a 20 ms grid, bridging gaps up to
  200 ms (≤ 9 missing samples, ~70 % of the gaps). Longer gaps get no grid points, so the
  100 ms session split still applies. **Amends the SPEC M1 "gaps are never interpolated"
  criterion** for gaps of 100–200 ms. `MAX_BRIDGE_MS` is the knob.
- **uwash `library_6`/`library_7` and `library_9`/`library_10` share one recording.**
  Same sensor data and timestamps. Only the labels differ, and each file labels only its
  own subject's washes, so the other subject's washes read as label 0 (→ `NULL`). That
  poisoned `NULL` and put the same samples in two LOSO folds. Each file is cropped to its
  own half, cut midway between the two subjects' washes (431 s and 381 s).
- **zhang Action 0, 0.5, 7 (wet, soap, rinse) → `-1`** (amends D5). uwash label 0 doesn't
  separate these from walking, and its edge zone (D14) makes them `-1`. Keeping them
  `WASH_OTHER` in zhang would test the model on a class meaning it never trained on.
  `WASH_OTHER` is now backs of fingers only, in both datasets.
- **zhang washes end at the last annotated sample.** Wash 16's right wrist ran 96 s past
  the rinse (166.7 s against 70.5 s on the left).
- Report plots four sessions per dataset so zhang gets both wrists.
- **One split number per dataset.** zhang splits at gaps > 100 ms (it has none). uwash
  bridges gaps ≤ 200 ms at ingest, so no gap of 100–200 ms survives and its effective
  split threshold is **200 ms**.
- **Session ids are `<session>_t<ms>`**, the piece's start offset in the source recording,
  not a running `_s<k>` count. A count renumbers every later piece whenever an earlier gap
  changes, so reviewers could not track a session across runs.

## D19 — 2026-09-24 — uwash is the left wrist; zhang_who mapped into its frame

- **uwash wrist = left.** The paper doesn't state it, but its data-collection photos show
  the watch on the left wrist (owner check).
- **zhang_who axes: (x, y, z) → (−x, y, −z)** for acc and gyro, both wrists. During wash
  steps every uwash subject has mean acc x < 0, while zhang left has x ≈ +3 and zhang right
  x ≈ −3.5. Fitting per-step gravity directions, zhang left maps onto uwash best with x and
  z flipped: a 180° rotation about y, a proper rotation (det +1). zhang right is the
  x-mirror of zhang left, so the same rotation serves both wrists and `preprocess.mirror`
  (x flip) stays valid. Details in `docs/label_mapping.md`.
- **Still unverified:** that Tizen's frame equals Android's. uwash is the reference frame,
  so if it is off, both datasets are off together. Check against a real Wear OS watch.
- **M2 consequence:** training data is left wrist only. Mirror augmentation is needed to
  cover right-wrist wearers, and zhang results are reported per wrist.

## D20 — 2026-09-24 — M2 feature baselines: first LOSO numbers and what they show

`uv run python -m haptic_ai.cli evaluate --config configs/step_trees.yaml` →
`reports/M2_step_trees.json`. LOSO over 51 `uwash` subjects, macro-F1 on steps 1–5, mean ± std
per subject, dummy (stratified) alongside.

| window | dummy | RF | GB | GB + MA k=5 | GB on zhang L / R |
|---|---|---|---|---|---|
| 2 s | 0.12 | 0.59 ± 0.11 | 0.61 ± 0.10 | 0.59 | 0.25 / 0.27 |
| 3 s | 0.12 | 0.61 ± 0.12 | 0.63 ± 0.11 | 0.52 | 0.22 / 0.22 |
| 5 s | 0.12 | 0.62 ± 0.12 | **0.67 ± 0.12** | 0.37 | 0.13 / 0.19 |

- **Choices:** 12 features per axis + 6 within-sensor correlations (78). RF uses
  `class_weight="balanced_subsample"`. GB (`HistGradientBoosting`, 100 iters) has no class
  weight: with it a fit took 108 s (sample weights disable histogram subtraction), and the
  step classes are within 2× of each other. Windows labelled −1 are kept for smoothing (the
  watch predicts every window) and dropped only for training and scoring. zhang right
  wrist is mirrored into the left frame at test time (D19).
- **Moving average k=5 hurts, more so at longer windows.** k counts windows, so it looks
  back k × stride = 5–12.5 s, but a uwash step lasts about 6 s per wash. The SPEC's k=5
  needs to be set in seconds (≈ 2–3 s) instead. Not tuned yet.
- **zhang_who transfer is poor (0.13–0.27), and almost all of it is "predicted NULL".**
  zhang subjects scrub at about half uwash's amplitude on steps 1, 2, 3, 5 (bandpassed
  acc RMS ≈ 1.5 vs 3.5 m/s², gyro ≈ 0.5 vs 1.0 rad/s). THUMBS has the same amplitude in
  both and transfers at 91–98 %, which also supports the D19 axis mapping. A scale bug
  would shrink every step equally, so this is a behaviour gap (a guided lab protocol vs
  natural washing), and exactly the risk in D10. Next: the D10 amplitude-scaling
  augmentation, and features that are less energy-bound.
- The window size is **not chosen yet**: 5 s wins LOSO but has the fewest training windows
  and transfers worst. Decide after the CNN comparison.

## D21 — 2026-09-24 — M2 CNN: Keras, smoothing in seconds, D10 augmentation, CPU training

- **Framework: Keras / TensorFlow 2.17** (closes the SPEC §2.2 choice and D2's "revisit at
  M2"). TF is already pinned in the `ml` extra, the stub model exports through it in CI, and
  `export.py` is written against it. PyTorch would not remove the Linux constraint:
  `ai-edge-torch` is Linux-first too. CNN training and export run in WSL or CI.
- **CNN:** 3 × (Conv1D → BatchNorm → ReLU → MaxPool), filters 32/64/64, kernels 5/5/3,
  global average pool, dropout 0.3, softmax over the 7 labels. About 22k weights, well under
  the 150 KB budget at int8. Balanced class weights. Fixed 30 epochs, no early stopping,
  because the only held-out data in a LOSO fold is the test subject. z-score stats come from
  the un-augmented training windows and become `norm_mean` / `norm_std`.
- **Smoothing is set in seconds:** `moving_average_s` in the configs, converted to
  k = round(s / stride) per window size. The report shows raw, 2 s and 3 s. The chosen value
  becomes `moving_average_k` in `model_meta.json` for the chosen window.
- **D10 augmentation** (`preprocess.augment`, training only, redrawn every epoch): rotation
  within ±15° about a random axis, amplitude × U(0.5, 1.5), jitter at 0.05 × channel std.
  `cnn` and `cnn_aug` are both reported. **Caveat:** the amplitude range was motivated by the
  zhang_who amplitude gap (D20), so zhang is no longer fully blind to the augmentation design.
  To limit that, the range is a generic ±50 % fixed before any augmented result, and it is not
  tuned on zhang numbers.
- **CPU, not GPU.** The RTX 4050 works in WSL2 with `tensorflow[and-cuda]`, but it was
  slower than the CPU for this model at every batch size (batch 256: ~3.5 s vs 1.9 s per
  epoch). The model is too small to hide the transfer and launch overhead. No `gpu` extra was
  added. Batch 256 with learning rate 0.002 (scaled up from 64 / 0.001, not tuned).
- **WSL gotcha:** piping TF's CUDA logging through `wsl` crashed the WSL VM twice
  (`RPC_S_CALL_FAILED`). Redirect long runs to a file.
- **Deferred:** mirror augmentation for right-wrist wearers. It has to act on raw axes
  before gravity alignment, so it belongs in `session_windows`. Until then, zhang right is
  mirrored at test time (D19).

## D22 — 2026-09-25 — Export: golden files from a held-out uwash subject, never zhang_who

`python -m haptic_ai.cli export --config configs/step_cnn.yaml` (Linux) trains the release
model and writes `artifacts/`: `model.tflite`, `model_meta.json`, `golden/{raw,inputs,
outputs}.npy`. The old `train` and `golden` subcommands are folded into it.

- **Golden windows come from uwash, not zhang.** Golden files ship with the release, and
  zhang_who is CC-BY-NC-ND, so its windows can't be redistributed (D4, D15). SPEC 9.2 wants
  windows from held-out subjects, so the release model trains on 50 of the 51 uwash subjects,
  and the 51st, `uwash_library_2` (MIT), supplies the golden files. It was chosen because it
  has the longest continuous session (449.5 s), not for its scores. Losing one subject from
  training is the price.
- **`golden/raw.npy`** is that whole session as a raw stream `(n, 6)` from its first sample,
  so the watch can run the stateful chain (align → bandpass → windows → z-score) from zero
  state and compare with `inputs.npy` (SPEC 9.2 item 4). `tests/test_export.py` reproduces
  `inputs.npy` from `raw.npy` in Python.
- **`golden/outputs.npy` is the TFLite model's output** in the Python interpreter, the same
  artifact the watch runs, so SPEC 9.2's 1e-2 tolerance measures runtime parity and not
  quantization error. The float Keras outputs go to `artifacts/float_outputs.npy` (not
  released). `test_export` bounds the difference: max |Δp| < 0.1, top-1 agreement ≥ 95 %.
- **Quantization:** dynamic range (int8 weights, float32 in/out, SPEC 8.3). The ≤ 0.02 F1
  drop is measured on zhang_who locally. It is never written to a released file.
- **Smoke run (1 epoch, not a release):** 38 KB model, F1 drop −0.0005, 298 golden windows,
  max |Δp| vs float 0.0014.

## D23 — 2026-09-25 — M2 release setting: CNN + augmentation, 3 s windows, 3 s smoothing

From the LOSO sweep on uwash only (`reports/M2_step_classifier.md`): 0.705 ± 0.127 vs a
0.10 dummy. The top three settings were within 0.005, inside the fold noise, so 3 s beats 5 s
on latency, and it is the SPEC default. `moving_average_k = 2` (3 s at a 1.5 s stride). Export
@ `c150c52`: 37,992 B, int8 F1 drop 0.0000. zhang_who external: left 0.56, right 0.44.

## D24 — 2026-09-27 — App M3: LiteRT 1.4.2, JUnit 4, `com.hapticwash` package

App-side (`HapticWash - Main`, branch `m3/app-skeleton`).

- **LiteRT `com.google.ai.edge.litert:litert:1.4.2`**, pinned. 2.x ships `litert` and
  `litert-api` under the same Android namespace, which AGP 9's manifest merger rejects; 1.4.2
  has distinct namespaces, merges no permissions, keeps `org.tensorflow.lite.Interpreter`, and
  bundles x86/x86_64 for the emulator. CPU, one thread.
- **Filter design at runtime.** The app computes the Butterworth SOS from `model_meta.json`
  (scipy's analog prototype -> lp2bp -> bilinear) instead of shipping coefficients, so §8.2
  stays unchanged. The JVM golden test reproduces `golden/inputs.npy` from `golden/raw.npy`
  with max |Δ| 0.0.
- **Tests: JUnit 4** (already in the template), not the JUnit 5 + Turbine in SPEC-APP §2.1:
  JUnit 5 on Android needs a third-party Gradle plugin, and nothing in M3 needs it.
  `org.json` is a test-only dependency (android.jar's copy is a stub on the JVM).
  `kotlinx-coroutines-android` 1.11.0 (the §8.4 `Flow` contract) and `androidx.test:runner`
  1.7.0 (instrumented tests) are also new.
- **R1 / §8.5 manifest checks are JVM tests** over AGP's merged debug and release manifests,
  not instrumented: they cover the release manifest too and run without an emulator. They read
  AGP's intermediate path, so an AGP upgrade may move it.
- **Replay files go to app-internal storage.** The app holds no storage permission, so a
  `/sdcard/Download` path from §8.5 is not readable. `scripts/replay_demo.sh` pipes the CSV in
  with `run-as` and opens the activity first, because a background app can't start a
  foreground service.
- **Deferred to M4:** `postprocessing.moving_average_k` is parsed but not applied (M3 shows the
  raw label), and timer-only mode shows no duration yet.
- **Package `com.hapticwash`** (was `com.example.hapticwash`) for the wear module, to match the
  §8.5 replay intent.

## D25 — 2026-09-28 — Mirrored golden file for the app's wrist mirroring (M4 slice 0)

- **`golden/inputs_right.npy`**: `golden/raw.npy` mirrored with `preprocess.mirror` on the raw
  axes, then the normal chain (gravity-align -> bandpass -> windows -> z-score with
  `norm_mean`/`norm_std`), float32, same shape as `inputs.npy`. The AI repo stays the one
  definition of mirroring; the app's Kotlin port (M4 slice (d)) must reproduce this file, as
  §9.2 does for `inputs.npy`.
- **Generated without retraining.** `export` trains the release model (D22), so re-running it
  would be a new release. The file depends only on `raw.npy` and the preprocessing in
  `model_meta.json`, so `haptic_ai.cli golden-mirror` (`windows.write_mirrored_golden`, no TF,
  runs on Windows) writes it, and `export` calls the same function so every release ships it.
  `model.tflite`, `model_meta.json` and the other golden files are unchanged.
- **uwash only** (MIT): the golden session is the held-out uwash subject; never zhang_who
  (D4, D15, D22).
- No `training_wrist` field in §8.2/§8.3: mirroring is frame geometry (D19), not a model
  constant. Mirror augmentation stays deferred (D21). The across-the-wrist x-axis assumption in
  `preprocess.mirror` is still unverified on hardware (D19, M6): the fixture proves the repos
  agree, not that the physics is right.

## D26 — 2026-09-28 — App M4 batch replay on zhang_who: 5 of 10, reported as measured

- **Result** (`HapticWash/reports/M4_replay.md`, model 0.1.0, `Wear_OS_Large_Round` AVD):
  5 of 10 held-out zhang_who sessions match ground-truth Step Status on ≥ 4 of 5 Steps. The
  §6 M4 criterion (≥ 8 of 10) is **not met**. Left wrist 4 of 5 pass, right wrist 1 of 5,
  consistent with the external macro-F1 gap (left 0.56 / right 0.44, D23).
- **Not tuned.** zhang_who is the external test set (D15); fitting Coverage thresholds to it
  would be leakage. Thresholds stay at the §6 defaults and M4 moves on.
- **Mirroring pays.** Right-wrist sessions replayed mirrored match 15 of 25 Step Statuses;
  the same sessions forced unmirrored match 7 of 25. The app's Kotlin mirror was pulled
  forward into slice (b) for this comparison; it reproduces D25's `inputs_right.npy`
  (max |diff| 4.8e-7).
- **Error shape.** Misses are app under-counting (`NOT_DETECTED`/`BRIEF` where ground truth
  is `OK`), on Between fingers (6 of 10 sessions), Fingertips (5) and,
  right wrist only, Palms (4 of 5); zero false `BRIEF`s, so hard
  credit stays (the soft-credit trigger in M4-PLAN did not fire).
- **The shortfall is the model, not the app** (to investigate in the AI repo). Same script,
  same build, on the one uwash subject held out of training (`uwash_library_2`, D22), its
  449 s session cut into washes at > 5 s NULL gaps (6 washes of ~48 s; a 6.9 s fragment
  dropped as under the 10 s minimum): **6 of 6 pass, 30 of 30 Step Statuses match**, app
  seconds within a few seconds of ground truth. Counting, smoothing and ground truth agree
  in-distribution; the zhang_who gap is cross-dataset generalisation (new people, new watch,
  and on the right wrist either the unverified mirror axis, D19, or genuinely different
  motion). Limits: one subject, and every Step there is really ≥ 4.6 s, so a correct
  `NOT_DETECTED` is never exercised.
- **Re-evaluating a new model** is a rerun, not new work: copy the new `model.tflite` and
  `model_meta.json` into the app's `wear/src/main/assets/model/` and the new
  `golden/*.npy` into `wear/src/androidTest/assets/golden/`; run the app's unit and
  `GoldenParityTest` checks; then `scripts/replay_batch.sh` on the zhang_who set (writes
  `reports/M4_replay.md`; the previous one stays in git history to diff against) and on the
  uwash set (cut command in the script header). If the new model trains on
  `uwash_library_2`, recut the uwash set from whichever uwash subject it holds out, or the
  uwash check is in-sample and proves nothing. Thresholds stay untuned on zhang_who.
- Budgets, indicative on the emulator with a debug build: cold start median 1.84 s (over the
  1.5 s budget; slice (d) moves model load off first composition and re-measures on a release
  build); session heap ~22 MB Java + native PSS (within 40 MB).

## D27 — 2026-09-29 — App M4 History and settings: JSON file and SharedPreferences, not Room/DataStore

Amends SPEC-APP §2.1 and §6 M4 ("Room persistence").

- **History** is one JSON file in app-internal storage (`files/history.json`) holding the last
  20 real (`sensor`) Washes, newest first, parsed with the platform's `org.json` (no new
  dependency). Each write goes to a temp file and then an atomic rename. A file that fails to
  parse is renamed aside to `history.corrupt.json`, logged, and History starts empty, so the
  data stays recoverable. Replay Washes go to a separate debug-only `replay_washes.json`
  (cap 50).
- **Settings** (Wrist side, Haptics, Progress ring) live in `SharedPreferences`.
- **Why:** 20 rows and three flags don't need an SQL engine, KSP, migrations or a Flow-based
  preferences store. Evidence it holds: `WashStoreTest` (restart, cap, corrupt file), and the
  M4 UX audit (`docs/UX_AUDIT.md`), where History survives a process kill and a reboot.
- Revisit if History grows past a few hundred rows or needs queries.

## D28 — 2026-09-29 — App M4 navigation: `androidx.wear.compose:compose-navigation`

- New dependency `androidx.wear.compose:compose-navigation` 1.5.6 (same version as
  `compose-foundation`), for `SwipeDismissableNavHost`: a back stack with swipe-to-dismiss,
  the Wear OS convention (§4.1.5).
- Map, max depth 2: Start → In-progress → Summary (In-progress is popped), Start → History →
  a past Wash's Summary, Start → Settings.
- No result notification. A Wash that ends while the app is in the background is saved, and
  its end haptics still fire.

## D29 — 2026-09-30 — App M4 banned-language test: health-result language, not every word

- §4.1.9 is read as prohibiting **health-result language**, not every technical occurrence of
  a word. Generic uses ("Failed to load model", "Clean up history") pass.
- `LanguageTest` (JVM) scans every `values*/strings.xml` in the wear module, strings and
  string-array items, case-insensitive, on word boundaries. It bans the whole words
  `guarantee`, `sanitized`, `sterile`, `germ-free`, `perfect`, `wrong`, `incorrect` and the
  phrases `you missed`, `you failed`, `missed step`, `failed step`, `clean hands`,
  `hands are clean`, `are clean`, `is clean`, `now clean`. A failure names the string and the
  matched term. A self-check pins the matcher on known-bad and known-good samples.
- A new string that is result language without one of these terms still needs review. The
  list is a floor, not a proof.

## D30 — 2026-10-10 — M5 is Twist start and Auto-end; spotting moves to Future Work

- **M5-App** is now: a Twist (three quick forearm rotations) starts a Wash, behind a toggle
  that is off by default and warns about battery (§4.1.8). Every model-backed Wash ends
  itself on sustained `NULL` (Auto-end): cancelled if no washing is seen within 30 s, a
  keep-going nudge if washing pauses before 40 s, ended normally if washing stops after
  40 s. Full text: SPEC-APP M5-App and SPEC-CHANGELOG C11.
- **M5-AI** (spotting) moves to Future Work (Core §13). Its SPEC-AI text is kept, marked
  deferred. The M5-AI → M5-App edge is gone from §7; M5-App needs only M4.
- **Why not spotting now:** it needs `ocdetect` ingest (31.6 GB), a second model,
  duty-cycling, and precision tuning. Published all-day spotting F1 has a mean near 0.33,
  so false starts would be common. A deliberate gesture is cheap, explainable and testable
  with the data already here.
- **Why not the platform's wrist gestures:** Wear OS wrist gestures (flick in/out) drive
  system navigation; a third-party app can't bind its own action to them, and they vary by
  device. The Twist is detected by the app from the same accelerometer and gyroscope.
- **`ocdetect`** stays un-ingested (D8 amended). M5 replays one participant-day through the
  app to count false Twist starts; that is evaluation of a heuristic, not training.
- **Open, to settle in M5:** with targetSdk 37 a `dataSync` foreground service is capped at
  about 6 h a day, and the `health` type needs a permission §2.1 forbids. How background
  listening runs is decided when it is built, with its own D-entry.

## D31 — 2026-10-11 — App M5 Auto-end: confident `NULL` for 15 s; right-wrist early stops were a replay bug

- **Setting** (`AutoEndConfig` in the app, `ed75961`): a `NULL` stretch counts only windows
  whose smoothed `NULL` probability is ≥ 0.8; a lower one breaks the stretch. "Sustained" is
  15 s. Give-up window 30 s, target 40 s, as in D30.
- **Why:** the first setting (8 s of any top-1 `NULL`) ended `zhang_who_9_1_left` at 52.5 s,
  27 s before its labelled washing did. The model emits short, low-confidence `NULL` mid-wash
  on zhang_who (the same generalisation gap as D26). With 15 s at 0.8, none of the 10 zhang
  replays ends before its labelled washing.
- **Right-wrist early stops: not a model issue.** Ticket 03 first recorded them as one.
  `scripts/replay_demo.sh` did not pass the CSV's wrist column, so right-wrist sessions
  replayed unmirrored and looked like confident `NULL` mid-wash. Fixed in the same commit;
  `replay_batch.sh` always passed it, so D26's numbers are unaffected.
- **Caveat: chosen against zhang_who.** D26 keeps Coverage thresholds untuned on zhang_who
  (D15). These knobs were checked against it for one safety condition (never end before
  washing does), not fitted to a score, but zhang_who is no longer a clean test of Auto-end.
  The honest check is `own_watch` in M6; both knobs stay calibration knobs for that.
- **Cost:** a real stop is detected up to ~15 s late, so a Wash that ends after the target
  runs on that long before its Summary. Manual Stop still ends it at once.

## D32 — 2026-10-11 — App M5 background listening: a `health` service with batched wake-up sensors

Settles D30's open question.

- **Service type `health`, not `dataSync`.** It needs `FOREGROUND_SERVICE_HEALTH` plus one
  prerequisite; declaring `HIGH_SAMPLING_RATE_SENSORS` in the manifest is enough. That is an
  install-time permission with no prompt, and not on SPEC-APP §2.1's forbidden list. Ticket
  04 assumed `health` needed `BODY_SENSORS` or `ACTIVITY_RECOGNITION`; those are only two
  of the ways to qualify.
- **Why not `dataSync`:** with targetSdk ≥ 35 it gets 6 h of background run time per 24 h,
  reset only when the user brings the app to the foreground. The docs list no exemption
  (battery "unrestricted" included), so the toggle would quietly stop most days. A
  `BOOT_COMPLETED` receiver can't start it either, which rules out ticket 05. And Play's
  intended use is data transfer, not sensing. `health` has no time limit, may start at
  boot, and is meant for long-running fitness-style tracking.
- **One type for the app.** `WashService` moves from `dataSync` to `health` too, so a
  Twist-started Wash doesn't depend on a background start of a second service type.
  `FOREGROUND_SERVICE_DATA_SYNC` is dropped.
- **Listening shape:** `registerListener` with a low sampling rate and a
  `maxReportLatencyUs` (a calibration knob, ~1–2 s to start) on the **wake-up** variants
  (`getDefaultSensor(type, true)`). Non-wake-up sensors fill their FIFO while the processor
  sleeps and overwrite it, so a Twist with the screen off is lost. Wake-up sensors wake it
  when the FIFO fills or the latency passes, so a Twist is seen up to one latency late. No
  wake lock while listening; the Wash keeps its partial wake lock (D12).
- **Unknown until a real watch:** whether wake-up accelerometer and gyroscope exist (wake-up
  gyroscopes are often missing), their FIFO sizes (which cap the latency at a given rate),
  and whether the detector holds at the lower rate. Ticket 04 starts with a probe that logs
  these. Fallback without a wake-up gyroscope: a wake-up accelerometer as the trigger, then
  the gyroscope on briefly to confirm the Twist.
- **Costs and risks:** `health` and the `HIGH_SAMPLING_RATE_SENSORS` declaration need a
  Play Console justification (a wash timer is a health use; review may still disagree).
  Background listening still costs battery; ticket 06 measures it, and the §4.1.8 warning
  stays.
- Sources: developer.android.com — foreground service timeouts, service types, Android 15
  behavior changes; source.android.com — sensor batching.
