# M2 step classifier report

Release candidate `step-classifier` 0.1.0: a 1D CNN with D10 augmentation, 3 s windows at
50 % overlap, and a causal moving average over 3 s (k = 2). Trained on `uwash` only (D15).
Sources: `M2_step_cnn.json` (sweep @ `1274110`), `M2_step_trees.json` (@ `c8ef204`), and the
export check below (@ `c150c52`). Metric: macro-F1 over the five WHO steps (labels 1–5),
computed **per subject** and given as mean ± std over the 51 leave-one-subject-out folds.
The subject-disjointness of every fold is asserted in code (`evaluate.check_disjoint`,
`tests/test_evaluate.py`).

## Headline

| | LOSO macro-F1 (uwash, 51 subjects) | chance (dummy) |
|---|---|---|
| **CNN + aug, 3 s, MA 3 s** | **0.705 ± 0.127** (median 0.73, min 0.44, max 0.96) | 0.10 |

Target ≥ 0.65: **met**. Stretch ≥ 0.75: not met. Two subjects fall below 0.5
(`uwash_hongli_6` 0.44, `uwash_canteen_8` 0.49). Every fold's score is in the JSON.

## Window-size and model sweep (LOSO macro-F1, mean ± std)

| window | dummy | RF | GB | CNN | CNN + aug |
|---|---|---|---|---|---|
| 2 s | 0.12 ± 0.03 | 0.59 ± 0.11 | 0.61 ± 0.10 | 0.66 ± 0.11 | 0.68 ± 0.11 |
| 3 s | 0.12 ± 0.04 | 0.61 ± 0.12 | 0.63 ± 0.11 | 0.68 ± 0.12 | 0.70 ± 0.11 |
| 5 s | 0.12 ± 0.04 | 0.62 ± 0.12 | 0.67 ± 0.12 | 0.71 ± 0.12 | 0.70 ± 0.13 |

Unsmoothed. **Why 3 s + aug:** the top three settings (5 s CNN 0.707, 3 s CNN + aug with
MA 3 s 0.705, 3 s CNN + aug raw 0.702) differ by less than 0.005, far inside the fold noise
(standard error ≈ 0.017). The tie is broken on the watch's needs: a 3 s window reacts faster
to a step change than a 5 s one, and it is the SPEC default. zhang_who played no part in the
choice (D15).

## Smoothing (causal moving average, set in seconds; D21)

| CNN + aug | raw | MA 2 s | MA 3 s |
|---|---|---|---|
| 2 s (k = 2 / 3) | 0.677 | 0.684 | 0.676 |
| 3 s (k = 1 / 2) | 0.702 | 0.702 | 0.705 |
| 5 s (k = 1 / 1) | 0.699 | 0.699 | 0.699 |

The SPEC's starting point, k = 5 *windows*, cut GB from 0.67 to 0.37 at 5 s, because it looked
back 12.5 s and a step lasts about 6 s (D20). Set in seconds, smoothing is roughly neutral on
F1. It is kept at 3 s to steady the haptic cues.

## Per-class F1 (release setting, pooled over folds)

| NULL | PALM_TO_PALM | BACK_OF_HAND | INTERLACED | THUMBS | FINGERTIPS | WASH_OTHER |
|---|---|---|---|---|---|---|
| 0.92 | 0.63 | 0.81 | 0.67 | 0.85 | 0.66 | 0.58 |

Main confusions (row %, true → predicted): interlaced → back of hand 15, palm → fingertips 11,
thumbs → WASH_OTHER 12, fingertips ↔ palm 10. These are the gestures with similar motion.

## Augmentation (D10, D21)

Rotation ±15°, amplitude × U(0.5, 1.5), jitter 0.05·std. LOSO gains +0.02 at 2 s and 3 s and
loses 0.01 at 5 s. On zhang_who it gains more (left wrist 0.47 → 0.59 at 3 s). Caveat
(D21): the amplitude range was motivated by zhang's lower amplitude (D20), so zhang is not
fully blind to this choice. The range was fixed before any augmented result and not tuned.

## External test: zhang_who (never used for training or selection)

Release setting, per wrist, mean over 10 subjects: **left 0.56, right 0.44** (right mirrored
at test time, D19). Trees managed 0.13–0.27 (D20). THUMBS transfers (0.86–0.88).
INTERLACED (0.16–0.33) and, on the right wrist, PALM_TO_PALM (0.13) do not. The remaining gap
is the D10 sensor/protocol risk. Only the on-watch test (M6) measures the real domain.

## Quantization and export (D22)

| check | value | requirement |
|---|---|---|
| model size | 37,992 B | ≤ 150 KB |
| macro-F1 drop, int8 vs float (zhang, local) | 0.0000 | ≤ 0.02 |
| golden top-1 agreement, TFLite vs float | 100 % (298 windows) | — |
| golden max \|Δp\|, TFLite vs float | 0.021 | test bound 0.1 |
| `model_meta.json` schema + label order | pass | `tests/test_stub_model.py` |

The release model trains on 50 uwash subjects. `uwash_library_2` is held out and supplies the
golden files (`raw.npy` is its 449.5 s session). No zhang data is in `artifacts/`.

## Known gaps

- Stretch target (0.75) not reached. Most headroom: the interlaced/back-of-hand and
  palm/fingertips confusions.
- Training data is left wrist only. Mirror augmentation for right-wrist wearers is deferred
  (D21), and right-wrist transfer is the weakest number here.
- Reproducibility (SPEC 4.4, ±0.005) is seeded and uses op determinism, but it has not been
  verified by a second run of the ~3 h sweep.
- Tizen-vs-Android axis frame still unverified (D19).
