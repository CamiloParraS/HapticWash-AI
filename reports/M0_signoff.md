# M0 sign-off — HapticWash-AI

**Date:** 2026-09-24 · **Commit checked:** `91395cf` (`develop` = `main` = tag `v0.0.1-stub`) · **Scope:** this repo only.
The Wear OS repo signs off its own half of M0.

**Status: ready to sign off. One step left:** CI is green and release `v0.0.1-stub` is
published, but `model.tflite` and `model_meta.json` are not attached to it yet (see
[Open items](#open-items)).

- CI: <https://github.com/CamiloParraS/HapticWash-AI/actions/runs/36010608635> (tag run, success)
- Release: <https://github.com/CamiloParraS/HapticWash-AI/releases/tag/v0.0.1-stub>

## Deliverables

| Deliverable | Status | Evidence |
|---|---|---|
| Build files, lint, CI workflow | ✅ | `pyproject.toml` + `uv.lock`, ruff, `.github/workflows/ci.yml` (`1d8bb33`). Green on `develop`, `main` and the tag at `91395cf`. |
| `haptic_ai/schema.py` with `validate(df)` | ✅ | `8a72781`; dataset keys changelog in `89c20e8` |
| `docs/DATASETS.md`: licence + citation per dataset | ✅ | All used datasets verified; `harage` dropped (D7) |
| Stub `model.tflite` + valid `model_meta.json` | ✅ | `scripts/make_stub_model.py` (`a18a2d7`) |
| Published as release `v0.0.1-stub` | ⚠️ | Release published 2026-09-24. **No files attached**: the stub is only a CI build artifact (`stub-model-v0.0.1`, deleted after 90 days). |
| `docs/DECISIONS.md` started | ✅ | D1–D15 |

## Acceptance criteria

| Criterion | Result | How it was checked |
|---|---|---|
| `make validate-schema` passes on the 100-row fixture | ✅ | `uv run python -m haptic_ai.cli validate-schema tests/fixtures/canonical_100rows.csv` → `OK … 100 rows, 2 subjects` (Windows has no `make`, so this is the command the target runs) |
| Stub artifacts load in a test; shapes match the tensor contract | ✅ | WSL Debian, `uv sync --all-extras`: `make_stub_model.py` wrote `model.tflite` (27,416 B); `pytest tests/test_stub_model.py tests/test_model_meta_schema.py` → 11 passed. On Windows the TFLite test skips (no TensorFlow, D2). |
| `model_meta.json` validates against `docs/model_meta.schema.json` | ✅ | Covered by `tests/test_model_meta_schema.py` (automated rather than manual) |
| CI green | ✅ | [Run 36010608635](https://github.com/CamiloParraS/HapticWash-AI/actions/runs/36010608635) on the tag, plus `develop` and `main`, all success: ruff, schema fixture, stub build, full `pytest` with TensorFlow (Linux, so no skips). |

## Done ahead of M1

Not required for M0; listed so the sign-off reflects the whole branch.

- Training/test split decided: train on `uwash`, test on `zhang_who` (D15). This closes the `zhang_who` licence question (D4).
- `uwash` ingest adapter: 1,009,165 rows, 51 subjects, all passing `validate` (`89c20e8`).
- `scripts/fetch_data.py`: downloads and checks MD5s; all 39 local files match the published checksums (`5f36a6b`).

## Open items

1. **Attach the stub files to the release.** The Wear OS app downloads them from there.
   Use the files CI built at the tagged commit: download the `stub-model-v0.0.1`
   artifact from [the tag run](https://github.com/CamiloParraS/HapticWash-AI/actions/runs/36010608635),
   unzip it, and drag `model.tflite` and `model_meta.json` onto the release's edit page. Or,
   with `gh`: `gh release upload v0.0.1-stub model.tflite model_meta.json`.
2. **Wear OS repo** M0 items (ktlint, CI, manifest test) are tracked there.

After item 1, change the release row to ✅ and sign below.

**Signed off by:** _pending_
