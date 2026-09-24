# M0 sign-off — HapticWash-AI

**Date:** 2026-09-24 · **Commit checked:** `5f36a6b` (`develop`) · **Scope:** this repo only.
The Wear OS repo signs off its own half of M0.

**Status: NOT SIGNED OFF.** Everything is built and passes locally. Two items are still
open: CI has never run on GitHub, and the stub release isn't published. Both need the
owner (see [Open items](#open-items)).

## Deliverables

| Deliverable | Status | Evidence |
|---|---|---|
| Build files, lint, CI workflow | ✅ built · ❌ never run | `pyproject.toml` + `uv.lock`, ruff, `.github/workflows/ci.yml` (`1d8bb33`). `develop` is not pushed, so GitHub shows no workflow runs. |
| `haptic_ai/schema.py` with `validate(df)` | ✅ | `8a72781`; dataset keys changelog in `89c20e8` |
| `docs/DATASETS.md`: licence + citation per dataset | ✅ | All used datasets verified; `harage` dropped (D7) |
| Stub `model.tflite` + valid `model_meta.json` | ✅ | `scripts/make_stub_model.py` (`a18a2d7`) |
| Published as release `v0.0.1-stub` | ❌ | The repo has no releases |
| `docs/DECISIONS.md` started | ✅ | D1–D15 |

## Acceptance criteria

| Criterion | Result | How it was checked |
|---|---|---|
| `make validate-schema` passes on the 100-row fixture | ✅ | `uv run python -m haptic_ai.cli validate-schema tests/fixtures/canonical_100rows.csv` → `OK … 100 rows, 2 subjects` (Windows has no `make`, so this is the command the target runs) |
| Stub artifacts load in a test; shapes match the tensor contract | ✅ | WSL Debian, `uv sync --all-extras`: `make_stub_model.py` wrote `model.tflite` (27,416 B); `pytest tests/test_stub_model.py tests/test_model_meta_schema.py` → 11 passed. On Windows the TFLite test skips (no TensorFlow, D2). |
| `model_meta.json` validates against `docs/model_meta.schema.json` | ✅ | Covered by `tests/test_model_meta_schema.py` (automated rather than manual) |
| CI green | ❌ | Not run: branch not pushed. Every CI step passes locally: ruff check + format, schema fixture, stub build (WSL), full `pytest` (25 passed, 1 skipped on Windows; skipped test passes in WSL). |

## Done ahead of M1

Not required for M0; listed so the sign-off reflects the whole branch.

- Training/test split decided: train on `uwash`, test on `zhang_who` (D15). This closes the `zhang_who` licence question (D4).
- `uwash` ingest adapter: 1,009,165 rows, 51 subjects, all passing `validate` (`89c20e8`).
- `scripts/fetch_data.py`: downloads and checks MD5s; all 39 local files match the published checksums (`5f36a6b`).

## Open items

1. **Push `develop`** so CI runs, then confirm the run is green: `git push origin develop`.
2. **Publish the stub release.** `gh` is not installed on the Windows machine; from WSL or
   after installing it:
   `gh release create v0.0.1-stub artifacts/model.tflite artifacts/model_meta.json --title "v0.0.1-stub" --notes "Stub model, random weights."`
   Build the artifacts first with `uv run --extra ml python scripts/make_stub_model.py` (Linux).
3. **Wear OS repo** M0 items (ktlint, CI, manifest test) are tracked there.

When 1 and 2 are done, update the status line above with the CI run URL and release link.

**Signed off by:** _pending_
