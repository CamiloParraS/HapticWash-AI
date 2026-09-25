"""SPEC M2 export checks on artifacts/ written by `haptic_ai.cli export`.

Skipped until export has run (needs TF: Linux only, D2). Shapes and the schema are in
test_stub_model.py; these add the size budget, quantization and golden-file checks.
"""

import json
from pathlib import Path

import numpy as np
import pytest

ART = Path(__file__).parents[1] / "artifacts"
pytestmark = pytest.mark.skipif(
    not (ART / "export_check.json").exists(), reason="run `haptic_ai.cli export` first"
)


@pytest.fixture(scope="module")
def check() -> dict:
    return json.loads((ART / "export_check.json").read_text())


def test_size_budget():
    assert (ART / "model.tflite").stat().st_size <= 150 * 1024  # SPEC 4.3


def test_quantization_drop(check):
    assert check["quantization_f1_drop"] <= 0.02  # SPEC M2


def test_golden_files_match_tflite_and_float_model():
    tf = pytest.importorskip("tensorflow")
    from haptic_ai.export import run_tflite

    inputs = np.load(ART / "golden" / "inputs.npy")
    outputs = np.load(ART / "golden" / "outputs.npy")
    meta = json.loads((ART / "model_meta.json").read_text())
    assert inputs.dtype == outputs.dtype == np.float32
    assert inputs.shape[0] >= 50 and list(inputs.shape[1:]) == meta["input"]["shape"][1:]
    assert tf is not None
    np.testing.assert_allclose(run_tflite((ART / "model.tflite").read_bytes(), inputs), outputs)
    # Int8 weights vs the float model: close, and the same top-1 on nearly every window.
    ref = np.load(ART / "float_outputs.npy")
    assert np.abs(outputs - ref).max() < 0.1
    assert (outputs.argmax(1) == ref.argmax(1)).mean() >= 0.95


def test_golden_raw_reproduces_inputs():
    """SPEC 9.2 item 4: raw stream -> align -> bandpass -> windows -> z-score == inputs."""
    from haptic_ai import preprocess
    from haptic_ai.windows import make_windows

    meta = json.loads((ART / "model_meta.json").read_text())
    raw = np.load(ART / "golden" / "raw.npy").astype(np.float64)
    fs = meta["sample_rate_hz"]
    acc, gyr = preprocess.gravity_align(raw[:, :3], raw[:, 3:], fs)
    x = preprocess.bandpass(np.hstack([acc, gyr]), fs).astype(np.float32)
    size = round(meta["window_size_s"] * fs)
    xw, _ = make_windows(x, np.zeros(len(x), np.int8), size, round(meta["window_stride_s"] * fs))
    p = meta["preprocessing"]
    z = (xw - np.array(p["norm_mean"])) / np.array(p["norm_std"])
    np.testing.assert_allclose(z, np.load(ART / "golden" / "inputs.npy"), atol=1e-4)
