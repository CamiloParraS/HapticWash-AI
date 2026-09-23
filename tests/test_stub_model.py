"""Stub release artifacts load and match the SPEC 8.3 tensor contract (SPEC M0).

Skipped unless `artifacts/` has been populated by `scripts/make_stub_model.py`
(CI runs it first; it needs the Linux-only `ml` extra).
"""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from haptic_ai.schema import LABELS

ART = Path(__file__).parents[1] / "artifacts"
SCHEMA_PATH = Path(__file__).parents[1] / "docs" / "model_meta.schema.json"

pytestmark = pytest.mark.skipif(
    not (ART / "model.tflite").exists() or not (ART / "model_meta.json").exists(),
    reason="run scripts/make_stub_model.py first (needs the ml extra)",
)


@pytest.fixture(scope="module")
def meta() -> dict:
    return json.loads((ART / "model_meta.json").read_text())


def test_meta_validates_against_schema(meta):
    Draft202012Validator(json.loads(SCHEMA_PATH.read_text())).validate(meta)


def test_tflite_tensor_shapes_match_spec_8_3(meta):
    tf = pytest.importorskip("tensorflow")
    interp = tf.lite.Interpreter(model_path=str(ART / "model.tflite"))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    assert list(inp["shape"]) == [1, 150, 6]  # W = window_size_s * sample_rate_hz
    assert inp["dtype"].__name__ == "float32"
    assert list(out["shape"]) == [1, len(LABELS)] == [1, 7]
    assert out["dtype"].__name__ == "float32"

    assert meta["input"]["shape"] == list(inp["shape"])
    assert meta["output"]["shape"] == list(out["shape"])
    assert meta["labels"] == list(LABELS)
