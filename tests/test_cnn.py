"""SPEC 8.3 tensor contract and 4.3 size budget for the CNN. Needs TF (Linux only, D2)."""

import pytest

tf = pytest.importorskip("tensorflow")
from haptic_ai.models import cnn1d  # noqa: E402


def test_cnn_matches_tensor_contract_and_budget():
    model = cnn1d.build(150)
    assert model.input_shape == (None, 150, 6)
    assert model.output_shape == (None, 7)
    assert model.count_params() < 150_000  # int8: ~1 byte per weight, SPEC 4.3 is 150 KB
