"""model_meta.json JSON Schema is well-formed and enforces the frozen contract (SPEC M0 / 8.2)."""

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from scripts.make_stub_model import model_meta

SCHEMA_PATH = Path(__file__).parents[1] / "docs" / "model_meta.schema.json"


@pytest.fixture(scope="module")
def schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


@pytest.fixture(scope="module")
def validator(schema) -> Draft202012Validator:
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def test_stub_meta_is_valid(validator):
    validator.validate(model_meta("deadbeef"))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: m.__setitem__("sample_rate_hz", 100),
        lambda m: m.__setitem__("spec_version", "2.0"),
        lambda m: m["labels"].reverse(),
        lambda m: m["labels"].append("EXTRA"),
        lambda m: m["output"].__setitem__("shape", [1, 6]),
        lambda m: m["channels"].__setitem__(0, "gx"),
        lambda m: m["training"].pop("commit"),
        lambda m: m.__setitem__("unexpected_key", 1),
    ],
)
def test_rejects_broken_meta(validator, mutate):
    m = copy.deepcopy(model_meta("deadbeef"))
    mutate(m)
    with pytest.raises(ValidationError):
        validator.validate(m)
