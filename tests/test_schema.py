"""Schema validator tests (SPEC M0 / 8.1)."""

from pathlib import Path

import pandas as pd
import pytest

from haptic_ai import schema
from haptic_ai.schema import CANONICAL_COLUMNS, SchemaError, coerce_dtypes, validate

FIXTURE = Path(__file__).parent / "fixtures" / "canonical_100rows.csv"


def test_labels_frozen_ordering():
    # SPEC 1.3 — frozen. Index order is the model output tensor order (SPEC 8.3).
    assert schema.LABELS == (
        "NULL",
        "PALM_TO_PALM",
        "BACK_OF_HAND",
        "INTERLACED_FINGERS",
        "THUMBS",
        "FINGERTIPS",
        "WASH_OTHER",
    )


def test_fixture_is_valid():
    df = schema.read_canonical_csv(FIXTURE)
    assert len(df) == 100
    assert list(df.columns) == list(CANONICAL_COLUMNS)


@pytest.fixture
def good() -> pd.DataFrame:
    return schema.read_canonical_csv(FIXTURE)


def test_coerce_produces_canonical_dtypes(good):
    raw = pd.read_csv(FIXTURE)
    out = coerce_dtypes(raw)
    assert {c: str(out[c].dtype) for c in out.columns} == schema.CANONICAL_DTYPES


def test_reject_wrong_column_order(good):
    swapped = good[["ax", "timestamp_ns", *CANONICAL_COLUMNS[2:]]]
    with pytest.raises(SchemaError, match="columns must be exactly"):
        validate(swapped)


def test_reject_wrong_dtype(good):
    good["label"] = good["label"].astype("int64")
    with pytest.raises(SchemaError, match="dtype must be int8"):
        validate(good)


def test_reject_null(good):
    good.loc[0, "ax"] = float("nan")
    with pytest.raises(SchemaError, match="null values"):
        validate(good)


def test_reject_out_of_range_label(good):
    good.loc[0, "label"] = 9
    with pytest.raises(SchemaError, match="label must be one of"):
        validate(good)


def test_reject_bad_wrist(good):
    good["wrist"] = good["wrist"].where(good.index != 0, "sideways")
    with pytest.raises(SchemaError, match="wrist must be in"):
        validate(good)


def test_reject_unknown_source(good):
    good["source"] = "made_up"
    good["subject_id"] = "made_up_sub01"
    with pytest.raises(SchemaError, match="source must be a dataset key"):
        validate(good)


def test_reject_subject_id_prefix_mismatch(good):
    good["subject_id"] = good["subject_id"].where(good.index != 0, "ocdetect_sub01")
    with pytest.raises(SchemaError, match="subject_id must be"):
        validate(good)


def test_reject_duplicate_triple(good):
    dup = pd.concat([good, good.iloc[[0]]], ignore_index=True)
    with pytest.raises(SchemaError, match="duplicate"):
        validate(dup)


def test_reject_non_monotonic_timestamp(good):
    good.loc[5, "timestamp_ns"] = 10_000_000  # < row 4, not a duplicate of any row
    with pytest.raises(SchemaError, match="monotonic"):
        validate(good)
