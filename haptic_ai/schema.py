"""Canonical sensor schema and validator (SPEC 8.1).

One queryable, canonical table format shared by processed data, replay files, and
on-device export. A frame that fails :func:`validate` is rejected, never coerced.
Producers that know their output is well-formed call :func:`coerce_dtypes` first.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# --- Frozen label set (SPEC 1.3). Index order defines the model output tensor. ---
LABELS: tuple[str, ...] = (
    "NULL",
    "PALM_TO_PALM",
    "BACK_OF_HAND",
    "INTERLACED_FINGERS",
    "THUMBS",
    "FINGERTIPS",
    "WASH_OTHER",
)
LABEL_IDS: frozenset[int] = frozenset(range(len(LABELS)))
UNLABELLED = -1
NOMINAL_RATE_HZ = 50

# --- Dataset keys (SPEC 5). subject_id must be "<key>_<local_id>". ---
DATASET_KEYS: tuple[str, ...] = (
    "zhang_who",
    "ablutomania",
    "ocdetect",
    "harage",
    "own_phone",
    "own_watch",
)

WRIST_VALUES: frozenset[str] = frozenset(("left", "right", "unknown"))

# --- Column order is fixed as listed (SPEC 8.1). ---
CANONICAL_DTYPES: dict[str, str] = {
    "timestamp_ns": "int64",
    "ax": "float32",
    "ay": "float32",
    "az": "float32",
    "gx": "float32",
    "gy": "float32",
    "gz": "float32",
    "label": "int8",
    "subject_id": "object",
    "session_id": "object",
    "wrist": "object",
    "source": "object",
}
CANONICAL_COLUMNS: tuple[str, ...] = tuple(CANONICAL_DTYPES)
_STRING_COLUMNS = ("subject_id", "session_id", "wrist", "source")
# pandas 2 stores strings as "object"; pandas 3 as "str". Accept both.
_STRING_DTYPES = frozenset(("object", "str", "string"))


class SchemaError(ValueError):
    """Raised when a DataFrame does not conform to the canonical schema."""


def coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with canonical column order and dtypes.

    For trusted producers (ingest adapters, fixtures written by hand). Raises
    ``KeyError`` if a column is missing so a malformed producer fails loudly.
    """
    out = df[list(CANONICAL_COLUMNS)].copy()
    for col, dtype in CANONICAL_DTYPES.items():
        out[col] = out[col].astype(dtype)
    return out


def read_canonical_csv(path: str | Path) -> pd.DataFrame:
    """Read a canonical sensor CSV with canonical dtypes applied, then validate it."""
    df = pd.read_csv(path, dtype=dict(CANONICAL_DTYPES))
    return validate(df[list(CANONICAL_COLUMNS)])


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Validate ``df`` against the canonical schema. Return it unchanged on success.

    Checks, in order: exact columns and order, exact dtypes, no nulls, value
    domains (label, wrist, source, subject_id prefix), no duplicate
    ``(subject_id, session_id, timestamp_ns)`` triples, and per-session
    monotonic non-decreasing ``timestamp_ns``.
    """
    cols = list(df.columns)
    if cols != list(CANONICAL_COLUMNS):
        raise SchemaError(
            f"columns must be exactly {CANONICAL_COLUMNS} in order, got {tuple(cols)}"
        )

    for col, want in CANONICAL_DTYPES.items():
        got = str(df[col].dtype)
        ok = got in _STRING_DTYPES if col in _STRING_COLUMNS else got == want
        if not ok:
            raise SchemaError(
                f"column {col!r}: dtype must be {want}, got {got} (call coerce_dtypes)"
            )

    null_cols = [c for c in CANONICAL_COLUMNS if df[c].isna().any()]
    if null_cols:
        raise SchemaError(f"null values not allowed; found in {null_cols}")

    bad_labels = sorted(set(df["label"].unique()) - LABEL_IDS - {UNLABELLED})
    if bad_labels:
        allowed = sorted(LABEL_IDS) + [UNLABELLED]
        raise SchemaError(f"label must be one of {allowed}, got {bad_labels}")

    bad_wrist = sorted(set(df["wrist"].unique()) - WRIST_VALUES)
    if bad_wrist:
        raise SchemaError(f"wrist must be in {sorted(WRIST_VALUES)}, got {bad_wrist}")

    bad_source = sorted(set(df["source"].unique()) - set(DATASET_KEYS))
    if bad_source:
        raise SchemaError(f"source must be a dataset key {DATASET_KEYS}, got {bad_source}")

    pairs = zip(df["subject_id"], df["source"], strict=True)
    prefix_ok = [s.startswith(f"{src}_") for s, src in pairs]
    if not all(prefix_ok):
        i = prefix_ok.index(False)
        example = df.iloc[i][["subject_id", "source"]].to_dict()
        raise SchemaError(f"subject_id must be '<source>_<local_id>'; offending row: {example}")

    dup = df.duplicated(subset=["subject_id", "session_id", "timestamp_ns"])
    if dup.any():
        raise SchemaError(f"{int(dup.sum())} duplicate (subject_id, session_id, timestamp_ns) rows")

    dt = df.groupby(["subject_id", "session_id"], sort=False)["timestamp_ns"].diff()
    if (dt < 0).any():
        raise SchemaError("timestamp_ns must be monotonic non-decreasing within each session")

    return df
