"""Adapter for the UWash dataset (DECISIONS D14).

One CSV per subject, ``{location}_{n}.csv``, columns
``acc_x,acc_y,acc_z,gyr_x,gyr_y,gyr_z,timestamp,label``: acc in m/s², gyro in °/s,
timestamp in epoch ms, ~50 Hz native (no resampling needed).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from haptic_ai import schema

from .base import DatasetAdapter

# uwash gesture label -> canonical id (D14). Label 0 is handled by the edge-zone rule.
LABEL_MAP = {1: 1, 2: 2, 3: 2, 4: 3, 5: 6, 6: 4, 7: 4, 8: 5, 9: 5}
# Label-0 samples this close to a gesture are wet/soap/rinse, not NULL -> UNLABELLED.
# ponytail: fixed 5 s guess, tune against the M1 plots (D14).
EDGE_ZONE_S = 5.0


def map_labels(t_ms: np.ndarray, raw: np.ndarray, edge_s: float = EDGE_ZONE_S) -> np.ndarray:
    """Map uwash labels to canonical ids; label 0 -> NULL, or -1 inside the edge zone."""
    out = np.zeros(len(raw), dtype=np.int8)
    gesture = raw != 0
    out[gesture] = [LABEL_MAP[int(v)] for v in raw[gesture]]
    g_t = t_ms[gesture]
    if len(g_t):
        i = np.searchsorted(g_t, t_ms)
        prev = np.abs(t_ms - g_t[np.clip(i - 1, 0, len(g_t) - 1)])
        nxt = np.abs(g_t[np.clip(i, 0, len(g_t) - 1)] - t_ms)
        near = np.minimum(prev, nxt) <= edge_s * 1000
        out[~gesture & near] = schema.UNLABELLED
    return out


def load_file(path: Path) -> pd.DataFrame:
    """Load one uwash subject file as a canonical DataFrame."""
    d = pd.read_csv(path)
    # 9 files have out-of-order rows and ~1.6 % of rows repeat a timestamp: sort, keep first.
    d = d.sort_values("timestamp", kind="stable").drop_duplicates("timestamp")
    t = d["timestamp"].to_numpy()
    local = path.stem  # e.g. canteen_3
    df = pd.DataFrame(
        {
            "timestamp_ns": np.round((t - t[0]) * 1e6).astype("int64"),
            "ax": d["acc_x"],
            "ay": d["acc_y"],
            "az": d["acc_z"],
            # Gyro peaks near ±600: °/s, not rad/s (D14).
            "gx": np.deg2rad(d["gyr_x"]),
            "gy": np.deg2rad(d["gyr_y"]),
            "gz": np.deg2rad(d["gyr_z"]),
            "label": map_labels(t, d["label"].to_numpy().astype(int)),
            "subject_id": f"uwash_{local}",
            "session_id": local,
            "wrist": "unknown",  # the paper doesn't say which wrist
            "source": "uwash",
        }
    )
    # ponytail: axes passed through as-is. Tizen shares Android's frame (az ≈ +g screen-up);
    # confirm in the D10 axis table at M1.
    # ponytail: gaps (max ~2.8 s) are left in place; preprocess splits sessions at > 100 ms.
    return schema.validate(schema.coerce_dtypes(df))


class UwashAdapter(DatasetAdapter):
    """Ingest all uwash subject files under ``root``."""

    def __init__(self, root: Path = Path("data/uwash")):
        self.root = Path(root)

    def ingest(self) -> pd.DataFrame:
        files = sorted(self.root.glob("*_*.csv"))
        if not files:
            raise FileNotFoundError(f"no uwash CSVs in {self.root}")
        return pd.concat([load_file(f) for f in files], ignore_index=True)
