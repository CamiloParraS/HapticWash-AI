"""Adapter for the UWash dataset (DECISIONS D14).

One CSV per subject, ``{location}_{n}.csv``, columns
``acc_x,acc_y,acc_z,gyr_x,gyr_y,gyr_z,timestamp,label``: acc in m/s², gyro in °/s,
timestamp in epoch ms, ~50 Hz native but jittery with short dropouts. Re-gridded to 50 Hz
at ingest (D18).
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
# Dropouts up to this long are interpolated; longer ones stay gaps and split the session (D18).
# ponytail: 200 ms bridges ~70 % of the gaps, tune if M2 shows interpolation artefacts.
MAX_BRIDGE_MS = 200.0
# Two recordings each hold two subjects; each file labels only its own washes, so the other
# subject's washes read as label 0. Keep each subject's half: seconds from file start (D18).
SHARED_CROP_S = {
    "library_6": (0, 431),
    "library_7": (431, None),
    "library_9": (0, 381),
    "library_10": (381, None),
}


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


def regrid(t_ms: np.ndarray, x: np.ndarray, max_bridge_ms: float = MAX_BRIDGE_MS):
    """Linear-interpolate ``x`` onto a 20 ms grid. Gaps > ``max_bridge_ms`` get no grid points."""
    brk = np.flatnonzero(np.diff(t_ms) > max_bridge_ms)
    starts, ends = t_ms[np.r_[0, brk + 1]], t_ms[np.r_[brk, len(t_ms) - 1]]
    step = 1000 / schema.NOMINAL_RATE_HZ
    # Count points, don't arange floats: epoch-ms values are too large for a 1e-6 tolerance.
    n = np.floor((ends - starts) / step + 1e-6).astype(int) + 1
    g = np.concatenate([a + np.arange(k) * step for a, k in zip(starts, n, strict=True)])
    return g, np.column_stack([np.interp(g, t_ms, c) for c in x.T])


def load_file(path: Path) -> pd.DataFrame:
    """Load one uwash subject file as a canonical DataFrame."""
    d = pd.read_csv(path)
    # 9 files have out-of-order rows and ~1.6 % of rows repeat a timestamp: sort, keep first.
    d = d.sort_values("timestamp", kind="stable").drop_duplicates("timestamp")
    local = path.stem  # e.g. canteen_3
    if local in SHARED_CROP_S:
        lo, hi = SHARED_CROP_S[local]
        s = (d["timestamp"] - d["timestamp"].iat[0]) / 1000
        d = d[(s >= lo) & (s < (hi if hi is not None else np.inf))]
    t = d["timestamp"].to_numpy()
    labels = map_labels(t, d["label"].to_numpy().astype(int))
    cols = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
    g, x = regrid(t, d[cols].to_numpy())
    df = pd.DataFrame(
        {
            "timestamp_ns": np.round((g - g[0]) * 1e6).astype("int64"),
            "ax": x[:, 0],
            "ay": x[:, 1],
            "az": x[:, 2],
            # Gyro peaks near ±600: °/s, not rad/s (D14).
            "gx": np.deg2rad(x[:, 3]),
            "gy": np.deg2rad(x[:, 4]),
            "gz": np.deg2rad(x[:, 5]),
            "label": labels[np.searchsorted(t, g, "right") - 1],  # last raw sample at or before
            "subject_id": f"uwash_{local}",
            "session_id": local,
            "wrist": "unknown",  # the paper doesn't say which wrist
            "source": "uwash",
        }
    )
    # ponytail: axes passed through as-is. Tizen shares Android's frame (az ≈ +g screen-up);
    # confirm in the D10 axis table at M1.
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
