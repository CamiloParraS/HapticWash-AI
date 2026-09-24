"""Adapter for zhang_who PART 1: 30 WHO-guided washes (DECISIONS D5, D15: test set only).

Byteflies, both wrists, 100 Hz. One integer per line per axis (``RAW/ACM_{X,Y,Z}_{L,R}<n>``,
``GRAW/GYR_...``), with a 2-line header. Washes 1-3 are participant 1, 4-6 participant 2,
and so on. Steps are in ``FRAME/FrameACM_<n>.xls``: ``Action | Start | # | End | #``, where
``#`` is the sample index at 100 Hz. Scaling is from the dataset's 00_Readme.txt.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from haptic_ai import preprocess, schema

from .base import DatasetAdapter

FS = 100
ACC_PER_G = 4096  # Real_ACM = ACM / 4096 (g)
# Readme says 16.384 (±2000 °/s); the data fits 131 (±250 °/s), see D17.
GYR_PER_DPS = 131
# zhang Action code -> canonical id (D5). Samples outside every step stay -1.
# Wet/soap/rinse (0, 0.5, 7) -> -1, as in uwash's edge zone (D18).
LABEL_MAP = {
    0: -1,
    0.5: -1,
    1: 1,
    2.1: 2,
    2.2: 2,
    3: 3,
    4: 6,
    5.1: 4,
    5.2: 4,
    6.1: 5,
    6.2: 5,
    7: -1,
}
WRISTS = {"L": "left", "R": "right"}


def read_axis(path: Path) -> np.ndarray:
    return np.loadtxt(path, skiprows=2)


def label_samples(n: int, frame: pd.DataFrame) -> np.ndarray:
    """Per-sample canonical labels at 100 Hz from a raw FrameACM sheet (no header row)."""
    action = pd.to_numeric(frame[0], errors="coerce")
    rows = frame[action.notna()]
    out = np.full(n, schema.UNLABELLED, dtype=np.int8)
    for a, start, end in zip(action.dropna(), rows[2], rows[4], strict=True):
        # Bounds are inclusive; a step that starts on the previous one's end sample wins it.
        out[int(start) : int(end) + 1] = LABEL_MAP[round(float(a), 1)]
    return out


def load_wash(files: dict[str, Path], n: int, w: str) -> pd.DataFrame:
    """One wash, one wrist, as a canonical frame at 50 Hz."""
    acc = np.stack([read_axis(files[f"ACM_{a}_{w}{n}.csv"]) for a in "XYZ"], 1)
    gyr = np.stack([read_axis(files[f"GYR_{a}_{w}{n}.csv"]) for a in "XYZ"], 1)
    frame = pd.read_excel(files[f"FrameACM_{n}.xls"], header=None)
    # End at the last annotated sample: wash 16's right wrist runs ~96 s past the rinse (D18).
    end = int(pd.to_numeric(frame[4], errors="coerce").max()) + 1
    m = min(len(acc), len(gyr), end)
    labels = label_samples(m, frame)
    x = preprocess.resample(np.hstack([acc[:m], gyr[:m]]), FS, schema.NOMINAL_RATE_HZ)
    y = labels[:: FS // schema.NOMINAL_RATE_HZ]  # the sample each 50 Hz output sits on
    acc50 = x[:, :3] / ACC_PER_G * preprocess.G
    gyr50 = np.deg2rad(x[:, 3:] / GYR_PER_DPS)
    # ponytail: Byteflies axes passed through as-is; the Android mapping is unverified.
    # See the axis-frame table in docs/label_mapping.md (D10).
    df = pd.DataFrame(np.hstack([acc50, gyr50]), columns=["ax", "ay", "az", "gx", "gy", "gz"])
    df.insert(0, "timestamp_ns", np.arange(len(df), dtype="int64") * 20_000_000)
    df["label"] = y
    df["subject_id"] = f"zhang_who_{(n - 1) // 3 + 1}"
    df["session_id"] = f"{(n - 1) % 3 + 1}_{WRISTS[w]}"
    df["wrist"] = WRISTS[w]
    df["source"] = "zhang_who"
    return schema.validate(schema.coerce_dtypes(df))


class ZhangWhoAdapter(DatasetAdapter):
    """Ingest the 30 PART 1 washes found anywhere under ``root``."""

    def __init__(self, root: Path = Path("data/zhang-who")):
        self.root = Path(root)

    def ingest(self) -> pd.DataFrame:
        # fetch_data.py and a manual unzip nest the folders differently, so find by name.
        files = {p.name: p for p in self.root.rglob("*") if p.suffix in (".csv", ".xls")}
        if "FrameACM_1.xls" not in files:
            raise FileNotFoundError(f"no zhang_who PART 1 files under {self.root}")
        return pd.concat(
            [load_wash(files, n, w) for n in range(1, 31) for w in WRISTS], ignore_index=True
        )
