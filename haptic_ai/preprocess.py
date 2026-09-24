"""Preprocessing (SPEC 8.7, M1): resample, units, gravity-align, bandpass, mirror.

Order: resample -> units -> gravity-align -> bandpass -> z-score. Ingest adapters do the
first two, so the stored corpus is still raw (SPEC 8.1: acc includes gravity).
Gravity-align, bandpass and mirror act on one continuous session, ``(n, 3)`` arrays,
causally, with filter state starting at zero. The watch runs the same code in Kotlin
and the golden files check both (SPEC 9.2).
"""

from fractions import Fraction

import numpy as np
import pandas as pd
from scipy import signal

G = 9.80665  # m/s² per g
# SPEC M1: a gap > 100 ms is a session break. uwash bridges gaps <= 200 ms at ingest, so its
# effective split threshold is 200 ms (D18).
MAX_GAP_NS = 100_000_000


def resample(x: np.ndarray, fs_in: float, fs_out: float = 50) -> np.ndarray:
    """Anti-aliased polyphase resampling along axis 0 (never plain decimation)."""
    r = Fraction(fs_out / fs_in).limit_denominator(1000)
    return signal.resample_poly(x, r.numerator, r.denominator, axis=0)


def split_sessions(df: pd.DataFrame, max_gap_ns: int = MAX_GAP_NS) -> pd.DataFrame:
    """Start a new session at every gap > ``max_gap_ns``; time restarts at 0.

    Ids get ``_t<ms>``, the piece's start in ms from the source session's start, so they stay
    the same across runs unless that piece's own start moves (D18).
    """
    key = ["subject_id", "session_id"]
    brk = df.groupby(key, sort=False)["timestamp_ns"].diff() > max_gap_ns
    k = brk.groupby([df[c] for c in key], sort=False).cumsum()
    out = df.copy()
    t = df["timestamp_ns"] - df.groupby(key, sort=False)["timestamp_ns"].transform("first")
    t0 = t.groupby([df[c] for c in key] + [k], sort=False).transform("first")
    out["session_id"] = df["session_id"] + "_t" + (t0 // 1_000_000).astype(str)
    out["timestamp_ns"] = t - t0
    return out


def gravity_align(
    acc: np.ndarray, gyr: np.ndarray, fs: float = 50, lowpass_hz: float = 0.3
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate each sample by the minimal rotation taking gravity to +z (SPEC 8.7).

    Gravity is a causal 2nd-order Butterworth low-pass of the raw accelerometer. The
    rotation is Rodrigues: R = I + [v]x + [v]x² / (1 + c), with v = u x z, c = u · z.
    Heading (rotation about z) is left free.
    """
    sos = signal.butter(2, lowpass_hz, fs=fs, output="sos")
    g = signal.sosfilt(sos, acc, axis=0)
    norm = np.linalg.norm(g, axis=1, keepdims=True)
    u = np.divide(g, norm, out=np.tile([0.0, 0.0, 1.0], (len(g), 1)), where=norm > 1e-9)
    vx, vy = u[:, 1], -u[:, 0]  # v = u x z = (uy, -ux, 0)
    c = u[:, 2]
    zero = np.zeros_like(c)
    k = np.stack(  # [v]x, skew matrix of v
        [
            np.stack([zero, zero, vy], 1),
            np.stack([zero, zero, -vx], 1),
            np.stack([-vy, vx, zero], 1),
        ],
        1,
    )
    # ponytail: gravity exactly on -z (c = -1) is singular; we use 180° about x.
    # The Kotlin side must match (SPEC 9.2 golden files would catch a mismatch).
    flip = c < -1 + 1e-9
    scale = np.where(flip, 0.0, 1.0 / np.where(flip, 1.0, 1.0 + c))
    r = np.eye(3) + k + (k @ k) * scale[:, None, None]
    r[flip] = np.diag([1.0, -1.0, -1.0])
    return np.einsum("nij,nj->ni", r, acc), np.einsum("nij,nj->ni", r, gyr)


def bandpass(
    x: np.ndarray, fs: float = 50, low_hz: float = 1.0, high_hz: float = 18.0, order: int = 3
) -> np.ndarray:
    """Causal Butterworth bandpass along axis 0, zero initial state (never filtfilt)."""
    sos = signal.butter(order, [low_hz, high_hz], btype="band", fs=fs, output="sos")
    return signal.sosfilt(sos, x, axis=0)


def mirror(acc: np.ndarray, gyr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Left<->right wrist augmentation: reflect across the plane normal to x.

    Acceleration is a polar vector (x flips). Angular velocity is an axial vector, so under
    a reflection it becomes -M·w (y and z flip). Applying this twice gives the original.
    """
    # ponytail: assumes x is the across-the-wrist axis (Android watch frame). Check it
    # against the axis-frame table in docs/label_mapping.md before M2 relies on it.
    return acc * [-1.0, 1.0, 1.0], gyr * [1.0, -1.0, -1.0]
