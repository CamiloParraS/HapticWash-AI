"""Sliding windows and the window-label purity rule (SPEC 8.7)."""

import numpy as np
import pandas as pd

from haptic_ai import preprocess
from haptic_ai.schema import LABELS, NOMINAL_RATE_HZ, UNLABELLED

PURITY = 0.6  # SPEC 8.7: majority label must cover >= 60 % of the window


def make_windows(
    x: np.ndarray, labels: np.ndarray, size: int, stride: int
) -> tuple[np.ndarray, np.ndarray]:
    """Cut ``(n, c)`` into ``(m, size, c)`` windows. Each gets its majority label if that
    label covers >= 60 % of its samples, else -1. Cut from one session only; windows
    never span a session break.
    """
    if len(x) < size:
        return np.empty((0, size, x.shape[1]), x.dtype), np.empty(0, np.int8)
    xw = np.lib.stride_tricks.sliding_window_view(x, size, axis=0)[::stride]
    lw = np.lib.stride_tricks.sliding_window_view(labels, size)[::stride]
    # Count each label id per window; -1 is shifted to column 0.
    counts = np.stack([(lw == k).sum(1) for k in range(UNLABELLED, len(LABELS))], 1)
    best = counts.argmax(1)
    y = np.where(counts.max(1) >= PURITY * size, best + UNLABELLED, UNLABELLED)
    return xw.transpose(0, 2, 1), y.astype(np.int8)


def session_windows(
    df: pd.DataFrame, size_s: float, stride_s: float, fs: int = NOMINAL_RATE_HZ, mirror=None
) -> dict[str, np.ndarray]:
    """Preprocess each session of a canonical frame and cut its windows (SPEC 8.7).

    Per session: optional wrist mirroring on the raw axes, then gravity-align and bandpass
    with state starting at zero, then windows. Windows labelled -1 are kept so causal
    smoothing sees every window, as on the watch; mask them out before training or
    scoring. ``mirror`` is a wrist value ("right") or None. Returns ``x (m, W, 6)``, ``y``,
    and per-window ``subject``, ``session``, ``wrist``, in time order within a session.
    """
    size, stride = round(size_s * fs), round(stride_s * fs)
    df = df.sort_values(["subject_id", "session_id", "timestamp_ns"])
    out = {k: [] for k in ("x", "y", "subject", "session", "wrist")}
    for (subj, sess), g in df.groupby(["subject_id", "session_id"], sort=False):
        acc = g[["ax", "ay", "az"]].to_numpy(np.float64)
        gyr = g[["gx", "gy", "gz"]].to_numpy(np.float64)
        wrist = g["wrist"].iloc[0]
        if wrist == mirror:
            acc, gyr = preprocess.mirror(acc, gyr)
        acc, gyr = preprocess.gravity_align(acc, gyr, fs)
        x = preprocess.bandpass(np.hstack([acc, gyr]), fs)
        xw, yw = make_windows(x.astype(np.float32), g["label"].to_numpy(), size, stride)
        out["x"].append(xw)
        out["y"].append(yw)
        for k, v in (("subject", subj), ("session", f"{subj}/{sess}"), ("wrist", wrist)):
            out[k].append(np.full(len(yw), v, dtype=object))
    return {k: np.concatenate(v) for k, v in out.items()}
