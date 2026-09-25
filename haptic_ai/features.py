"""Hand-rolled window features for the tree baselines (SPEC M2).

Input is preprocessed windows ``(m, W, 6)`` (gravity-aligned, bandpassed). Only the tree
baselines use these. If a tree model ever ships, the Kotlin side must mirror this file.
"""

from itertools import combinations

import numpy as np

CHANNELS = ("ax", "ay", "az", "gx", "gy", "gz")
BANDS = ((1, 3), (3, 6), (6, 12))  # Hz
PER_AXIS = (
    "mean", "std", "min", "max", "rms", "energy", "zcr", "dom_freq", "spec_entropy",
    *(f"bp_{lo}_{hi}" for lo, hi in BANDS),
)  # fmt: skip
# Correlation within each sensor: acc pairs, then gyro pairs.
PAIRS = tuple(combinations(range(3), 2)) + tuple(combinations(range(3, 6), 2))


def names() -> list[str]:
    return [f"{c}_{f}" for c in CHANNELS for f in PER_AXIS] + [
        f"corr_{CHANNELS[i]}_{CHANNELS[j]}" for i, j in PAIRS
    ]


def extract_features(x: np.ndarray, fs: float = 50) -> np.ndarray:
    """``(m, W, c)`` windows -> ``(m, 12c + 6)`` features, ordered as :func:`names`."""
    x = np.asarray(x, np.float64)
    w = x.shape[1]
    mean = x.mean(1)
    xc = x - mean[:, None]
    std = xc.std(1)
    ms = (x**2).mean(1)
    zcr = (np.diff(np.signbit(xc), axis=1) != 0).mean(1)

    psd = np.abs(np.fft.rfft(x, axis=1)) ** 2 / w  # (m, f, c)
    freqs = np.fft.rfftfreq(w, 1 / fs)
    dom = freqs[1:][psd[:, 1:].argmax(1)]  # skip DC
    p = psd[:, 1:] / np.maximum(psd[:, 1:].sum(1, keepdims=True), 1e-12)
    entropy = -(p * np.log(np.where(p > 0, p, 1))).sum(1) / np.log(p.shape[1])
    bands = [psd[:, (freqs >= lo) & (freqs < hi)].sum(1) for lo, hi in BANDS]

    per_axis = np.stack(
        [mean, std, x.min(1), x.max(1), np.sqrt(ms), ms, zcr, dom, entropy, *bands], 2
    )  # (m, c, 12)
    z = xc / np.where(std > 1e-12, std, np.inf)[:, None]  # constant axis -> corr 0
    corr = np.stack([(z[..., i] * z[..., j]).mean(1) for i, j in PAIRS], 1)
    return np.concatenate([per_axis.reshape(len(x), -1), corr], 1).astype(np.float32)
