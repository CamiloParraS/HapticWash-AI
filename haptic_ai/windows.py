"""Sliding windows and the window-label purity rule (SPEC 8.7)."""

import numpy as np

from haptic_ai.schema import LABELS, UNLABELLED

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
