"""SPEC M1 acceptance: gravity alignment, mirroring, causality, window purity, gap split."""

import numpy as np
import pandas as pd

from haptic_ai import preprocess
from haptic_ai.windows import make_windows


def quat_to_matrix(w, x, y, z):
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def test_gravity_align_recovers_rotated_signal():
    # Aligned-frame signal: gravity on +z plus small 10 Hz motion, 60 s at 50 Hz.
    t = np.arange(3000) / 50
    motion = 0.2 * np.stack([np.sin(2 * np.pi * 10 * t), np.cos(2 * np.pi * 10 * t), 0 * t], 1)
    acc = motion + [0, 0, preprocess.G]
    gyr = np.stack([np.sin(t), np.cos(t), 0.5 * np.sin(3 * t)], 1)
    # Known quaternion with its axis in the xy-plane (heading is free, SPEC 8.7): 70°.
    axis = np.array([0.6, 0.8, 0.0])
    h = np.deg2rad(70) / 2
    r = quat_to_matrix(np.cos(h), *(np.sin(h) * axis))

    a2, g2 = preprocess.gravity_align(acc @ r.T, gyr @ r.T)

    settled = slice(1500, None)  # the 0.3 Hz low-pass starts from zero state
    assert np.abs(a2[settled] - acc[settled]).max() < 1e-3
    assert np.abs(g2[settled] - gyr[settled]).max() < 1e-3


def test_gravity_align_upside_down_is_finite():
    acc = np.tile([0.0, 0.0, -preprocess.G], (500, 1))
    a2, _ = preprocess.gravity_align(acc, acc)
    assert np.allclose(a2[-1], [0, 0, preprocess.G])


def test_mirror_twice_is_identity():
    rng = np.random.default_rng(0)
    acc, gyr = rng.normal(size=(100, 3)), rng.normal(size=(100, 3))
    a1, g1 = preprocess.mirror(acc, gyr)
    a2, g2 = preprocess.mirror(a1, g1)
    assert not np.allclose(a1, acc)
    assert np.array_equal(a2, acc) and np.array_equal(g2, gyr)


def test_filters_are_causal():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(1000, 3)) + [0, 0, 9.8]
    y = x.copy()
    t = 600
    y[t + 1 :] = rng.normal(size=(1000 - t - 1, 3)) * 50
    assert np.array_equal(preprocess.bandpass(x)[: t + 1], preprocess.bandpass(y)[: t + 1])
    ax, gx = preprocess.gravity_align(x, x)
    ay, gy = preprocess.gravity_align(y, y)
    assert np.array_equal(ax[: t + 1], ay[: t + 1]) and np.array_equal(gx[: t + 1], gy[: t + 1])


def test_window_purity_rule():
    x = np.zeros((10, 6))
    labels = np.array([1] * 6 + [2] * 4 + [3] * 10 + [4] * 5 + [5] * 5, dtype=np.int8)
    _, y = make_windows(np.zeros((30, 6)), labels, size=10, stride=10)
    # 60 % label 1 -> 1 (inclusive bound); 100 % label 3 -> 3; 50/50 -> -1.
    assert y.tolist() == [1, 3, -1]
    _, y = make_windows(x, np.array([6] * 5 + [-1] * 5, dtype=np.int8), size=10, stride=5)
    assert y.tolist() == [-1]  # half unlabelled: no label reaches 60 %
    xw, y = make_windows(np.arange(20.0)[:, None].repeat(6, 1), np.zeros(20, np.int8), 10, 5)
    assert xw.shape == (3, 10, 6) and xw[1, 0, 0] == 5 and y.tolist() == [0, 0, 0]
    assert make_windows(np.zeros((5, 6)), np.zeros(5, np.int8), 10, 5)[0].shape == (0, 10, 6)


def test_split_sessions_at_gaps():
    ms = 1_000_000
    df = pd.DataFrame(
        {
            "timestamp_ns": np.array([0, 20, 40, 141, 161, 261], dtype="int64") * ms,
            "subject_id": "uwash_a_1",
            "session_id": "a_1",
        }
    )
    out = preprocess.split_sessions(df)
    # 40 -> 141 is 101 ms (break); 161 -> 261 is exactly 100 ms (no break).
    assert out["session_id"].tolist() == ["a_1_t0"] * 3 + ["a_1_t141"] * 3
    assert (out["timestamp_ns"] // ms).tolist() == [0, 20, 40, 0, 20, 120]


def test_resample_halves_rate():
    x = np.sin(2 * np.pi * 2 * np.arange(1000) / 100)[:, None]
    y = preprocess.resample(x, 100, 50)
    assert y.shape == (500, 1)
    ref = np.sin(2 * np.pi * 2 * np.arange(500) / 50)
    assert np.abs(y[50:-50, 0] - ref[50:-50]).max() < 1e-2  # 2 Hz passes the anti-alias filter


def test_augment_rotation_preserves_norm_and_scale_scales():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(8, 100, 6)).astype(np.float32)
    rot = preprocess.augment(x, np.random.default_rng(1), scale=(1, 1), jitter=0)
    np.testing.assert_allclose(
        np.linalg.norm(rot[..., :3], axis=2), np.linalg.norm(x[..., :3], axis=2), rtol=1e-5
    )
    assert not np.allclose(rot, x)
    big = preprocess.augment(x, np.random.default_rng(1), rotate_deg=0, scale=(2, 2), jitter=0)
    np.testing.assert_allclose(big, 2 * x, rtol=1e-6)
