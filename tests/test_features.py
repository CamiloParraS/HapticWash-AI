"""SPEC M2 feature extraction."""

import numpy as np

from haptic_ai import features


def test_known_sine():
    t = np.arange(150) / 50
    x = np.zeros((1, 150, 6))
    x[0, :, 0] = np.sin(2 * np.pi * 4 * t)  # 4 Hz on ax -> 3-6 Hz band
    x[0, :, 1] = 2 * x[0, :, 0]  # ay perfectly correlated with ax
    f = dict(zip(features.names(), features.extract_features(x)[0], strict=True))
    assert len(f) == features.extract_features(x).shape[1]
    assert abs(f["ax_dom_freq"] - 4.0) < 0.2
    assert abs(f["ax_rms"] - np.sqrt(0.5)) < 1e-2
    assert f["ax_bp_3_6"] > 100 * f["ax_bp_6_12"]
    assert abs(f["corr_ax_ay"] - 1.0) < 1e-6
    assert f["corr_ax_az"] == 0.0  # constant axis: no NaN
    assert np.isfinite(features.extract_features(x)).all()
