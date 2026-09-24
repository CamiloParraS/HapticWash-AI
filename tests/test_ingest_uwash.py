import numpy as np
import pandas as pd

from haptic_ai.ingest.uwash import load_file, regrid

HEADER = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z", "timestamp", "label"]


def test_load_file(tmp_path):
    # 20 s at 50 Hz: label 0 for 10 s, then gesture 3 (back of hand) for 10 s.
    t = 1_625_826_105_000 + np.arange(1000) * 20.0
    lab = np.where(np.arange(1000) < 500, 0, 3)
    rows = pd.DataFrame({h: 0.0 for h in HEADER[:6]} | {"timestamp": t, "label": lab})
    rows["gyr_x"] = 180.0
    rows = pd.concat([rows.iloc[[5]], rows]).iloc[::-1]  # a duplicate and reversed order
    path = tmp_path / "canteen_3.csv"
    rows.to_csv(path, index=False)

    df = load_file(path)  # validate() inside would raise on dupes/disorder

    assert len(df) == 1000
    assert df["subject_id"].iat[0] == "uwash_canteen_3"
    assert df["timestamp_ns"].iat[0] == 0
    assert np.allclose(df["gx"], np.pi)
    # 0-5 s is NULL; 5-10 s is the edge zone before the gesture; then BACK_OF_HAND.
    assert set(df["label"].iloc[:249]) == {0}
    assert set(df["label"].iloc[251:500]) == {-1}
    assert set(df["label"].iloc[500:]) == {2}


def test_regrid():
    # Jittered 50 Hz with a 150 ms dropout (bridged) and a 1 s gap (kept).
    t = np.r_[0, 21, 39, 60, 210, 230, 1230, 1250.0]
    g, x = regrid(t, t[:, None] * 2)
    assert g.tolist() == [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 1230, 1250]
    assert np.allclose(x[:, 0], g * 2)  # linear signal survives interpolation
