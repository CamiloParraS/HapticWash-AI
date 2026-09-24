import numpy as np
import pandas as pd

from haptic_ai.ingest.zhang_who import label_samples, load_wash


def frame(rows):
    """A FrameACM sheet as read with header=None: header row, then Action/start/#/end/#."""
    head = [["Action", "Start time", "#", "End time", "#"]]
    return pd.DataFrame(head + [[a, "", s, "", e] for a, s, e in rows])


def test_label_samples():
    y = label_samples(20, frame([(0, 2, 5), (2.1, 5, 8), (4, 10, 12), (6.2, 13, 30)]))
    # 0 wet -> -1 (D18), 2.1 -> BACK_OF_HAND (wins shared sample 5), gap 9 -> -1,
    # 4 -> WASH_OTHER (D5), 6.2 -> FINGERTIPS, clipped at n.
    assert y.tolist() == [-1, -1, -1, -1, -1, 2, 2, 2, 2, -1, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5]


def test_load_wash(tmp_path):
    n = 1000  # 10 s at 100 Hz
    for w in "LR":
        for sensor, folder, raw in (("ACM", "RAW", 4096), ("GYR", "GRAW", 131 * 180)):
            (tmp_path / folder).mkdir(exist_ok=True)
            for a in "XYZ":
                v = np.full(n, raw if a == "Z" else 0)
                header = "#Epoch Timestamp: 1614176552 \n#Sampling Rate: 100 Hz"
                np.savetxt(
                    tmp_path / folder / f"{sensor}_{a}_{w}1.csv",
                    v,
                    "%d",
                    header=header,
                    comments="",
                )
    (tmp_path / "FRAME").mkdir()
    frame([(1, 0, 999)]).to_excel(
        tmp_path / "FRAME" / "FrameACM_1.xls", header=False, index=False, engine="openpyxl"
    )

    files = {p.name: p for p in tmp_path.rglob("*.*")}
    df = pd.concat([load_wash(files, 1, w) for w in "LR"])

    assert len(df) == 2 * 500
    assert set(df["session_id"]) == {"1_left", "1_right"}
    assert set(df["subject_id"]) == {"zhang_who_1"}
    assert set(df["label"]) == {1}
    assert df["timestamp_ns"].iat[1] == 20_000_000
    mid = df.iloc[100:400]  # away from resample edge effects
    assert np.allclose(mid["az"], 9.80665, atol=1e-3)
    assert np.allclose(mid["gz"], np.pi, atol=1e-3)
