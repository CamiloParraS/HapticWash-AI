"""Regenerate canonical_100rows.csv — a schema-valid 100-row fixture (SPEC M0)."""

from pathlib import Path

import numpy as np
import pandas as pd

from haptic_ai.schema import CANONICAL_COLUMNS, coerce_dtypes, validate

OUT = Path(__file__).with_name("canonical_100rows.csv")


def build() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    plan = [("zhang_who_sub01", "s1", "left", 60), ("zhang_who_sub02", "s1", "right", 40)]
    label_cycle = [0, 1, 2, 3, 4, 5, 6, -1]
    for subject_id, session_id, wrist, n in plan:
        for i in range(n):
            rows.append(
                {
                    "timestamp_ns": i * 20_000_000,  # 50 Hz
                    "ax": rng.normal(0, 1),
                    "ay": rng.normal(0, 1),
                    "az": rng.normal(9.81, 0.5),
                    "gx": rng.normal(0, 0.3),
                    "gy": rng.normal(0, 0.3),
                    "gz": rng.normal(0, 0.3),
                    "label": label_cycle[i % len(label_cycle)],
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "wrist": wrist,
                    "source": "zhang_who",
                }
            )
    return coerce_dtypes(pd.DataFrame(rows, columns=list(CANONICAL_COLUMNS)))


if __name__ == "__main__":
    df = validate(build())
    df.to_csv(OUT, index=False, lineterminator="\n", float_format="%.6f")
    print(f"wrote {OUT} ({len(df)} rows)")
