"""M1 corpus: ingest every M1 dataset into data/processed/, then report on it."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from haptic_ai import preprocess, schema
from haptic_ai.ingest.uwash import UwashAdapter
from haptic_ai.ingest.zhang_who import ZhangWhoAdapter

PROCESSED = Path("data/processed")
REPORTS = Path("reports")
# ponytail: ablutomania is missing. It has no step labels (D15) and isn't downloaded yet
# (python scripts/fetch_data.py ablutomania). Add its adapter here once the files are here.
ADAPTERS = {"zhang_who": ZhangWhoAdapter, "uwash": UwashAdapter}


def build(out: Path = PROCESSED) -> pd.DataFrame:
    """Ingest, split sessions at gaps, validate the whole corpus, write partitioned Parquet."""
    df = pd.concat([a().ingest() for a in ADAPTERS.values()], ignore_index=True)
    df = schema.validate(preprocess.split_sessions(df))
    if out.exists():
        # Stale partitions would otherwise survive a rebuild.
        for p in sorted(out.rglob("*"), reverse=True):
            p.unlink() if p.is_file() else p.rmdir()
    df.to_parquet(out, partition_cols=["subject_id"], index=False)
    return df


def load(path: Path = PROCESSED) -> pd.DataFrame:
    """Read the corpus back with canonical column order and dtypes, then validate it."""
    df = pd.read_parquet(path)
    df["subject_id"] = df["subject_id"].astype(str)
    return schema.validate(schema.coerce_dtypes(df))


def stats(df: pd.DataFrame) -> dict:
    """Corpus statistics for reports/M1_corpus.json."""
    key = ["subject_id", "session_id"]
    dt_ms = df.groupby(key, sort=False)["timestamp_ns"].diff().dropna() / 1e6
    src = df.loc[dt_ms.index, "source"]
    s = {"rows": len(df), "subjects": int(df["subject_id"].nunique()), "datasets": {}}
    for name, d in df.groupby("source"):
        dts = dt_ms[src == name]
        lens = d.groupby(key).size() / schema.NOMINAL_RATE_HZ
        labels = d["label"].value_counts().sort_index()
        s["datasets"][name] = {
            "rows": len(d),
            "subjects": int(d["subject_id"].nunique()),
            "sessions": len(lens),
            # Splits at gaps (sessions that don't start at the source's t = 0).
            "gap_breaks": int((~lens.index.get_level_values(1).str.endswith("_t0")).sum()),
            "session_s": {"min": lens.min(), "median": lens.median(), "max": lens.max()},
            "per_subject_rows": d["subject_id"].value_counts().sort_index().to_dict(),
            "label_duration_s": {
                (schema.LABELS[k] if k >= 0 else "UNLABELLED"): n / schema.NOMINAL_RATE_HZ
                for k, n in labels.items()
            },
            "wrist_rows": d["wrist"].value_counts().to_dict(),
            "dt_ms": {
                "median": dts.median(),
                "std": dts.std(),
                "p1": dts.quantile(0.01),
                "p99": dts.quantile(0.99),
                "max": dts.max(),
            },
        }
    return json.loads(json.dumps(s, default=float))  # numpy scalars -> plain JSON


def plot_washes(df: pd.DataFrame, out: Path, per_dataset: int = 4) -> list[str]:
    """|acc|, |gyro| and the label track for a spread of labelled sessions (>= 30 s) per dataset."""
    names = []
    for _, d in df.groupby("source"):
        groups = d.groupby(["subject_id", "session_id"])
        # Whole washes only: short gap fragments are useless to eyeball.
        sessions = [
            g for _, g in groups if (g["label"] > 0).any() and len(g) >= 30 * schema.NOMINAL_RATE_HZ
        ]
        for g in sessions[:: max(1, len(sessions) // per_dataset)][:per_dataset]:
            t = g["timestamp_ns"].to_numpy() / 1e9
            fig, ax = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
            ax[0].plot(t, np.linalg.norm(g[["ax", "ay", "az"]], axis=1), lw=0.5)
            ax[0].set_ylabel("|acc| m/s²")
            ax[1].plot(t, np.linalg.norm(g[["gx", "gy", "gz"]], axis=1), lw=0.5)
            ax[1].set_ylabel("|gyro| rad/s")
            ax[2].plot(t, g["label"], drawstyle="steps-post")
            ax[2].set_yticks(
                range(-1, len(schema.LABELS)), ["UNLABELLED", *schema.LABELS], fontsize=7
            )
            ax[2].set_xlabel("s")
            sid, ses = g["subject_id"].iat[0], g["session_id"].iat[0]
            fig.suptitle(f"{sid} / {ses}")
            fname = f"M1_{sid}_{ses}.png"
            fig.savefig(out / fname, dpi=80)
            plt.close(fig)
            names.append(fname)
    return names


def report(df: pd.DataFrame, out: Path = REPORTS) -> dict:
    """Write reports/M1_corpus.json, reports/M1_corpus.md and wash plots."""
    s = stats(df)
    for p in (out / "figures").glob("M1_*.png"):
        p.unlink()
    (out / "figures").mkdir(parents=True, exist_ok=True)
    figs = plot_washes(df, out / "figures")
    (out / "M1_corpus.json").write_text(json.dumps(s, indent=2) + "\n")
    lines = [
        "# M1 corpus report",
        "",
        "Generated by `make report-corpus` from `data/processed/`. Every row passes",
        "`schema.validate`. Sessions split at gaps over 100 ms. uwash is first re-gridded to",
        "50 Hz with dropouts up to 200 ms interpolated, so its effective split threshold is",
        "200 ms (D18). Session ids end in `_t<ms>`, the start offset. Full numbers are in",
        "`M1_corpus.json`.",
        "",
        f"**{s['rows']:,} rows, {s['subjects']} subjects.**",
        "",
        "| dataset | subjects | sessions | rows | session s (min / median / max) |",
        "|---|---|---|---|---|",
    ]
    for name, d in s["datasets"].items():
        ss = d["session_s"]
        lines.append(
            f"| `{name}` | {d['subjects']} | {d['sessions']} | {d['rows']:,} "
            f"| {ss['min']:.1f} / {ss['median']:.1f} / {ss['max']:.1f} |"
        )
    lines += ["", "## Label duration (s)", "", "| label | " + " | ".join(s["datasets"]) + " |"]
    lines.append("|---" * (len(s["datasets"]) + 1) + "|")
    for lab in ["UNLABELLED", *schema.LABELS]:
        row = [f"{d['label_duration_s'].get(lab, 0):,.0f}" for d in s["datasets"].values()]
        lines.append(f"| `{lab}` | " + " | ".join(row) + " |")
    lines += ["", "## Wrist and sample timing", ""]
    lines += ["| dataset | wrist rows | dt median ms | dt std | dt p1 / p99 | dt max |"]
    lines += ["|---|---|---|---|---|---|"]
    for name, d in s["datasets"].items():
        t = d["dt_ms"]
        lines.append(
            f"| `{name}` | {d['wrist_rows']} | {t['median']:.1f} | {t['std']:.1f} "
            f"| {t['p1']:.1f} / {t['p99']:.1f} | {t['max']:.1f} |"
        )
    lines += ["", "## Washes to eyeball", "", "Human review (SPEC M1): check each plot against"]
    lines += ["the source paper's description.", ""]
    lines += [f"![{f}](figures/{f})" for f in figs]
    (out / "M1_corpus.md").write_text("\n".join(lines) + "\n")
    return s
