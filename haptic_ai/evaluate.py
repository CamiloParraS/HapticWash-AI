"""LOSO cross-validation, the leak guard, metrics and the M2 report (SPEC M2, 9.4).

Train and select on `uwash` with leave-one-subject-out. `zhang_who` is an external test
set only: fitted once on all of `uwash`, never used for training or selection (D15).
"""

import json
import random
import subprocess
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import confusion_matrix, f1_score

from haptic_ai import corpus, features
from haptic_ai.models import tree
from haptic_ai.schema import LABELS, UNLABELLED
from haptic_ai.windows import session_windows

STEPS = [1, 2, 3, 4, 5]  # SPEC M2: macro-F1 is over the five WHO steps
ALL = list(range(len(LABELS)))


class LeakError(AssertionError):
    """A subject appears in both train and test."""


def check_disjoint(train_subjects: np.ndarray, test_subjects: np.ndarray) -> None:
    leak = set(train_subjects) & set(test_subjects)
    if leak:
        raise LeakError(f"subjects in both train and test: {sorted(leak)}")


def loso_folds(subjects: np.ndarray):
    """Yield ``(train_idx, test_idx)``, one fold per subject, each checked for leaks."""
    for s in sorted(set(subjects)):
        te = subjects == s
        check_disjoint(subjects[~te], subjects[te])
        yield np.flatnonzero(~te), np.flatnonzero(te)


def moving_average(proba: np.ndarray, sessions: np.ndarray, k: int) -> np.ndarray:
    """Causal mean of the last ``k`` window probabilities, restarting at each session."""
    out = np.empty_like(proba)
    for s in dict.fromkeys(sessions):
        i = np.flatnonzero(sessions == s)
        c = np.cumsum(proba[i], 0)
        c[k:] = c[k:] - c[:-k]
        out[i] = c / np.minimum(np.arange(1, len(i) + 1), k)[:, None]
    return out


def macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    return float(f1_score(y, pred, labels=STEPS, average="macro", zero_division=0))


def _predict(model, f_train, y_train, f_test) -> np.ndarray:
    """Fit on labelled windows; return ``(n, 7)`` probabilities for every test window."""
    model.fit(f_train, y_train)
    proba = np.zeros((len(f_test), len(LABELS)))
    proba[:, model.classes_] = model.predict_proba(f_test)
    return proba


def learner(name: str, cfg: dict, seed: int):
    """``fit_predict(x_train, y_train, x_test) -> proba``. Trees take features, CNNs windows."""
    if name.startswith("cnn"):
        from haptic_ai.models import cnn1d  # TF is Linux-only (D2)

        aug = cfg["augment"] if name == "cnn_aug" else None
        return lambda a, b, c: cnn1d.fit_predict(a, b, c, cfg["cnn"], aug, seed)
    return lambda a, b, c: _predict(tree.make(name, seed), a, b, c)


def smoothing_ks(seconds: list[float], stride_s: float) -> dict[str, int]:
    """Causal moving average set in seconds, as a window count at this stride (D21)."""
    return {"raw": 1} | {f"ma_{s:g}s": max(1, round(s / stride_s)) for s in seconds}


def _score(y, proba, sessions, subjects, ks: dict[str, int]) -> dict:
    """Per smoothing: per-subject macro-F1, per-class F1, and summed confusion matrix."""
    out = {}
    m = y != UNLABELLED
    for tag, k in ks.items():
        pred = moving_average(proba, sessions, k).argmax(1)
        per_subject = {
            s: macro_f1(y[m & (subjects == s)], pred[m & (subjects == s)])
            for s in sorted(set(subjects[m]))
        }
        f = np.array(list(per_subject.values()))
        out[tag] = {
            "k": k,
            "macro_f1_mean": float(f.mean()),
            "macro_f1_std": float(f.std()),
            "per_subject": per_subject,
            "per_class_f1": dict(
                zip(
                    LABELS,
                    f1_score(y[m], pred[m], labels=ALL, average=None, zero_division=0),
                    strict=True,
                )
            ),
            "confusion": confusion_matrix(y[m], pred[m], labels=ALL).tolist(),
        }
    return out


def loso(fit_predict, x: np.ndarray, w: dict, ks: dict) -> dict:
    """LOSO over ``w["subject"]``: every window is predicted by the fold that held it out."""
    labelled = w["y"] != UNLABELLED
    proba = np.zeros((len(x), len(LABELS)))
    for tr, te in loso_folds(w["subject"]):
        tr = tr[labelled[tr]]
        proba[te] = fit_predict(x[tr], w["y"][tr], x[te])
    return _score(w["y"], proba, w["session"], w["subject"], ks)


def external(fit_predict, x_tr, w_tr, x_te, w_te, ks: dict) -> dict:
    """Fit on all training windows, score the external set per wrist."""
    check_disjoint(w_tr["subject"], w_te["subject"])
    m = w_tr["y"] != UNLABELLED
    proba = fit_predict(x_tr[m], w_tr["y"][m], x_te)
    out = {}
    for wrist in sorted(set(w_te["wrist"])):
        i = w_te["wrist"] == wrist
        out[wrist] = _score(w_te["y"][i], proba[i], w_te["session"][i], w_te["subject"][i], ks)
    return out


def _git_sha() -> str:
    r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    return r.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


def run(config: str | Path, out: Path = corpus.REPORTS) -> dict:
    """Window-size sweep x models, LOSO on uwash, external test on zhang_who."""
    cfg = yaml.safe_load(Path(config).read_text())
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    df = corpus.load()
    train, test = df[df["source"] == "uwash"], df[df["source"] == "zhang_who"]
    report = {"config": cfg, "commit": _git_sha(), "seed": seed, "results": {}}
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"M2_{Path(config).stem}.json"
    for size in cfg["window_sizes_s"]:
        stride = size * (1 - cfg["overlap"])
        w_tr = session_windows(train, size, stride)
        # Training data is left wrist only (D19); right-wrist test data is mirrored into it.
        w_te = session_windows(test, size, stride, mirror="right")
        ks = smoothing_ks(cfg["moving_average_s"], stride)
        feats = {}
        n = {"train_windows": int((w_tr["y"] != UNLABELLED).sum())}
        for name in cfg["models"]:
            print(f"{size} s  {name}", flush=True)
            if name.startswith("cnn"):
                x_tr, x_te = w_tr["x"], w_te["x"]
            else:
                if not feats:
                    feats = {
                        k: features.extract_features(w["x"])
                        for k, w in (("tr", w_tr), ("te", w_te))
                    }
                x_tr, x_te = feats["tr"], feats["te"]
            fp = learner(name, cfg, seed)
            report["results"][f"{size}s/{name}"] = {
                **n,
                "loso": loso(fp, x_tr, w_tr, ks),
                "zhang_who": external(fp, x_tr, w_tr, x_te, w_te, ks),
            }
            # Save after every block so a killed multi-hour sweep keeps what it finished.
            dest.write_text(json.dumps(report, indent=1))
    return report
