"""Train the release model, quantize to TFLite, write model_meta.json and golden files.

SPEC M2, 8.2, 8.3, 9.2. Runs on Linux (TF, D2). The release model trains on every
`uwash` subject except one, the golden subject (D22): its longest session becomes
`golden/raw.npy` and the windows cut from it become `golden/inputs.npy`, so the parity
windows come from a subject the model never saw. `zhang_who` (CC-BY-NC-ND) is used only
for the local quantization check and never written to `artifacts/`.
"""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml

from haptic_ai import corpus, evaluate
from haptic_ai.models import cnn1d
from haptic_ai.schema import LABELS, NOMINAL_RATE_HZ, UNLABELLED
from haptic_ai.windows import session_windows

OUT = Path("artifacts")
CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz"]


def to_tflite(model: tf.keras.Model) -> bytes:
    """Dynamic-range quantization: int8 weights, float32 in/out (SPEC 8.3)."""
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    return conv.convert()


def run_tflite(tflite: bytes, x: np.ndarray) -> np.ndarray:
    """Run ``(m, W, 6)`` z-scored windows one at a time, like the watch does."""
    interp = tf.lite.Interpreter(model_content=tflite)
    interp.allocate_tensors()
    i, o = interp.get_input_details()[0]["index"], interp.get_output_details()[0]["index"]
    out = np.empty((len(x), len(LABELS)), np.float32)
    for n, win in enumerate(x.astype(np.float32)):
        interp.set_tensor(i, win[None])
        interp.invoke()
        out[n] = interp.get_tensor(o)[0]
    return out


def _f1_per_subject(y, pred, subjects) -> float:
    m = y != UNLABELLED
    return float(
        np.mean([evaluate.macro_f1(y[m & (subjects == s)], pred[m & (subjects == s)])
                 for s in sorted(set(subjects[m]))])
    )  # fmt: skip


def export(config: str | Path, out: Path = OUT) -> dict:
    cfg = yaml.safe_load(Path(config).read_text())
    rel, seed = cfg["release"], cfg["seed"]
    size, fs = rel["window_s"], NOMINAL_RATE_HZ
    stride = size * (1 - cfg["overlap"])
    aug = cfg["augment"] if rel["model"] == "cnn_aug" else None

    df = corpus.load()
    uwash = df[df["source"] == "uwash"]
    hold = uwash["subject_id"] == rel["golden_subject"]
    w = session_windows(uwash[~hold], size, stride)
    m = w["y"] != UNLABELLED
    evaluate.check_disjoint(w["subject"], [rel["golden_subject"]])
    model, mean, std = cnn1d.fit(w["x"][m], w["y"][m], cfg["cnn"], aug, seed)
    tflite = to_tflite(model)

    def zscore(x):
        return ((x - mean) / std).astype(np.float32)

    # Golden files (SPEC 9.2): the held-out subject's longest session, from its first sample.
    g = uwash[hold]
    sess = g["session_id"].value_counts().idxmax()
    g = g[g["session_id"] == sess].sort_values("timestamp_ns")
    gw = session_windows(g, size, stride)
    inputs = zscore(gw["x"])
    assert len(inputs) >= 50, (
        f"golden session {sess} gives {len(inputs)} windows; SPEC 9.2 needs 50"
    )
    outputs = run_tflite(tflite, inputs)
    float_out = model.predict(inputs, verbose=0)

    # Quantization check on data the release model never saw: the golden session and zhang.
    z = session_windows(df[df["source"] == "zhang_who"], size, stride, mirror="right")
    zx = zscore(z["x"])
    f1_float = _f1_per_subject(z["y"], model.predict(zx, verbose=0).argmax(1), z["subject"])
    f1_tflite = _f1_per_subject(z["y"], run_tflite(tflite, zx).argmax(1), z["subject"])

    sweep = json.loads((corpus.REPORTS / f"M2_{Path(config).stem}.json").read_text())
    loso = sweep["results"][f"{size}s/{rel['model']}"]["loso"][rel["smoothing"]]
    meta = {
        "spec_version": "1.0",
        "model_id": "step-classifier",
        "model_version": rel["model_version"],
        "task": "step_classification",
        "sample_rate_hz": fs,
        "window_size_s": size,
        "window_stride_s": stride,
        "channels": CHANNELS,
        "preprocessing": {
            "bandpass": {"low_hz": 1.0, "high_hz": 18.0, "order": 3},
            "gravity_align": True,
            "gravity_lowpass_hz": 0.3,
            "normalization": "zscore_per_channel",
            "norm_mean": [float(v) for v in mean],
            "norm_std": [float(v) for v in std],
        },
        "input": {"shape": [1, round(size * fs), len(CHANNELS)], "dtype": "float32"},
        "output": {"shape": [1, len(LABELS)], "dtype": "float32", "activation": "softmax"},
        "labels": list(LABELS),
        "postprocessing": {"moving_average_k": loso["k"]},
        "training": {
            "datasets": ["uwash"],
            "loso_macro_f1": round(loso["macro_f1_mean"], 4),
            "loso_macro_f1_std": round(loso["macro_f1_std"], 4),
            "commit": sweep["commit"],
        },
    }
    check = {
        "model_bytes": len(tflite),
        "golden_subject": rel["golden_subject"],
        "golden_session": sess,
        "golden_windows": len(inputs),
        "golden_max_abs_diff_vs_float": float(np.abs(outputs - float_out).max()),
        "golden_top1_agreement_vs_float": float((outputs.argmax(1) == float_out.argmax(1)).mean()),
        "zhang_macro_f1_float": f1_float,
        "zhang_macro_f1_tflite": f1_tflite,
        "quantization_f1_drop": f1_float - f1_tflite,
        "commit": evaluate._git_sha(),
    }

    (out / "golden").mkdir(parents=True, exist_ok=True)
    (out / "model.tflite").write_bytes(tflite)
    (out / "model_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    raw = g[CHANNELS].to_numpy(np.float32)
    np.save(out / "golden" / "raw.npy", raw)
    np.save(out / "golden" / "inputs.npy", inputs)
    np.save(out / "golden" / "outputs.npy", outputs)
    np.save(out / "float_outputs.npy", float_out)  # test_export only, not released
    (out / "export_check.json").write_text(json.dumps(check, indent=2) + "\n")
    return check
