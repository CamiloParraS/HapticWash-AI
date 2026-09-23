"""Generate the M0 stub model: random-weight TFLite with correct shapes + model_meta.json.

SPEC M0. Not a trained model — it exists so `HapticWash` can wire up LiteRT loading
and the golden-file harness against real tensor shapes before M2 produces a real one.

Requires the `ml` extra (TensorFlow), which is Linux-only — run on CI or WSL:
    uv run --extra ml python scripts/make_stub_model.py
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from haptic_ai.schema import LABELS, NOMINAL_RATE_HZ

OUT = Path("artifacts")
WINDOW_S = 3.0
STRIDE_S = 1.5
CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz"]
W = int(WINDOW_S * NOMINAL_RATE_HZ)  # 150


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001 - best effort in a stub generator
        return "unknown"


def build_tflite() -> bytes:
    import tensorflow as tf

    tf.random.set_seed(0)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(W, len(CHANNELS)), name="window"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(len(LABELS), activation="softmax", name="probs"),
        ]
    )
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    # ponytail: float32 stub — int8 quantization is M2 export.py, not needed to prove shapes.
    return conv.convert()


def model_meta(sha: str) -> dict:
    return {
        "spec_version": "1.0",
        "model_id": "step-classifier",
        "model_version": "0.0.1",
        "task": "step_classification",
        "sample_rate_hz": NOMINAL_RATE_HZ,
        "window_size_s": WINDOW_S,
        "window_stride_s": STRIDE_S,
        "channels": CHANNELS,
        "preprocessing": {
            "bandpass": {"low_hz": 1.0, "high_hz": 18.0, "order": 3},
            "gravity_align": True,
            "gravity_lowpass_hz": 0.3,
            "normalization": "zscore_per_channel",
            "norm_mean": [0.0] * 6,
            "norm_std": [1.0] * 6,
        },
        "input": {"shape": [1, W, len(CHANNELS)], "dtype": "float32"},
        "output": {"shape": [1, len(LABELS)], "dtype": "float32", "activation": "softmax"},
        "labels": list(LABELS),
        "postprocessing": {"moving_average_k": 5},
        "training": {
            "datasets": ["zhang_who", "ablutomania"],
            "loso_macro_f1": 0.00,
            "loso_macro_f1_std": 0.00,
            "commit": sha,
        },
    }


def main() -> None:
    try:
        tflite = build_tflite()
    except ModuleNotFoundError:
        raise SystemExit(
            "TensorFlow not installed. The `ml` extra is Linux-only (DECISIONS.md D2); "
            "run this on CI or WSL: uv run --extra ml python scripts/make_stub_model.py"
        ) from None
    OUT.mkdir(exist_ok=True)
    (OUT / "model.tflite").write_bytes(tflite)
    sha = _git_sha()
    (OUT / "model_meta.json").write_text(json.dumps(model_meta(sha), indent=2) + "\n")
    size = (OUT / "model.tflite").stat().st_size
    print(f"wrote {OUT / 'model.tflite'} ({size} bytes) and {OUT / 'model_meta.json'} @ {sha[:8]}")


if __name__ == "__main__":
    main()
