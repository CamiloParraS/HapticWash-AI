"""Small 1D CNN on preprocessed windows (SPEC M2, tensor contract SPEC 8.3). Keras (D21)."""

import numpy as np
import tensorflow as tf

from haptic_ai.preprocess import augment
from haptic_ai.schema import LABELS

layers = tf.keras.layers


def build(window: int, channels: int = 6) -> tf.keras.Model:
    """``float32[1, W, 6]`` z-scored window -> ``float32[1, 7]`` softmax (SPEC 8.3)."""
    inp = tf.keras.Input((window, channels))
    x = inp
    for filters, kernel in ((32, 5), (64, 5), (64, 3)):
        x = layers.Conv1D(filters, kernel, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling1D(2)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    return tf.keras.Model(inp, layers.Dense(len(LABELS), activation="softmax")(x))


def fit(x: np.ndarray, y: np.ndarray, cfg: dict, aug: dict | None, seed: int):
    """Train on raw-unit windows. Returns ``(model, norm_mean, norm_std)``.

    z-score stats come from the un-augmented training windows (SPEC 8.2 ``norm_*``).
    Augmentation is redrawn every epoch. Fixed epoch count: no early stopping, since
    the only held-out data is the LOSO test subject.
    """
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()  # SPEC 4.4
    mean, std = x.mean((0, 1)), x.std((0, 1))
    counts = np.bincount(y, minlength=len(LABELS))
    weights = {c: len(y) / (len(LABELS) * n) for c, n in enumerate(counts) if n}
    model = build(x.shape[1], x.shape[2])
    model.compile(tf.keras.optimizers.Adam(cfg["learning_rate"]), "sparse_categorical_crossentropy")
    rng = np.random.default_rng(seed)
    for _ in range(cfg["epochs"]):
        xa = augment(x, rng, **aug) if aug else x
        model.fit(
            (xa - mean) / std, y, batch_size=cfg["batch_size"], class_weight=weights, verbose=0
        )
    return model, mean, std


def fit_predict(x_tr, y_tr, x_te, cfg: dict, aug: dict | None, seed: int) -> np.ndarray:
    model, mean, std = fit(x_tr, y_tr, cfg, aug, seed)
    return model.predict((x_te - mean) / std, batch_size=1024, verbose=0)
