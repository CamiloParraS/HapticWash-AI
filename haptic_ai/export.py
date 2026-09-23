"""Quantize to TFLite, emit model_meta.json and golden files."""

import json

import tensorflow as tf


def quantize_and_export(model, output_path: str, representative_data=None):
    """Export model to TFLite with int8 quantization."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if representative_data:
        converter.representative_dataset = representative_data
    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)


def write_metadata(path: str, metadata: dict):
    """Write model metadata."""
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
