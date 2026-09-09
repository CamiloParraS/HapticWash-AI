"""1D CNN on raw windows."""
import tensorflow as tf
import numpy as np


class CNN1D(tf.keras.Model):
    """Small 1D CNN for haptic classification."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def call(self, x: tf.Tensor, training=False) -> tf.Tensor:
        """Forward pass."""
        return tf.identity(x)
