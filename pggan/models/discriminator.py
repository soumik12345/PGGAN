from typing import List

import tensorflow as tf

from .layers import WeightScaledConv2D, PixelNorm, ConvolutionBlock
from .utils import get_shape


class MiniBatchStd(tf.keras.layers.Layer):
    def __init__(self, epsilon: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon

    def call(self, inputs):
        n, h, w, c = get_shape(inputs)
        group_size = tf.minimum(4, n)
        x = tf.reshape(inputs, [group_size, -1, h, w, c])
        group_mean, group_var = tf.nn.moments(x, axes=(0), keepdims=False)
        group_std = tf.sqrt(group_var + self.epsilon)
        avg_std = tf.reduce_mean(group_std, axis=[1, 2, 3], keepdims=True)
        x = tf.tile(avg_std, [group_size, h, w, 1])
        return tf.concat([inputs, x], axis=-1)
