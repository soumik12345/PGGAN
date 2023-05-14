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


class DiscriminatorFinalBlock(tf.keras.layers.Layer):
    def __init__(self, filters: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filters = filters

    def build(self, input_shape):
        self.weight_scaled_convolution_1 = WeightScaledConv2D(
            filters=self.filters, kernel_size=3, strides=1, padding="same"
        )
        self.weight_scaled_convolution_2 = WeightScaledConv2D(
            filters=self.filters, kernel_size=4, strides=1, padding="valid"
        )
        self.weight_scaled_convolution_3 = WeightScaledConv2D(
            filters=1, kernel_size=1, strides=1, padding="valid"
        )

    def call(self, inputs):
        x = tf.nn.leaky_relu(self.weight_scaled_convolution_1(inputs), alpha=0.2)
        x = tf.nn.leaky_relu(self.weight_scaled_convolution_2(x), alpha=0.2)
        return self.weight_scaled_convolution_3(x)
