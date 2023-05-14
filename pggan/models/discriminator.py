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


class ProgressivelyGrowingDiscriminator(tf.keras.Model):
    def __init__(
        self,
        latent_dimension: int,
        filters: int,
        resolution_scales: List[float],
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.latent_dimension = latent_dimension
        self.filters = filters
        self.resolution_scales = resolution_scales

    def build(self, input_shape):
        self.minibatch_std = MiniBatchStd(epsilon=1e-8)

        self.progressive_blocks, self.rgb_layers = [], []
        for idx in range(len(self.resolution_scales) - 1, 0, -1):
            self.progressive_blocks.append(
                ConvolutionBlock(
                    filters=int(self.filters * self.resolution_scales[idx - 1]),
                    use_pixelnorm=False,
                )
            )
            self.rgb_layers.append(
                WeightScaledConv2D(
                    filters=int(self.filters * self.resolution_scales[idx]),
                    kernel_size=1,
                    strides=1,
                    padding="valid",
                )
            )

        self.initial_rgb = WeightScaledConv2D(
            filters=self.filters, kernel_size=1, strides=1, padding="valid"
        )
        self.rgb_layers.append(self.initial_rgb)
        self.average_pooling = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)
        self.final_block = DiscriminatorFinalBlock(filters=self.filters)
    
    def fade_in(self, alpha, downscaled, output):
        return alpha * output + (1 - alpha) * downscaled

    def call(self, inputs):
        image, alpha, resolution_steps = inputs
        current_step = len(self.progressive_blocks) - resolution_steps
        output = tf.nn.leaky_relu(self.rgb_layers[current_step](image), alpha=0.2)

        if resolution_steps == 0:
            output = self.minibatch_std(output)
            return self.final_block(output)

        downscaled = tf.nn.leaky_relu(
            self.rgb_layers[current_step + 1](self.average_pooling(image)), alpha=0.2
        )
        output = self.average_pooling(self.progressive_blocks[current_step](output))
        output = self.fade_in(alpha, downscaled, output)

        for step in range(current_step + 1, len(self.progressive_blocks)):
            output = self.progressive_blocks[step](output)
            output = self.average_pooling(output)
        
        output = self.minibatch_std(output)
        output = self.final_block(output)

        output_shape = get_shape(output)
        output = tf.reshape(output, shape=[output_shape[0], output_shape[-1]])

        return output
