from typing import List

import tensorflow as tf

from .layers import WeightScaledConv2D, PixelNorm, ConvolutionBlock


class GeneratorInitialBlock(tf.keras.layers.Layer):
    def __init__(self, filters: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filters = filters

    def build(self, input_shape):
        self.pixel_norm = PixelNorm()
        self.conv_transpose_2d = tf.keras.layers.Conv2DTranspose(
            filters=self.filters, kernel_size=4, strides=1, padding="valid"
        )
        self.weight_scaled_conv2d = WeightScaledConv2D(
            filters=self.filters, kernel_size=3, strides=1, padding="same"
        )

    def call(self, inputs):
        x = tf.nn.leaky_relu(self.conv_transpose_2d(self.pixel_norm(inputs)), alpha=0.2)
        x = self.pixel_norm(tf.nn.leaky_relu(self.weight_scaled_conv2d(x), alpha=0.2))
        return x


class ProgressivelyGrowingGenerator(tf.keras.Model):
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
        self.initial_block = GeneratorInitialBlock(filters=self.filters)

        self.initial_rgb = WeightScaledConv2D(
            filters=3, kernel_size=1, strides=1, padding="valid"
        )
        self.progressive_blocks, self.rgb_layers = [], [self.initial_rgb]
        for idx in range(len(self.resolution_scales) - 1):
            conv_filters = int(self.filters * self.resolution_scales[idx + 1])
            self.progressive_blocks.append(ConvolutionBlock(filters=conv_filters))
            self.rgb_layers.append(
                WeightScaledConv2D(filters=3, kernel_size=1, strides=1, padding="valid")
            )

    def fade_in(self, alpha, upscaled, generated):
        return tf.tanh(alpha * generated + (1 - alpha) * upscaled)

    def call(self, inputs):
        latent_noise, alpha, resolution_steps = inputs
        output = self.initial_block(latent_noise)

        if resolution_steps == 0:
            return self.initial_rgb(output)

        for step in range(resolution_steps):
            upscaled_x = tf.keras.layers.UpSampling2D(
                size=(2, 2), interpolation="nearest"
            )(output)
            output = self.progressive_blocks[step](upscaled_x)

        final_upscaled = self.rgb_layers[resolution_steps - 1](upscaled_x)
        final_output = self.rgb_layers[resolution_steps](output)
        return self.fade_in(alpha, final_upscaled, final_output)
