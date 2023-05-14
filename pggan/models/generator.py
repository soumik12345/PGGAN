import tensorflow as tf

from .layers import WeightScaledConv2D, PixelNorm


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
