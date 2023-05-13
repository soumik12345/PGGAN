import tensorflow as tf


class WeightScaledConv2D(tf.keras.layers.Conv2D):
    def __init__(self, filters, kernel_size, gain: int = 2, **kwargs):
        super().__init__(filters, kernel_size, **kwargs)
        self.gain = gain
        self.kernel_initializer = (
            tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0),
        )

    def build(self, input_shape):
        input_channels = input_shape[-1]
        fan_in = (self.kernel_size**2) * input_channels
        self.scale = tf.sqrt(self.gain / fan_in)

    def call(self, inputs):
        x = tf.nn.conv2d(
            input=inputs,
            filters=self.kernel * self.scale,
            strides=self.strides,
            padding=self.padding,
        )
        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias)
        return x


class WeightScaledDense(tf.keras.layers.Dense):
    def __init__(self, units, gain: int = 2, **kwargs):
        super().__init__(units, **kwargs)
        self.gain = gain
        self.kernel_initializer = (
            tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0),
        )

    def build(self, input_shape):
        input_channels = input_shape[-1]
        self.scale = tf.sqrt(self.gain / input_channels)

    def call(self, inputs):
        x = tf.matmul(inputs, self.kernel * self.scale)
        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias)
        return x


class PixelNorm(tf.keras.layers.Layer):
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon

    def call(self, inputs):
        return inputs / tf.sqrt(
            tf.reduce_mean(inputs**2, axis=-1, keepdims=True) + self.epsilon
        )
