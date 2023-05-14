import tensorflow as tf


class WeightScaledConv2D(tf.keras.layers.Conv2D):
    def __init__(self, filters, kernel_size, gain: int = 2, **kwargs):
        super().__init__(filters, kernel_size, **kwargs)
        self.gain = gain

    def build(self, input_shape):
        input_channels = input_shape[-1]
        fan_in = (self.kernel_size[0] * self.kernel_size[1]) * input_channels
        self.scale = tf.sqrt(self.gain / fan_in)
        self.kernel = self.add_weight(
            name="kernel",
            shape=[
                self.kernel_size[0],
                self.kernel_size[1],
                input_channels,
                self.filters,
            ],
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0),
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(self.filters,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
        )

    def call(self, inputs):
        x = tf.nn.conv2d(
            input=inputs,
            filters=self.kernel * self.scale,
            strides=self.strides,
            padding=self.padding.upper(),
        )
        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias)
        return x


class WeightScaledDense(tf.keras.layers.Dense):
    def __init__(self, units, gain: int = 2, **kwargs):
        super().__init__(units, **kwargs)
        self.gain = gain

    def build(self, input_shape):
        input_channels = input_shape[-1]
        self.scale = tf.sqrt(self.gain / input_channels)
        self.kernel = self.add_weight(
            name="kernel",
            shape=[input_channels, self.units],
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0),
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(self.units,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
        )

    def call(self, inputs):
        x = tf.matmul(inputs, self.kernel * self.scale)
        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias)
        return x


class PixelNorm(tf.keras.layers.Layer):
    def __init__(self, epsilon: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, inputs):
        return inputs / tf.sqrt(
            tf.reduce_mean(inputs**2, axis=-1, keepdims=True) + self.epsilon
        )


class ConvolutionBlock(tf.keras.layers.Layer):
    def __init__(self, filters: int, use_pixelnorm: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.use_pixelnorm = use_pixelnorm

    def build(self, input_shape):
        self.convolution_1 = WeightScaledConv2D(filters=self.filters, kernel_size=3)
        self.convolution_2 = WeightScaledConv2D(filters=self.filters, kernel_size=3)
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.pixel_norm = PixelNorm()

    def call(self, inputs):
        x = self.leaky_relu(self.convolution_1(inputs))
        x = self.pixel_norm(x) if self.use_pixelnorm else x
        x = self.leaky_relu(self.convolution_2(x))
        return self.pixel_norm(x) if self.use_pixelnorm else x
