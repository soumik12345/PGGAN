# class WSConv2d(nn.Module):

#     def __init__(
#         self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2
#     ):
#         super(WSConv2d, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
#         self.scale = (gain / (in_channels * (kernel_size ** 2))) ** 0.5
#         self.bias = self.conv.bias
#         self.conv.bias = None

#         # initialize conv layer
#         nn.init.normal_(self.conv.weight)
#         nn.init.zeros_(self.bias)

#     def forward(self, x):
#         return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)


import tensorflow as tf


class WeightScaledConv2D(tf.keras.layers.Conv2D):
    def __init__(self, filters, kernel_size, gain: int = 2, **kwargs):
        super().__init__(
            filters,
            kernel_size,
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0),
            **kwargs
        )
        self.gain = gain

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
