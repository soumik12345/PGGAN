import unittest
from math import log2

import tensorflow as tf

from pggan.models import (
    ProgressivelyGrowingGenerator,
    ProgressivelyGrowingDiscriminator,
)


class EndToEndModelTester(unittest.TestCase):
    def __init__(self):
        self.latent_dimension = 100
        self.filters = 256
        self.resolution_scales = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]
        self.resolutions = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
        self.generator = ProgressivelyGrowingGenerator(
            latent_dimension=self.latent_dimension,
            filters=self.filters,
            resolution_scales=self.resolution_scales,
        )
        self.discriminator = ProgressivelyGrowingDiscriminator(
            latent_dimension=self.latent_dimension,
            filters=self.filters,
            resolution_scales=self.resolution_scales,
        )

    def test_end_to_end(self):
        for image_size in self.resolutions:
            num_steps = int(log2(image_size / 4))
            latent_noise = tf.random.normal((1, 1, 1, self.latent_dimension))
            generated_image = self.generator([latent_noise, 1e-5, num_steps])
            self.assertEqual(generated_image.shape, (1, image_size, image_size, 3))
            disc_output = self.discriminator([generated_image, 1e-5, num_steps])
            self.assertEqual(disc_output.shape, (1, 1))
