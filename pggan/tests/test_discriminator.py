import unittest
from math import log2

import tensorflow as tf
from pggan.models import ProgressivelyGrowingDiscriminator


class ProgressivelyGrowingDiscriminatorTester(unittest.TestCase):
    def test_discriminator(self):
        critic = ProgressivelyGrowingDiscriminator(
            latent_dimension=256,
            filters=256,
            resolution_scales=[1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32],
        )
        image = tf.random.normal(shape=(1, 128, 128, 3))
        self.assertEqual(critic([image, 1e-5, int(log2(128 / 4))]).shape, (1, 1))
