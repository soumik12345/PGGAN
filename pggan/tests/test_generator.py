import unittest
from math import log2

import tensorflow as tf
from pggan.models.generator import GeneratorInitialBlock, ProgressivelyGrowingGenerator


class GeneratorInitialBlockTester(unittest.TestCase):
    def test_initial_block(self):
        block = GeneratorInitialBlock(filters=256)
        noise = tf.random.normal(shape=(1, 1, 1, 256))
        self.assertEqual(block(noise).shape, (1, 4, 4, 256))


class ProgressivelyGrowingGeneratorTester(unittest.TestCase):
    def test_generator(self):
        noise = tf.random.normal(shape=(1, 1, 1, 256))
        model = ProgressivelyGrowingGenerator(
            latent_dimension=256,
            filters=256,
            resolution_scales=[1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32],
        )
        self.assertEqual(
            model([noise, 1e-5, int(log2(128 / 4))]).shape, (1, 128, 128, 3)
        )
