from math import log2
from functools import partial

import tensorflow as tf
import tensorflow_datasets as tfds


class DataLoader:
    def __init__(self, name: str) -> None:
        if name not in tfds.list_builders():
            raise ValueError(f"Unable to find dataset {name} in tfds catalog")
        self.dataset_splits, self.dataset_builder_info = self.build_datasets(name)
        self.batch_size_list = [32, 32, 32, 16, 16, 16, 16, 8, 4]

    def build_datasets(self, name: str):
        dataset_builder = tfds.builder(name)
        dataset_builder.download_and_prepare()
        dataset_builder_info = dataset_builder.info
        splits = dataset_builder_info.splits
        dataset_splits = {}
        for key, _ in splits.items():
            num_shards = dataset_builder.info.splits[key].num_shards
            num_examples = dataset_builder.info.splits[key].num_examples
            dataset_splits[key] = dataset_builder.as_dataset(key)
        return dataset_splits, dataset_builder_info

    def preprocess(self, sample, image_size):
        image = sample["image"]
        image = tf.keras.layers.Resizing(height=image_size, width=image_size)(image)
        image = tf.keras.layers.RandomFlip(mode="horizontal")(image)
        image = tf.keras.layers.Normalization(mean=0.5, variance=0.25)(image)
        return image

    def build_split(self, split: str, image_size: int):
        dataset = self.dataset_splits[split]
        dataset = dataset.batch(self.batch_size_list[int(log2(image_size / 4))])
        dataset = dataset.map(
            partial(self.preprocess, image_size=image_size),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        dataset = dataset.prefetch(256)
        return dataset
