from typing import List

import tensorflow as tf


def get_shape(input_tensor: tf.Tensor) -> List[int]:
    dynamic_shape = tf.shape(input_tensor)
    return (
        dynamic_shape
        if input_tensor.shape == tf.TensorShape(None)
        else [
            dynamic_shape[i] if s is None else s
            for i, s in enumerate(input_tensor.shape.as_list())
        ]
    )
