# Copyright (c) 2026, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import pytest
import tensorflow as tf

from examples.quick_start.scripts.ranking.mtl import get_task_sample_weights


@pytest.mark.parametrize("mode", ["training", "testing"])
def test_regression_sample_space_masks_loss_and_gradient(mode):
    post = get_task_sample_weights("value", False, {}, {"value": "eligible"})
    assert post is not None
    predictions = tf.Variable([[1.0], [20.0], [3.0]])
    targets = {"value": tf.zeros((3, 1))}
    features = {"eligible": tf.constant([[1.0], [0.0], [1.0]])}
    with tf.GradientTape() as tape:
        outputs = post(
            predictions,
            features=features,
            targets=targets,
            target_name="value",
            **{mode: True},
        )
        loss = tf.keras.losses.MeanSquaredError()(
            outputs.targets, outputs.outputs, sample_weight=outputs.sample_weight
        )
    np.testing.assert_allclose(outputs.sample_weight.numpy(), [[1.0], [0.0], [1.0]])
    np.testing.assert_allclose(loss.numpy(), 10.0 / 3.0)
    np.testing.assert_allclose(
        tape.gradient(loss, predictions).numpy(), [[2 / 3], [0], [2]]
    )


def test_binary_sample_space_composes_with_class_weights():
    post = get_task_sample_weights("click", True, {"click": 3.0}, {"click": "eligible"})
    outputs = post(
        tf.zeros((3, 1)),
        targets={"click": tf.constant([[0.0], [1.0], [1.0]])},
        features={"eligible": tf.constant([[1.0], [0.0], [1.0]])},
        target_name="click",
        training=True,
    )
    np.testing.assert_allclose(outputs.sample_weight.numpy(), [[1.0], [0.0], [3.0]])


def test_regression_without_sample_space_needs_no_weight_block():
    assert get_task_sample_weights("value", False, {}, {"value": None}) is None
