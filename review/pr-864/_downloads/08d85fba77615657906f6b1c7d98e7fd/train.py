#
# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import argparse
import json
import logging
import os
import sys
import tempfile

# We can control how much memory to give tensorflow with this environment variable
# IMPORTANT: make sure you do this before you initialize TF's runtime, otherwise
# TF will have claimed all free GPU memory
os.environ["TF_MEMORY_ALLOCATION"] = "0.7"  # fraction of free memory

import merlin.io
import merlin.models.tf as mm
import nvtabular as nvt
import tensorflow as tf
from merlin.schema.tags import Tags
from merlin.systems.dag.ops.workflow import TransformWorkflow
from merlin.systems.dag.ops.tensorflow import PredictTensorflow
from merlin.systems.dag.ensemble import Ensemble
import numpy as np
from nvtabular.ops import *


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def parse_args():
    """
    Parse arguments passed from the SageMaker API to the container.
    """

    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1024)

    # Data directories
    parser.add_argument(
        "--train_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN")
    )
    parser.add_argument(
        "--valid_dir", type=str, default=os.environ.get("SM_CHANNEL_VALID")
    )

    # Model directory: we will use the default set by SageMaker, /opt/ml/model
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))

    return parser.parse_known_args()


def create_nvtabular_workflow(train_path, valid_path):
    user_id = ["user_id"] >> Categorify() >> TagAsUserID()
    item_id = ["item_id"] >> Categorify() >> TagAsItemID()
    targets = ["click"] >> AddMetadata(tags=[Tags.BINARY_CLASSIFICATION, "target"])

    item_features = (
        ["item_category", "item_shop", "item_brand"]
        >> Categorify()
        >> TagAsItemFeatures()
    )

    user_features = (
        [
            "user_shops",
            "user_profile",
            "user_group",
            "user_gender",
            "user_age",
            "user_consumption_2",
            "user_is_occupied",
            "user_geography",
            "user_intentions",
            "user_brands",
            "user_categories",
        ]
        >> Categorify()
        >> TagAsUserFeatures()
    )

    outputs = user_id + item_id + item_features + user_features + targets

    workflow = nvt.Workflow(outputs)

    return workflow


def create_ensemble(workflow, model):
    serving_operators = (
        workflow.input_schema.column_names
        >> TransformWorkflow(workflow)
        >> PredictTensorflow(model)
    )
    ensemble = Ensemble(serving_operators, workflow.input_schema)
    return ensemble


def train():
    """
    Train the Merlin model.
    """
    train_path = os.path.join(args.train_dir, "*.parquet")
    valid_path = os.path.join(args.valid_dir, "*.parquet")

    workflow = create_nvtabular_workflow(
        train_path=train_path,
        valid_path=valid_path,
    )

    train_dataset = nvt.Dataset(train_path)
    valid_dataset = nvt.Dataset(valid_path)

    output_path = tempfile.mkdtemp()
    workflow_path = os.path.join(output_path, "workflow")

    workflow.fit(train_dataset)
    workflow.transform(train_dataset).to_parquet(
        output_path=os.path.join(output_path, "train")
    )
    workflow.transform(valid_dataset).to_parquet(
        output_path=os.path.join(output_path, "valid")
    )

    workflow.save(workflow_path)
    logger.info(f"Workflow saved to {workflow_path}.")

    train_data = merlin.io.Dataset(os.path.join(output_path, "train", "*.parquet"))
    valid_data = merlin.io.Dataset(os.path.join(output_path, "valid", "*.parquet"))

    schema = train_data.schema
    target_column = schema.select_by_tag(Tags.TARGET).column_names[0]

    model = mm.DLRMModel(
        schema,
        embedding_dim=64,
        bottom_block=mm.MLPBlock([128, 64]),
        top_block=mm.MLPBlock([128, 64, 32]),
        prediction_tasks=mm.BinaryClassificationTask(target_column),
    )

    model.compile("adam", run_eagerly=False, metrics=[tf.keras.metrics.AUC()])

    batch_size = args.batch_size
    epochs = args.epochs
    logger.info(f"batch_size = {batch_size}, epochs = {epochs}")

    model.fit(
        train_data,
        validation_data=valid_data,
        batch_size=args.batch_size,
        epochs=epochs,
        verbose=2,
    )

    model_path = os.path.join(output_path, "dlrm")
    model.save(model_path)
    logger.info(f"Model saved to {model_path}.")

    # We remove the label columns from its inputs.
    # This removes all columns with the TARGET tag from the workflow.
    # We do this because we need to set the workflow to only require the
    # features needed to predict, not train, when creating an inference
    # pipeline.
    label_columns = workflow.output_schema.select_by_tag(Tags.TARGET).column_names
    workflow.remove_inputs(label_columns)

    ensemble = create_ensemble(workflow, model)
    ensemble_path = args.model_dir
    ensemble.export(ensemble_path)
    logger.info(f"Ensemble graph saved to {ensemble_path}.")


if __name__ == "__main__":
    args, _ = parse_args()
    train()
