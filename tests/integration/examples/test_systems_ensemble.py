import os
import itertools
import tempfile
import shutil
import unittest
from merlin.schema import Schema, ColumnSchema
from merlin.systems.dag.ops.tensorflow import PredictTensorflow
from merlin.systems.dag.ensemble import Ensemble
from nvtabular.workflow import Workflow
from nvtabular.ops import (
    Categorify,
    TagAsUserID,
    TagAsItemID,
    TagAsItemFeatures,
    TagAsUserFeatures,
    AddMetadata,
    Filter,
    Rename,
)
import pytest

from merlin.schema.tags import Tags

import merlin.models.tf as mm
from merlin.io.dataset import Dataset
from merlin.datasets.ecommerce import transform_aliccp
import tensorflow as tf
from merlin.datasets.synthetic import generate_data

NUM_ROWS = 10_000
USER_FEATURES = [
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
ITEM_FEATURES = ["item_category", "item_shop", "item_brand"]


class WidgetTestCase(unittest.TestCase):
    def setUp(self):
        """
        Setup prepares the data to be used
        """
        self.base_data_dir = tempfile.mkdtemp()

        # GENERATE DATA AND NVT WORKFLOW FOR RETRIEVAL MODEL
        train_raw, valid_raw = generate_data(
            "aliccp-raw", NUM_ROWS, set_sizes=(0.7, 0.3)
        )
        self.retrieval_workflow_path = os.path.join(
            self.base_data_dir, "processed", "retrieval"
        )
        self.retrieval_nvt_workflow = _retrieval_workflow()

        # This fits and saves the workflow to the output_path
        transform_aliccp(
            (train_raw, valid_raw),
            self.retrieval_workflow_path,
            nvt_workflow=self.retrieval_nvt_workflow,
            workflow_name="workflow_retrieval",
        )

        # GENERATE DATA AND NVT WORKFLOW FOR RANKING MODEL
        self.ranking_workflow_path = os.path.join(
            self.base_data_dir, "processed", "ranking"
        )

        self.ranking_nvt_workflow = _retrieval_workflow()
        transform_aliccp(
            (train_raw, valid_raw),
            self.ranking_workflow_path,
            nvt_workflow=self.ranking_nvt_workflow,
            workflow_name="workflow_ranking",
        )

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.base_data_dir)
        # TODO: destroy the feast feature store

    def test_training_data_exists(self):
        for model in [self.retrieval_workflow_path, self.ranking_workflow_path]:
            for split in ["train", "valid"]:
                self.assertTrue(os.path.exists(os.path.join(model, split)))

    def test_nvt_retrieval_ensemble(self):
        # Builds the most basic ensemble:
        # Features come in and get transformed with TransformWorkflow
        # Then get fed into a retrieval model.
        # Items come out.
        retrieval_model = _train_two_tower(
            os.path.join(self.base_data_dir, "processed", "ranking", "train"),
            os.path.join(self.base_data_dir, "processed", "ranking", "valid"),
            os.path.join(self.base_data_dir, "two_tower_model"),
        )
        retrieval_workflow_op = Workflow.load(self.retrieval_nvt_workflow)
        retrieval_model_op = PredictTensorflow(retrieval_model)
        pipeline = retrieval_workflow_op + retrieval_model_op

        all_integer_columns = list(
            itertools.chain(*[USER_FEATURES, ITEM_FEATURES, ["user_id", "item_id"]])
        )
        request_schema = Schema(
            column_schemas=[ColumnSchema(name=n) for n in all_integer_columns]
        )
        ensemble = Ensemble(pipeline, request_schema)


def _retrieval_workflow():
    user_id = (
        ["user_id"]
        >> Categorify(dtype="int32", out_path="./categories_tt")
        >> TagAsUserID()
    )
    item_id = (
        ["item_id"]
        >> Categorify(dtype="int32", out_path="./categories_tt")
        >> TagAsItemID()
    )

    item_features = (
        ITEM_FEATURES
        >> Categorify(dtype="int32", out_path="./categories_tt")
        >> TagAsItemFeatures()
    )

    user_features = (
        USER_FEATURES
        >> Categorify(dtype="int32", out_path="./categories_tt")
        >> TagAsUserFeatures()
    )

    inputs = user_id + item_id + item_features + user_features + ["click"]

    outputs = inputs >> Filter(f=lambda df: df["click"] == 1)
    return outputs


def _ranking_workflow():
    user_id = ["user_id"] >> Categorify(dtype="int32") >> TagAsUserID()
    item_id = ["item_id"] >> Categorify(dtype="int32") >> TagAsItemID()

    item_features = (
        ["item_category", "item_shop", "item_brand"]
        >> Categorify(dtype="int32")
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
        >> Categorify(dtype="int32")
        >> TagAsUserFeatures()
    )

    targets = ["click"] >> AddMetadata(tags=[Tags.BINARY_CLASSIFICATION, "target"])

    outputs = user_id + item_id + item_features + user_features + targets
    return outputs


def _train_two_tower(train_data_path, valid_data_path, output_path):
    train_tt = Dataset(os.path.join(train_data_path, "*.parquet"))
    valid_tt = Dataset(os.path.join(valid_data_path, "*.parquet"))

    schema = train_tt.schema
    schema = schema.select_by_tag([Tags.ITEM_ID, Tags.USER_ID, Tags.ITEM, Tags.USER])
    model_tt = mm.TwoTowerModel(
        schema,
        query_tower=mm.MLPBlock([128, 64], no_activation_last_layer=True),
        samplers=[mm.InBatchSampler()],
        embedding_options=mm.EmbeddingOptions(infer_embedding_sizes=True),
    )
    model_tt.compile(
        optimizer="adam",
        run_eagerly=False,
        loss="categorical_crossentropy",
        metrics=[mm.RecallAt(10), mm.NDCGAt(10)],
    )
    model_tt.fit(train_tt, validation_data=valid_tt, batch_size=1024 * 8, epochs=1)
    query_tower = model_tt.retrieval_block.query_block()
    query_tower.save(output_path)
    return query_tower


def _train_dlrm_model(train_data_path, valid_data_path, model_output_path):
    # define train and valid dataset objects
    train = Dataset(os.path.join(train_data_path, "*.parquet"), part_size="500MB")
    valid = Dataset(os.path.join(valid_data_path, "*.parquet"), part_size="500MB")

    # define schema object
    schema = train.schema

    target_column = schema.select_by_tag(Tags.TARGET).column_names[0]
    model = mm.DLRMModel(
        schema,
        embedding_dim=64,
        bottom_block=mm.MLPBlock([128, 64]),
        top_block=mm.MLPBlock([128, 64, 32]),
        prediction_tasks=mm.BinaryClassificationTask(target_column),
    )

    model.compile(optimizer="adam", run_eagerly=False, metrics=[tf.keras.metrics.AUC()])
    model.fit(train, validation_data=valid, batch_size=16 * 1024)
    model.save(model_output_path)
