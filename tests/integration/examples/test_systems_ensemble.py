import subprocess
import os
import tempfile
import shutil
import unittest
import feast
import numpy as np

from merlin.schema import Schema, ColumnSchema
from merlin.systems.dag.ops.feast import QueryFeast
from merlin.systems.dag.ops.tensorflow import PredictTensorflow
from merlin.systems.dag.ops.workflow import TransformWorkflow
from merlin.systems.dag.ops.softmax_sampling import SoftmaxSampling
from merlin.systems.dag.ensemble import Ensemble
from nvtabular.ops import (
    Categorify,
    TagAsUserID,
    TagAsItemID,
    TagAsItemFeatures,
    TagAsUserFeatures,
    AddMetadata,
    Filter,
)

from merlin.schema.tags import Tags

import merlin.models.tf as mm
from merlin.io.dataset import Dataset
from merlin.datasets.ecommerce import transform_aliccp
import tensorflow as tf
from merlin.datasets.synthetic import generate_data

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

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


class MerlinTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Setup prepares the data to be used
        """
        cls.base_data_dir = tempfile.mkdtemp()

        # GENERATE DATA AND NVT WORKFLOW FOR RETRIEVAL MODEL
        train_raw, valid_raw = generate_data(
            "aliccp-raw", NUM_ROWS, set_sizes=(0.7, 0.3)
        )
        cls.retrieval_workflow_path = os.path.join(
            cls.base_data_dir, "processed", "retrieval"
        )
        cls.retrieval_nvt_workflow = _retrieval_workflow(
            os.path.join(cls.base_data_dir, "retrieval_categories")
        )

        # This fits and saves the workflow and the saved data to the MerlinTestCase.retrieval_workflow_path
        transform_aliccp(
            (train_raw, valid_raw),
            cls.retrieval_workflow_path,
            nvt_workflow=cls.retrieval_nvt_workflow,
            workflow_name="workflow_retrieval",
        )

        # GENERATE DATA AND NVT WORKFLOW FOR RANKING MODEL
        cls.ranking_workflow_path = os.path.join(
            cls.base_data_dir, "processed", "ranking"
        )

        # Configures a feast feature store. This requires reshaping the train/valid data a bit
        # and then ingesting into a feature store. It is only used for feature retrieval, not
        # training
        cls.feature_store_path = os.path.join(cls.base_data_dir, "feast_feature_store")
        _configure_feast(cls.feature_store_path, train_raw)

        cls.ranking_nvt_workflow = _ranking_workflow()
        transform_aliccp(
            (train_raw, valid_raw),
            cls.ranking_workflow_path,
            nvt_workflow=cls.ranking_nvt_workflow,
            workflow_name="workflow_ranking",
        )

        # Train a two-tower model and save it. This is used by several tests.
        cls.retrieval_model_path = os.path.join(cls.base_data_dir, "two_tower_model")
        _train_two_tower(
            os.path.join(cls.base_data_dir, "processed", "ranking", "train"),
            os.path.join(cls.base_data_dir, "processed", "ranking", "valid"),
            cls.retrieval_model_path,
        )

        # remove `click` since that's not used again.
        cls.retrieval_nvt_workflow.remove_inputs(["click"])

    @classmethod
    def tearDownClass(cls):
        """
        Remove the directory after the test
        This will also destroy the feast feature store(s).
        """
        shutil.rmtree(cls.base_data_dir)

    def test_training_data_exists(self):
        for model in [
            MerlinTestCase.retrieval_workflow_path,
            MerlinTestCase.ranking_workflow_path,
        ]:
            for split in ["train", "valid"]:
                self.assertTrue(os.path.exists(os.path.join(model, split)))

    def test_nvt_retrieval_ensemble(self):
        # Builds the most basic ensemble:
        # Features come in and get transformed with TransformWorkflow
        # Then get fed into a retrieval model.
        # Items come out.

        request_schema = Schema(
            column_schemas=[ColumnSchema(name=n) for n in USER_FEATURES + ["user_id"]]
        )

        pipeline = (
            request_schema.column_names
            >> TransformWorkflow(MerlinTestCase.retrieval_nvt_workflow)
            >> PredictTensorflow(MerlinTestCase.retrieval_model_path)
        )

        ensemble = Ensemble(pipeline, request_schema)
        # TODO: run ensemble in triton, make a request.

    def test_nvt_retrieval_with_sampling_ensemble(self):
        # Builds the most basic ensemble:
        # Features come in and get transformed with TransformWorkflow
        # Then get fed into a retrieval model.
        # Items come out.

        request_schema = Schema(
            column_schemas=[ColumnSchema(name="user_id", dtype=np.int32)]
        )

        pipeline = (
            ["user_id"]
            >> QueryFeast.from_feature_view(
                store=feast.FeatureStore(MerlinTestCase.feature_store_path),
                view="user_features",
                column="user_id",
                include_id=True,
            )
            >> TransformWorkflow(MerlinTestCase.retrieval_nvt_workflow)
            >> PredictTensorflow(MerlinTestCase.retrieval_model_path)
        )

        ensemble = Ensemble(pipeline, request_schema)


def _configure_feast(feature_store_path, data_to_ingest: Dataset):
    # Make the directory if it doesn't exist
    # Copy user|item_features.py files to the path
    # Modify the data and store it in the directory
    # feast init, apply, materialize
    if not os.path.exists(feature_store_path):
        os.makedirs(os.path.join(feature_store_path, "data"))
    with open(
        os.path.join(feature_store_path, "feature_store.yaml"), "w"
    ) as config_file:
        config_file.write(
            """
project: feast_feature_store
registry: data/registry.db
provider: local
online_store:
    path: data/online_store.db
"""
        )
    cwd = os.path.dirname(os.path.abspath(__file__))
    shutil.copy(os.path.join(cwd, "feast", "item_features.py"), feature_store_path)
    shutil.copy(os.path.join(cwd, "feast", "user_features.py"), feature_store_path)

    # Split training data into user / item and add timestamps.
    from datetime import datetime
    from merlin.models.utils.dataset import unique_rows_by_features

    user_features = (
        unique_rows_by_features(data_to_ingest, Tags.USER, Tags.USER_ID)
        .compute()
        .reset_index(drop=True)
    )
    user_features["datetime"] = datetime.now()
    user_features["datetime"] = user_features["datetime"].astype("datetime64[ns]")
    user_features["created"] = datetime.now()
    user_features["created"] = user_features["created"].astype("datetime64[ns]")
    user_features.to_parquet(
        os.path.join(feature_store_path, "data", "user_features.parquet")
    )

    item_features = (
        unique_rows_by_features(data_to_ingest, Tags.ITEM, Tags.ITEM_ID)
        .compute()
        .reset_index(drop=True)
    )
    item_features["datetime"] = datetime.now()
    item_features["datetime"] = item_features["datetime"].astype("datetime64[ns]")
    item_features["created"] = datetime.now()
    item_features["created"] = item_features["created"].astype("datetime64[ns]")
    item_features.to_parquet(
        os.path.join(feature_store_path, "data", "item_features.parquet")
    )

    os.environ.setdefault(
        "FEAST_USER_FEATURES_PATH",
        os.path.join(feature_store_path, "data", "user_features.parquet"),
    )
    os.environ.setdefault(
        "FEAST_ITEM_FEATURES_PATH",
        os.path.join(feature_store_path, "data", "item_features.parquet"),
    )
    subprocess.run(["feast", "-c", feature_store_path, "apply"])
    subprocess.run(
        [
            "feast",
            "-c",
            feature_store_path,
            "materialize",
            "1995-01-01T01:01:01",
            "2025-01-01T01:01:01",
        ]
    )

    # This is a useful place to drop a debugger and manually see what is in feast.
    # import pdb; pdb.set_trace()
    # import feast
    # fs = feast.FeatureStore(feature_store_path)
    # fs.get_online_features(["user_features:user_shops"], [{"user_id":1}]).to_dict()


def _retrieval_workflow(category_out_path):
    user_id = (
        ["user_id"]
        >> Categorify(dtype="int32", out_path=category_out_path)
        >> TagAsUserID()
    )
    # item_id = (
    #     ["item_id"]
    #     >> Categorify(dtype="int32", out_path=category_out_path)
    #     >> TagAsItemID()
    # )

    # item_features = (
    #     ITEM_FEATURES
    #     >> Categorify(dtype="int32", out_path=category_out_path)
    #     >> TagAsItemFeatures()
    # )

    user_features = (
        USER_FEATURES
        >> Categorify(dtype="int32", out_path=category_out_path)
        >> TagAsUserFeatures()
    )

    inputs = user_id + user_features + ["click"]

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
