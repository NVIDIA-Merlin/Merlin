import subprocess
import os
import tempfile
import shutil
import unittest
from feast import FeatureStore
import numpy as np

from datetime import datetime
from merlin.models.loader.tf_utils import configure_tensorflow
from merlin.models.utils.dataset import unique_rows_by_features
from merlin.schema import Schema, ColumnSchema
from merlin.systems.dag.ops.faiss import QueryFaiss, setup_faiss
from merlin.systems.dag.ops.feast import QueryFeast
from merlin.systems.dag.ops.tensorflow import PredictTensorflow
from merlin.systems.dag.ops.unroll_features import UnrollFeatures
from merlin.systems.dag.ops.workflow import TransformWorkflow
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


configure_tensorflow()


class MerlinTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        This method runs once when the test class is invoked and prepares the data to be used in
        all of the tests. It will:

        * Simulate `aliccp` data
        * Define NVT workflows for the retrieval (TT) and ranking (DLRM) models
        * Train the retrieval and ranking models
        * Ingest data into a Feast online store

        All of the data and models are stored in a temp directory that is removed when the test
        suite is complete, in `tearDownClass`.
        """
        cls.base_data_dir = tempfile.mkdtemp()

        # GENERATE DATA AND NVT WORKFLOW FOR RETRIEVAL MODEL
        train_raw, valid_raw = generate_data(
            "aliccp-raw", NUM_ROWS, set_sizes=(0.7, 0.3)
        )

        ####################################################################################
        # Ingest raw data into feast.
        ####################################################################################
        # Configures a feast feature store using PRE-TRANSFORMED features. This requires reshaping
        # the train/valid data a bit and then ingesting into a feature store. It is only used for
        # feature retrieval, not training.
        cls.feature_store_path = os.path.join(cls.base_data_dir, "feast_feature_store")
        _configure_feast(cls.feature_store_path, train_raw)

        ####################################################################################
        # Retrieval stage: NVT workflow, TT model, FAISS ingest
        ####################################################################################

        (
            cls.retrieval_nvt_workflow_training,
            _,  # retrieval_nvt_workflow_serving,
        ) = _retrieval_workflow()

        cls.retrieval_workflow_path = os.path.join(
            cls.base_data_dir, "processed", "retrieval"
        )

        # This fits and saves the workflow and the saved data to cls.retrieval_workflow_path
        transform_aliccp(
            (train_raw, valid_raw),
            cls.retrieval_workflow_path,
            nvt_workflow=cls.retrieval_nvt_workflow_training,
            workflow_name="workflow_retrieval",
        )

        # TODO: This is wrong. The model should only expect user features as inputs, but it is
        # requiring the item features as well. We should be able to uncomment the two lines below
        # when it is fixed.
        cls.retrieval_nvt_workflow_serving = cls.retrieval_nvt_workflow_training

        # This also fits the scoring workflow to the same data. Messy!
        # cls.retrieval_nvt_workflow_serving = Workflow(retrieval_nvt_workflow_serving)
        # cls.retrieval_nvt_workflow_serving.fit(train_raw)

        # Train and save model.
        cls.query_tower_output_path = os.path.join(cls.base_data_dir, "two_tower_model")
        model_tt = _train_two_tower(
            os.path.join(cls.retrieval_workflow_path, "train"),
            os.path.join(cls.retrieval_workflow_path, "valid"),
            cls.query_tower_output_path,
        )

        # We will now start using this workflow for inference, so we don't want the target "click".
        cls.retrieval_nvt_workflow_training.remove_inputs(["click"])

        # Ingest the generated item embeddings into FAISS.
        cls.faiss_index_dir = os.path.join(cls.base_data_dir, "faiss")
        _configure_faiss(
            cls.faiss_index_dir,
            model_tt,
            Dataset(
                os.path.join(cls.retrieval_workflow_path, "train", "*.parquet"),
            ),
        )

        ####################################################################################
        # Ranking stage: NVT workflow, DLRM model
        ####################################################################################
        # GENERATE DATA AND NVT WORKFLOW FOR RANKING MODEL
        cls.ranking_workflow_path = os.path.join(
            cls.base_data_dir, "processed", "ranking"
        )

        cls.ranking_nvt_workflow = _ranking_workflow()
        transform_aliccp(
            (train_raw, valid_raw),
            cls.ranking_workflow_path,
            nvt_workflow=cls.ranking_nvt_workflow,
            workflow_name="workflow_ranking",
        )

        cls.dlrm_model_path = os.path.join(cls.base_data_dir, "dlrm_model")
        _train_dlrm_model(
            os.path.join(cls.ranking_workflow_path, "train"),
            os.path.join(cls.ranking_workflow_path, "valid"),
            cls.dlrm_model_path,
        )

        # remove `click` since that's not used again.
        cls.ranking_nvt_workflow.remove_inputs(["click"])

    @classmethod
    def tearDownClass(cls):
        """
        Remove the base temporary data directory after the tests.
        """
        shutil.rmtree(cls.base_data_dir)

    def test_training_data_exists(self):
        # Ensures that train/valid sets have been created.
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
            >> TransformWorkflow(MerlinTestCase.retrieval_nvt_workflow_serving)
            >> PredictTensorflow(MerlinTestCase.query_tower_output_path)
        )

        Ensemble(pipeline, request_schema)
        # TODO: run ensemble in triton, make a request.

    def test_nvt_retrieval_ensemble_from_feast(self):
        # Builds the ensemble:
        # User Id comes in, and we look up user features from Feast.
        # Features get transformed with TransformWorkflow
        # Then get fed into a retrieval model.
        # Items come out.

        request_schema = Schema(
            column_schemas=[ColumnSchema(name="user_id", dtype=np.int32)]
        )

        pipeline = (
            ["user_id"]
            >> QueryFeast.from_feature_view(
                store=FeatureStore(MerlinTestCase.feature_store_path),
                view="user_features",
                column="user_id",
                include_id=True,
            )
            >> TransformWorkflow(MerlinTestCase.retrieval_nvt_workflow_serving)
            >> PredictTensorflow(MerlinTestCase.query_tower_output_path)
        )

        Ensemble(pipeline, request_schema)

    def test_nvt_retrieval_with_faiss_ensemble(self):
        # Builds the ensemble:
        # User Id comes in, and we look up user features from Feast.
        # Features get transformed with TransformWorkflow
        # Then get fed into a retrieval model.
        # Items come out.

        request_schema = Schema(
            column_schemas=[ColumnSchema(name="user_id", dtype=np.int32)]
        )

        pipeline = (
            ["user_id"]
            >> QueryFeast.from_feature_view(
                store=FeatureStore(MerlinTestCase.feature_store_path),
                view="user_features",
                column="user_id",
                include_id=True,
            )
            >> TransformWorkflow(MerlinTestCase.retrieval_nvt_workflow_serving)
            >> PredictTensorflow(MerlinTestCase.query_tower_output_path)
            >> QueryFaiss(MerlinTestCase.faiss_index_dir, topk=10)
        )

        Ensemble(pipeline, request_schema)

    def test_whole_ensemble(self):
        # Builds the ensemble:
        # User Id comes in, and we look up user features from Feast.
        # Features get transformed with TransformWorkflow
        # Then get fed into a retrieval model.
        # Items come out.

        request_schema = Schema(
            column_schemas=[ColumnSchema(name="user_id", dtype=np.int32)]
        )

        user_features_retrieval = (
            ["user_id"]
            >> QueryFeast.from_feature_view(
                store=FeatureStore(MerlinTestCase.feature_store_path),
                view="user_features",
                column="user_id",
                include_id=True,
            )
            >> TransformWorkflow(MerlinTestCase.retrieval_nvt_workflow_serving)
        )

        retrieval_pipeline = (
            user_features_retrieval
            >> PredictTensorflow(MerlinTestCase.query_tower_output_path)
            >> QueryFaiss(MerlinTestCase.faiss_index_dir, topk=10)
        )

        item_features_ranking = retrieval_pipeline[
            "candidate_ids"
        ] >> QueryFeast.from_feature_view(
            store=FeatureStore(MerlinTestCase.feature_store_path),
            view="item_features",
            column="candidate_ids",
            include_id=True,
        )

        user_features_ranking = ["user_id"] >> QueryFeast.from_feature_view(
            store=FeatureStore(MerlinTestCase.feature_store_path),
            view="user_features",
            column="user_id",
            include_id=True,
        )

        combined_features_ranking = (
            item_features_ranking
            >> UnrollFeatures("item_id", user_features_ranking[USER_FEATURES])
            >> TransformWorkflow(MerlinTestCase.ranking_nvt_workflow)
        )
        ranking_pipeline = combined_features_ranking >> PredictTensorflow(
            MerlinTestCase.dlrm_model_path
        )
        pipeline = retrieval_pipeline + ranking_pipeline

        Ensemble(pipeline, request_schema)


def _configure_feast(feature_store_path, data_to_ingest: Dataset):
    """
    This configures a file-based Feast feature store in the directory `feast_feature_store`, which
    should be a subdirectory of MerlinTestCase.base_data_dir.

    The steps that this takes are:

    * Make the directory if it doesn't already exist.
    * Copy `user_features.py` and `item_features.py` into the directory.
    * Write the feature_store.yaml file into the directory.
    * Add timestamp columns to the user and item features.
    * Save the user and item features into the feature store directory.
    * Run `feast apply` and `feast materialize` to load data into the online store.
    """

    if not os.path.exists(feature_store_path):
        os.makedirs(os.path.join(feature_store_path, "data"))

    # Move the required files into the feature_store_path
    with open(
        os.path.join(feature_store_path, "feature_store.yaml"), "w", encoding="utf8"
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
    shutil.copy(
        os.path.join(cwd, "feast_feature_definitions", "item_features.py"),
        feature_store_path,
    )
    shutil.copy(
        os.path.join(cwd, "feast_feature_definitions", "user_features.py"),
        feature_store_path,
    )

    # Split training data into user / item and add timestamps.

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

    # These environment variables tell `user_features.py` and `item_features.py` where the parquet
    # files are stored, since it changes with every test.
    os.environ.setdefault(
        "FEAST_USER_FEATURES_PATH",
        os.path.join(feature_store_path, "data", "user_features.parquet"),
    )
    os.environ.setdefault(
        "FEAST_ITEM_FEATURES_PATH",
        os.path.join(feature_store_path, "data", "item_features.parquet"),
    )

    # Ingest data.
    subprocess.run(["feast", "-c", feature_store_path, "apply"], check=True)
    subprocess.run(
        [
            "feast",
            "-c",
            feature_store_path,
            "materialize",
            "1995-01-01T01:01:01",
            "2025-01-01T01:01:01",
        ],
        check=True,
    )

    # This is a useful place to drop a debugger and manually see what is in feast.
    # import pdb; pdb.set_trace()
    # import feast
    # fs = feast.FeatureStore(feature_store_path)
    # fs.get_online_features(["user_features:user_shops"], [{"user_id":1}]).to_dict()


def _retrieval_workflow():
    """
    NVT workflow used to transform features for retrieval model.

    Because we only use the user features at inference time, we actually define two workflows here:
    * training_workflow contains both user and item features and is used for training and for
      producing item embeddings
    * serving_workflow contains just user features and is used at inference time to produce a user
      vector in the item space.
    """
    user_id = ["user_id"] >> Categorify(dtype="int32") >> TagAsUserID()
    item_id = ["item_id"] >> Categorify(dtype="int32") >> TagAsUserID()

    item_features = ITEM_FEATURES >> Categorify(dtype="int32") >> TagAsUserFeatures()
    user_features = USER_FEATURES >> Categorify(dtype="int32") >> TagAsUserFeatures()

    training_inputs = user_id + item_id + item_features + user_features + ["click"]
    training_workflow = training_inputs >> Filter(f=lambda df: df["click"] == 1)

    serving_workflow = user_id + user_features
    return training_workflow, serving_workflow


def _ranking_workflow():
    user_id = ["user_id"] >> Categorify(dtype="int32") >> TagAsUserID()
    item_id = ["item_id"] >> Categorify(dtype="int32") >> TagAsItemID()

    item_features = ITEM_FEATURES >> Categorify(dtype="int32") >> TagAsItemFeatures()

    user_features = USER_FEATURES >> Categorify(dtype="int32") >> TagAsUserFeatures()

    targets = ["click"] >> AddMetadata(tags=[Tags.BINARY_CLASSIFICATION, "target"])

    outputs = user_id + item_id + item_features + user_features + targets
    return outputs


def _configure_faiss(faiss_index_path: str, model_tt, item_features: Dataset):
    item_embs = model_tt.item_embeddings(item_features, batch_size=1024)
    item_embeddings = np.ascontiguousarray(
        item_embs.compute(scheduler="synchronous").to_numpy()
    )
    setup_faiss(item_embeddings, faiss_index_path)


def _train_two_tower(train_data_path, valid_data_path, query_tower_output_path):
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
    query_tower.save(query_tower_output_path)

    return model_tt


def _train_dlrm_model(
    train_data_path: str, valid_data_path: str, model_output_path: str
):
    # define train and valid dataset objects
    train = Dataset(os.path.join(train_data_path, "*.parquet"), part_size="500MB")
    valid = Dataset(os.path.join(valid_data_path, "*.parquet"), part_size="500MB")

    # define schema object
    schema: Schema = train.schema

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
