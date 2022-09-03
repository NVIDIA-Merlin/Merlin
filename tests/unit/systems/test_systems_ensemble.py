import os
import shutil
import tempfile
import unittest

import numpy as np
from feast import FeatureStore
from merlin.datasets.ecommerce import transform_aliccp
from merlin.datasets.synthetic import generate_data
from merlin.io.dataset import Dataset
from merlin.schema import ColumnSchema, Schema
from merlin.schema.tags import Tags
from merlin.systems.dag.ensemble import Ensemble
from merlin.systems.dag.ops.faiss import QueryFaiss
from merlin.systems.dag.ops.feast import QueryFeast
from merlin.systems.dag.ops.tensorflow import PredictTensorflow
from merlin.systems.dag.ops.unroll_features import UnrollFeatures
from merlin.systems.dag.ops.workflow import TransformWorkflow
from nvtabular.ops import (
    AddMetadata,
    Categorify,
    Filter,
    TagAsItemFeatures,
    TagAsItemID,
    TagAsUserFeatures,
    TagAsUserID,
)
from nvtabular.workflow import Workflow

from .configuration_helpers import (
    _configure_faiss,
    _configure_feast,
    _train_dlrm_model,
    _train_two_tower,
)

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
            retrieval_nvt_workflow_serving,
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

        # This also fits the scoring workflow to the same data. Messy!
        cls.retrieval_nvt_workflow_serving = Workflow(retrieval_nvt_workflow_serving)
        cls.retrieval_nvt_workflow_serving.fit(train_raw)

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
    item_id = ["item_id"] >> Categorify(dtype="int32") >> TagAsItemID()

    item_features = ITEM_FEATURES >> Categorify(dtype="int32") >> TagAsItemFeatures()
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
