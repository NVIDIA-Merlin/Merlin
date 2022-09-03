import os
import shutil
import subprocess
from datetime import datetime

import merlin.models.tf as mm
import numpy as np
import tensorflow as tf
from merlin.io.dataset import Dataset
from merlin.models.loader.tf_utils import configure_tensorflow
from merlin.models.utils.dataset import unique_rows_by_features
from merlin.schema import Schema
from merlin.schema.tags import Tags
from merlin.systems.dag.ops.faiss import setup_faiss

configure_tensorflow()


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
