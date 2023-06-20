import os
import tempfile

import cudf
import numpy as np
import pytest
from examples.quick_start.scripts.preproc.preprocessing import PreprocessingRunner
from merlin.schema import Tags
from merlin.schema.io.tensorflow_metadata import TensorflowMetadata

STANDARD_CI_TENREC_DATA_PATH = "/raid/data/tenrec_ci/"


def kwargs_to_cli_ags(**kwargs):
    cli_args = []
    for k, v in kwargs.items():
        cli_args.append(f"--{k}")
        if v is not None:
            cli_args.append(str(v))
    args = PreprocessingRunner.parse_cli_args(cli_args)
    return args


@pytest.fixture
def tenrec_data_path():
    data_path = os.getenv("CI_TENREC_DATA_PATH", STANDARD_CI_TENREC_DATA_PATH)
    return data_path


def get_schema_from_path(path):
    tf_metadata = TensorflowMetadata.from_proto_text_file(str(path))
    schema = tf_metadata.to_merlin_schema()
    return schema


def check_schema(path, categ_cols_max_values=None):
    schema = get_schema_from_path(path)
    assert set(schema.column_names) == set(
        [
            "user_id",
            "item_id",
            "video_category",
            "gender",
            "age",
            "click",
            "follow",
            "like",
            "share",
            "watching_times",
            "TE_user_id_follow",
            "TE_item_id_follow",
            "TE_user_id_click",
            "TE_item_id_click",
        ]
    )

    assert set(schema.select_by_tag(Tags.USER_ID).column_names) == set(["user_id"])
    assert set(schema.select_by_tag(Tags.ITEM_ID).column_names) == set(["item_id"])
    assert set(schema.select_by_tag(Tags.CATEGORICAL).column_names) == set(
        ["user_id", "item_id", "video_category", "gender"]
    )
    target_encoding_feats = [
        "TE_user_id_click",
        "TE_user_id_follow",
        "TE_item_id_click",
        "TE_item_id_follow",
    ]
    assert set(schema.select_by_tag(Tags.CONTINUOUS).column_names) == set(
        ["age"] + target_encoding_feats
    )
    assert set(schema.select_by_tag(Tags.BINARY_CLASSIFICATION).column_names) == set(
        ["click", "follow", "like", "share"]
    )
    assert set(schema.select_by_tag(Tags.REGRESSION).column_names) == set(
        ["watching_times"]
    )
    assert set(schema.select_by_tag(Tags.TARGET).column_names) == set(
        ["click", "follow", "like", "share", "watching_times"]
    )

    if categ_cols_max_values:
        categ_features = schema.select_by_tag(Tags.CATEGORICAL).column_names
        for col in categ_features:
            assert schema[col].int_domain.max == categ_cols_max_values[col]

    return schema


@pytest.mark.parametrize("use_dask_cluster", [True, False])
def test_ranking_preprocessing(tenrec_data_path, use_dask_cluster):
    with tempfile.TemporaryDirectory() as tmp_output_folder:
        additional_kwargs = {}
        if use_dask_cluster:
            additional_kwargs["enable_dask_cuda_cluster"] = None
            additional_kwargs["persist_intermediate_files"] = None

        args = kwargs_to_cli_ags(
            data_path=os.path.join(tenrec_data_path, "raw/QK-video-10M.csv"),
            input_data_format="csv",
            csv_na_values="\\N",
            output_path=tmp_output_folder,
            categorical_features="user_id,item_id,video_category,gender",
            continuous_features="age",
            target_encoding_features="user_id,item_id",
            target_encoding_targets="click,follow",
            binary_classif_targets="click,follow,like,share",
            regression_targets="watching_times",
            to_int32="user_id,item_id",
            to_int16="watching_times",
            to_int8="gender,age,video_category,click,follow,like,share",
            user_id_feature="user_id",
            item_id_feature="item_id",
            **additional_kwargs,
        )
        runner = PreprocessingRunner(args)
        runner.run()

        expected_max_values = {
            "user_id": 296088,
            "item_id": 617033,
            "video_category": 2,
            "gender": 3,
            "click": 1,
            "follow": 1,
            "like": 1,
            "share": 1,
            "watching_times": 528,
        }

        schema = check_schema(
            os.path.join(tmp_output_folder, "train/"), expected_max_values
        )

        expected_dtypes = {
            "user_id": np.dtype("int64"),
            "item_id": np.dtype("int64"),
            "video_category": np.dtype("int64"),
            "gender": np.dtype("int64"),
            "age": np.dtype("float64"),
            "TE_user_id_click": np.dtype("float32"),
            "TE_item_id_click": np.dtype("float32"),
            "TE_user_id_follow": np.dtype("float32"),
            "TE_item_id_follow": np.dtype("float32"),
            "click": np.dtype("int8"),
            "follow": np.dtype("int8"),
            "like": np.dtype("int8"),
            "share": np.dtype("int8"),
            "watching_times": np.dtype("int16"),
        }

        train_df = cudf.read_parquet(os.path.join(tmp_output_folder, "train/*.parquet"))
        assert not train_df.isna().max().max()  # Check if there are null values
        assert len(train_df) == 10000000  # row count

        assert train_df.dtypes.to_dict() == expected_dtypes

        categ_features = schema.select_by_tag(Tags.CATEGORICAL).column_names
        target_features = schema.select_by_tag(Tags.TARGET).column_names
        assert (
            train_df[categ_features + target_features].max().to_dict()
            == expected_max_values
        )

        # Checking age standardization
        assert 0.0 == pytest.approx(train_df["age"].mean(), abs=1e-3)
        assert 1.0 == pytest.approx(train_df["age"].std(), abs=1e-3)

        # Check target encoding features
        te_features = [
            "TE_user_id_follow",
            "TE_item_id_follow",
            "TE_user_id_click",
            "TE_item_id_click",
        ]
        assert (
            train_df[te_features].min().min() >= 0
            and train_df[te_features].max().max() <= 1
        )


@pytest.mark.parametrize("split_strategy", ["random", "random_by_user", "temporal"])
def test_ranking_preprocessing_split_strategies(tenrec_data_path, split_strategy):
    with tempfile.TemporaryDirectory() as tmp_output_folder:
        additional_kwargs = {}
        if split_strategy in ["random", "random_by_user"]:
            additional_kwargs["random_split_eval_perc"] = 0.2
        elif split_strategy == "temporal":
            additional_kwargs["timestamp_feature"] = "item_id"
            additional_kwargs["dataset_split_temporal_timestamp"] = 15000

        args = kwargs_to_cli_ags(
            data_path=os.path.join(tenrec_data_path, "raw/QK-video-10M.csv"),
            input_data_format="csv",
            csv_na_values="\\N",
            output_path=tmp_output_folder,
            categorical_features="user_id,item_id,video_category,gender,age",
            binary_classif_targets="click,follow,like,share",
            regression_targets="watching_times",
            to_int32="user_id,item_id",
            to_int16="watching_times",
            to_int8="gender,age,video_category,click,follow,like,share",
            user_id_feature="user_id",
            item_id_feature="item_id",
            dataset_split_strategy=split_strategy,
            **additional_kwargs,
        )
        runner = PreprocessingRunner(args)
        runner.run()

        total_rows = 10000000

        train_df = cudf.read_parquet(os.path.join(tmp_output_folder, "train/*.parquet"))
        rows_train = len(train_df)

        eval_df = cudf.read_parquet(os.path.join(tmp_output_folder, "eval/*.parquet"))
        rows_eval = len(eval_df)

        assert rows_train + rows_eval == total_rows

        if split_strategy in ["random", "random_by_user"]:
            assert 0.20 == pytest.approx(rows_eval / float(total_rows), abs=0.01)

            if split_strategy == "random_by_user":
                assert train_df["user_id"].nunique() == pytest.approx(
                    eval_df["user_id"].nunique(), rel=0.05
                )

        elif split_strategy == "temporal":
            assert rows_train == 4636381
            assert rows_eval == 5363619


def test_ranking_preprocessing_filter_strategies(tenrec_data_path):
    with tempfile.TemporaryDirectory() as tmp_output_folder:
        args = kwargs_to_cli_ags(
            data_path=os.path.join(tenrec_data_path, "raw/QK-video-10M.csv"),
            input_data_format="csv",
            csv_na_values="\\N",
            output_path=tmp_output_folder,
            categorical_features="user_id,item_id,video_category,gender,age",
            binary_classif_targets="click,follow,like,share",
            regression_targets="watching_times",
            to_int32="user_id,item_id",
            to_int16="watching_times",
            to_int8="gender,age,video_category,click,follow,like,share",
            user_id_feature="user_id",
            item_id_feature="item_id",
            filter_query="click==1 or (click==0 and follow==0 and like==0 and share==0)",
            min_item_freq=5,
            min_user_freq=5,
            max_user_freq=200,
            num_max_rounds_filtering=5,
        )
        runner = PreprocessingRunner(args)
        runner.run()

        total_rows = 9102904

        train_df = cudf.read_parquet(os.path.join(tmp_output_folder, "train/*.parquet"))
        assert len(train_df) == total_rows

        assert train_df.groupby("item_id").size().min() >= 5
        assert train_df.groupby("user_id").size().min() >= 5


def test_ranking_preprocessing_freq_capping(tenrec_data_path):
    with tempfile.TemporaryDirectory() as tmp_output_folder:
        args = kwargs_to_cli_ags(
            data_path=os.path.join(tenrec_data_path, "raw/QK-video-10M.csv"),
            input_data_format="csv",
            csv_na_values="\\N",
            output_path=tmp_output_folder,
            categorical_features="user_id,item_id,video_category,gender,age",
            binary_classif_targets="click,follow,like,share",
            regression_targets="watching_times",
            to_int32="user_id,item_id",
            to_int16="watching_times",
            to_int8="gender,age,video_category,click,follow,like,share",
            user_id_feature="user_id",
            item_id_feature="item_id",
            categ_min_freq_capping=30,
        )
        runner = PreprocessingRunner(args)
        runner.run()

        total_rows = 10000000

        train_df = cudf.read_parquet(os.path.join(tmp_output_folder, "train/*.parquet"))
        assert len(train_df) == total_rows

        assert train_df.groupby("item_id").size().min() >= 30
        assert train_df.groupby("user_id").size().min() >= 30
