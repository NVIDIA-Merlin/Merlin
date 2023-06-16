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
        ]
    )

    assert set(schema.select_by_tag(Tags.USER_ID).column_names) == set(["user_id"])
    assert set(schema.select_by_tag(Tags.ITEM_ID).column_names) == set(["item_id"])
    assert set(schema.select_by_tag(Tags.CATEGORICAL).column_names) == set(
        ["user_id", "item_id", "video_category", "gender", "age"]
    )
    assert set(schema.select_by_tag(Tags.CONTINUOUS).column_names) == set([])
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
            filter_query="click==1 or (click==0 and follow==0 and like==0 and share==0)",
            min_item_freq=30,
            min_user_freq=30,
            max_user_freq=150,
            num_max_rounds_filtering=5,
            output_path=tmp_output_folder,
            categorical_features="user_id,item_id,video_category,gender,age",
            binary_classif_targets="click,follow,like,share",
            regression_targets="watching_times",
            to_int32="user_id,item_id",
            to_int16="watching_times",
            to_int8="gender,age,video_category,click,follow,like,share",
            user_id_feature="user_id",
            item_id_feature="item_id",
            dataset_split_strategy="random_by_user",
            random_split_eval_perc=0.2,
            **additional_kwargs,
        )
        runner = PreprocessingRunner(args)
        runner.run()

        expected_max_values = {
            "user_id": 69719,
            "item_id": 19174,
            "video_category": 2,
            "gender": 3,
            "age": 8,
            "click": 1,
            "follow": 1,
            "like": 1,
            "share": 1,
            "watching_times": 121,
        }

        check_schema(os.path.join(tmp_output_folder, "train/"), expected_max_values)

        expected_dtypes = {
            "user_id": np.dtype("int64"),
            "item_id": np.dtype("int64"),
            "video_category": np.dtype("int64"),
            "gender": np.dtype("int64"),
            "age": np.dtype("int64"),
            "click": np.dtype("int8"),
            "follow": np.dtype("int8"),
            "like": np.dtype("int8"),
            "share": np.dtype("int8"),
        }

        train_df = cudf.read_parquet(os.path.join(tmp_output_folder, "train/*.parquet"))
        assert len(train_df) == 2440155  # row count
        assert not train_df.isna().max().max()  # Check if there are null values
        assert train_df.max().to_dict() == expected_max_values
        assert train_df.dtypes.to_dict() == expected_dtypes

        eval_df = cudf.read_parquet(os.path.join(tmp_output_folder, "eval/*.parquet"))
        assert len(eval_df) == 577760
        assert not eval_df.isna().max().max()
        assert eval_df.max().to_dict() == expected_max_values
        assert eval_df.dtypes.to_dict() == expected_dtypes


@pytest.mark.parametrize(
    "split_strategy", [None, "random", "random_by_user", "temporal"]
)
def test_ranking_preprocessing(tenrec_data_path, split_strategy):
    with tempfile.TemporaryDirectory() as tmp_output_folder:
        additional_kwargs = {}
        if split_strategy in ["random", "random_by_user"]:
            additional_kwargs["random_split_eval_perc"] = 0.2
        elif split_strategy == "temporal":
            additional_kwargs["item_id"] = 15000

        args = kwargs_to_cli_ags(
            data_path=os.path.join(tenrec_data_path, "raw/QK-video-10M.csv"),
            input_data_format="csv",
            csv_na_values="\\N",
            filter_query="click==1 or (click==0 and follow==0 and like==0 and share==0)",
            min_item_freq=30,
            min_user_freq=30,
            max_user_freq=150,
            num_max_rounds_filtering=5,
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

        total_rows = 3017915

        train_df = cudf.read_parquet(os.path.join(tmp_output_folder, "train/*.parquet"))
        rows_train = len(train_df)
        if split_strategy is None:
            assert len(train_df) == total_rows
        else:
            eval_df = cudf.read_parquet(
                os.path.join(tmp_output_folder, "eval/*.parquet")
            )
            rows_eval = len(eval_df)

            assert rows_train + rows_eval == total_rows

            if split_strategy in ["random", "random_by_user"]:
                assert rows_train == int(total_rows * 0.8)
                assert rows_eval == int(total_rows * 0.2)
            elif split_strategy == "temporal":
                assert rows_train == 3017915
                assert rows_eval == 3017915
