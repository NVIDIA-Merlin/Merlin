import os
import tempfile
import time

import pytest
from examples.quick_start.scripts.ranking.ranking import RankingTrainEvalRunner
from merlin.io.dataset import Dataset

# STANDARD_CI_TENREC_DATA_PATH = "/raid/data/tenrec_ci/"

STANDARD_CI_TENREC_DATA_PATH = "/data_ci/"


@pytest.fixture
def tenrec_data_path():
    data_path = os.getenv("CI_TENREC_DATA_PATH", STANDARD_CI_TENREC_DATA_PATH)
    return data_path


def kwargs_to_cli_ags(**kwargs):
    cli_args = []
    for k, v in kwargs.items():
        cli_args.append(f"--{k}")
        if v is not None:
            cli_args.append(str(v))
    args = RankingTrainEvalRunner.parse_cli_args(cli_args)
    return args


def get_datasets(path):
    train_ds = Dataset(os.path.join(path, "preproc/train/*.parquet"), part_size="500MB")
    eval_ds = Dataset(os.path.join(path, "preproc/eval/*.parquet"), part_size="500MB")
    return train_ds, eval_ds


def test_ranking_single_task_mlp(tenrec_data_path):
    args = kwargs_to_cli_ags()
    train_ds, eval_ds = get_datasets(tenrec_data_path)

    with tempfile.TemporaryDirectory() as tmp_output_folder:
        args = kwargs_to_cli_ags(
            output_path=tmp_output_folder,
            tasks="click",
            stl_positive_class_weight=3,
            model="mlp",
            mlp_layers="64,32",
            embedding_sizes_multiplier=6,
            l2_reg=1e-6,
            embeddings_l2_reg=1e-8,
            dropout=0.05,
            lr=1e-4,
            lr_decay_rate=0.99,
            lr_decay_steps=100,
            train_batch_size=8192,
            eval_batch_size=8192,
            epoch=3,
        )

        runner = RankingTrainEvalRunner(args, train_ds, train_ds, None, logger=None)

        current_time = time.time()

        metrics = runner.run()

        elapsed_time = time.time() - current_time

        assert set(metrics.keys()) == set(
            ["loss", "auc", "prauc", "logloss", "regularization_loss", "loss_batch"]
        )

        assert metrics["loss"] < 0.7
        assert metrics["logloss"] < 0.7
        assert metrics["auc"] > 0.75
        assert metrics["prauc"] > 0.60
        assert metrics["regularization_loss"] > 0.0
        assert metrics["loss_batch"] < 0.8
        # assert elapsed_time < 60  # 23s in a V100


def test_ranking_single_task_dlrm(tenrec_data_path):
    args = kwargs_to_cli_ags()
    train_ds, eval_ds = get_datasets(tenrec_data_path)

    with tempfile.TemporaryDirectory() as tmp_output_folder:
        args = kwargs_to_cli_ags(
            output_path=tmp_output_folder,
            tasks="click",
            stl_positive_class_weight=3,
            model="dlrm",
            embeddings_dim=64,
            l2_reg=1e-6,
            embeddings_l2_reg=1e-8,
            dropout=0.05,
            mlp_layers="64,32",
            lr=1e-4,
            lr_decay_rate=0.99,
            lr_decay_steps=100,
            train_batch_size=8192,
            eval_batch_size=8192,
            epoch=3,
        )

        runner = RankingTrainEvalRunner(args, train_ds, train_ds, None, logger=None)

        current_time = time.time()

        metrics = runner.run()

        elapsed_time = time.time() - current_time

        assert set(metrics.keys()) == set(
            ["loss", "auc", "prauc", "logloss", "regularization_loss", "loss_batch"]
        )

        assert metrics["loss"] < 0.7
        assert metrics["logloss"] < 0.7
        assert metrics["auc"] > 0.75
        assert metrics["prauc"] > 0.60
        assert metrics["regularization_loss"] > 0.0
        assert metrics["loss_batch"] < 0.8
        # assert elapsed_time < 60


def test_ranking_single_task_dcn(tenrec_data_path):
    args = kwargs_to_cli_ags()
    train_ds, eval_ds = get_datasets(tenrec_data_path)

    with tempfile.TemporaryDirectory() as tmp_output_folder:
        args = kwargs_to_cli_ags(
            output_path=tmp_output_folder,
            tasks="click",
            stl_positive_class_weight=3,
            model="dcn",
            dcn_interacted_layer_num=5,
            mlp_layers="64,32",
            embedding_sizes_multiplier=6,
            l2_reg=1e-6,
            embeddings_l2_reg=1e-8,
            dropout=0.05,
            lr=1e-4,
            lr_decay_rate=0.99,
            lr_decay_steps=100,
            train_batch_size=8192,
            eval_batch_size=8192,
            epoch=3,
        )

        runner = RankingTrainEvalRunner(args, train_ds, train_ds, None, logger=None)

        current_time = time.time()

        metrics = runner.run()

        elapsed_time = time.time() - current_time

        assert set(metrics.keys()) == set(
            ["loss", "auc", "prauc", "logloss", "regularization_loss", "loss_batch"]
        )

        assert metrics["loss"] < 0.7
        assert metrics["logloss"] < 0.7
        assert metrics["auc"] > 0.75
        assert metrics["prauc"] > 0.60
        assert metrics["regularization_loss"] > 0.0
        assert metrics["loss_batch"] < 0.8
        # assert elapsed_time < 60


def test_ranking_single_task_wide_n_deep(tenrec_data_path):
    args = kwargs_to_cli_ags()
    train_ds, eval_ds = get_datasets(tenrec_data_path)

    with tempfile.TemporaryDirectory() as tmp_output_folder:
        args = kwargs_to_cli_ags(
            output_path=tmp_output_folder,
            tasks="click",
            stl_positive_class_weight=3,
            model="wide_n_deep",
            wnd_hashed_cross_num_bins=5000,
            wnd_ignore_combinations="item_id:video_category,user_id:gender,user_id:age",
            wnd_wide_l2_reg=1e-5,
            mlp_layers="64,32",
            embedding_sizes_multiplier=6,
            l2_reg=1e-6,
            embeddings_l2_reg=1e-8,
            dropout=0.05,
            lr=1e-4,
            lr_decay_rate=0.99,
            lr_decay_steps=100,
            train_batch_size=8192,
            eval_batch_size=8192,
            epoch=3,
        )

        runner = RankingTrainEvalRunner(args, train_ds, train_ds, None, logger=None)

        current_time = time.time()

        metrics = runner.run()

        elapsed_time = time.time() - current_time

        assert set(metrics.keys()) == set(
            ["loss", "auc", "prauc", "logloss", "regularization_loss", "loss_batch"]
        )

        assert metrics["loss"] < 0.7
        assert metrics["logloss"] < 0.7
        assert metrics["auc"] > 0.75
        assert metrics["prauc"] > 0.60
        assert metrics["regularization_loss"] > 0.0
        assert metrics["loss_batch"] < 0.8
        # assert elapsed_time < 60


def test_ranking_single_task_deepfm(tenrec_data_path):
    args = kwargs_to_cli_ags()
    train_ds, eval_ds = get_datasets(tenrec_data_path)

    with tempfile.TemporaryDirectory() as tmp_output_folder:
        args = kwargs_to_cli_ags(
            output_path=tmp_output_folder,
            tasks="click",
            stl_positive_class_weight=3,
            model="deepfm",
            mlp_layers="64,32",
            embedding_sizes_multiplier=6,
            l2_reg=1e-6,
            embeddings_l2_reg=1e-8,
            dropout=0.05,
            lr=1e-4,
            lr_decay_rate=0.99,
            lr_decay_steps=100,
            train_batch_size=8192,
            eval_batch_size=8192,
            epoch=3,
        )

        runner = RankingTrainEvalRunner(args, train_ds, train_ds, None, logger=None)

        current_time = time.time()

        metrics = runner.run()

        elapsed_time = time.time() - current_time

        assert set(metrics.keys()) == set(
            ["loss", "auc", "prauc", "logloss", "regularization_loss", "loss_batch"]
        )

        assert metrics["loss"] < 0.7
        assert metrics["logloss"] < 0.7
        assert metrics["auc"] > 0.75
        assert metrics["prauc"] > 0.60
        assert metrics["regularization_loss"] > 0.0
        assert metrics["loss_batch"] < 0.8
        # assert elapsed_time < 120


def test_ranking_multi_task_dlrm(tenrec_data_path):
    args = kwargs_to_cli_ags()
    train_ds, eval_ds = get_datasets(tenrec_data_path)

    with tempfile.TemporaryDirectory() as tmp_output_folder:
        args = kwargs_to_cli_ags(
            output_path=tmp_output_folder,
            tasks="click,follow,watching_times",
            mtl_pos_class_weight_click=1,
            mtl_pos_class_weight_like=2,
            mtl_loss_weight_click=1,
            mtl_loss_weight_follow=2,
            mtl_loss_weight_watching_times=5,
            use_task_towers=True,
            tower_layers=64,
            model="dlrm",
            embeddings_dim=64,
            l2_reg=1e-6,
            embeddings_l2_reg=1e-8,
            dropout=0.05,
            mlp_layers="64,32",
            lr=1e-4,
            lr_decay_rate=0.99,
            lr_decay_steps=100,
            train_batch_size=8192,
            eval_batch_size=8192,
            epoch=3,
        )

        runner = RankingTrainEvalRunner(args, train_ds, train_ds, None, logger=None)

        current_time = time.time()

        metrics = runner.run()

        elapsed_time = time.time() - current_time

        assert set(metrics.keys()) == set(
            [
                "loss",
                "click/binary_output_loss",
                "follow/binary_output_loss",
                "click/binary_output/auc",
                "click/binary_output/prauc",
                "click/binary_output/logloss",
                "follow/binary_output/auc",
                "follow/binary_output/prauc",
                "follow/binary_output/logloss",
                "watching_times/regression_output_loss",
                "watching_times/regression_output/root_mean_squared_error",
                "regularization_loss",
                "loss_batch",
            ]
        )

        assert metrics["loss"] < 4
        assert metrics["click/binary_output_loss"] < 0.7
        assert metrics["follow/binary_output_loss"] < 0.1
        assert metrics["watching_times/regression_output_loss"] < 0.6
        assert metrics["click/binary_output/auc"] > 0.65
        assert metrics["click/binary_output/prauc"] > 0.5
        assert metrics["click/binary_output/logloss"] < 0.65
        assert metrics["follow/binary_output/auc"] > 0.35
        assert metrics["follow/binary_output/prauc"] > 0
        assert metrics["follow/binary_output/logloss"] < 0.1
        assert metrics["watching_times/regression_output/root_mean_squared_error"] < 0.8
        assert metrics["regularization_loss"] > 0.0
        assert metrics["loss_batch"] > 0.0
        # assert elapsed_time < 60


def test_ranking_multi_task_mmoe(tenrec_data_path):
    args = kwargs_to_cli_ags()
    train_ds, eval_ds = get_datasets(tenrec_data_path)

    with tempfile.TemporaryDirectory() as tmp_output_folder:
        args = kwargs_to_cli_ags(
            output_path=tmp_output_folder,
            tasks="click,follow,watching_times",
            mtl_pos_class_weight_click=1,
            mtl_pos_class_weight_like=2,
            mtl_loss_weight_click=1,
            mtl_loss_weight_follow=2,
            mtl_loss_weight_watching_times=5,
            use_task_towers=True,
            tower_layers=64,
            model="mmoe",
            mmoe_num_mlp_experts=4,
            embedding_sizes_multiplier=5,
            l2_reg=1e-6,
            embeddings_l2_reg=1e-8,
            dropout=0.05,
            mlp_layers="64,32",
            lr=1e-4,
            lr_decay_rate=0.99,
            lr_decay_steps=100,
            train_batch_size=8192,
            eval_batch_size=8192,
            epoch=3,
        )

        runner = RankingTrainEvalRunner(args, train_ds, train_ds, None, logger=None)

        current_time = time.time()

        metrics = runner.run()

        elapsed_time = time.time() - current_time

        assert set(metrics.keys()) == set(
            [
                "loss",
                "click/binary_output_loss",
                "follow/binary_output_loss",
                "click/binary_output/auc",
                "click/binary_output/prauc",
                "click/binary_output/logloss",
                "follow/binary_output/auc",
                "follow/binary_output/prauc",
                "follow/binary_output/logloss",
                "watching_times/regression_output_loss",
                "watching_times/regression_output/root_mean_squared_error",
                "regularization_loss",
                "loss_batch",
                "gate_click/binary_output_weight_0",
                "gate_click/binary_output_weight_1",
                "gate_click/binary_output_weight_2",
                "gate_click/binary_output_weight_3",
                "gate_follow/binary_output_weight_0",
                "gate_follow/binary_output_weight_1",
                "gate_follow/binary_output_weight_2",
                "gate_follow/binary_output_weight_3",
                "gate_watching_times/regression_output_weight_0",
                "gate_watching_times/regression_output_weight_1",
                "gate_watching_times/regression_output_weight_2",
                "gate_watching_times/regression_output_weight_3",
            ]
        )

        assert metrics["loss"] < 4
        assert metrics["click/binary_output_loss"] < 0.7
        assert metrics["follow/binary_output_loss"] < 0.1
        assert metrics["watching_times/regression_output_loss"] < 0.6
        assert metrics["click/binary_output/auc"] > 0.65
        assert metrics["click/binary_output/prauc"] > 0.5
        assert metrics["click/binary_output/logloss"] < 0.65
        assert metrics["follow/binary_output/auc"] > 0.35
        assert metrics["follow/binary_output/prauc"] > 0
        assert metrics["follow/binary_output/logloss"] < 0.1
        assert metrics["watching_times/regression_output/root_mean_squared_error"] < 0.8
        assert metrics["regularization_loss"] > 0.0
        assert metrics["loss_batch"] > 0.0
        # assert elapsed_time < 60


def test_ranking_multi_task_ple(tenrec_data_path):
    args = kwargs_to_cli_ags()
    train_ds, eval_ds = get_datasets(tenrec_data_path)

    with tempfile.TemporaryDirectory() as tmp_output_folder:
        args = kwargs_to_cli_ags(
            output_path=tmp_output_folder,
            tasks="click,follow,watching_times",
            mtl_pos_class_weight_click=1,
            mtl_pos_class_weight_like=2,
            mtl_loss_weight_click=1,
            mtl_loss_weight_follow=2,
            mtl_loss_weight_watching_times=5,
            ple_num_layers=2,
            use_task_towers=True,
            tower_layers=64,
            model="ple",
            cgc_num_shared_experts=3,
            cgc_num_task_experts=1,
            embedding_sizes_multiplier=5,
            l2_reg=1e-6,
            embeddings_l2_reg=1e-8,
            dropout=0.05,
            mlp_layers="64,32",
            lr=1e-4,
            lr_decay_rate=0.99,
            lr_decay_steps=100,
            train_batch_size=8192,
            eval_batch_size=8192,
            epoch=3,
        )

        runner = RankingTrainEvalRunner(args, train_ds, train_ds, None, logger=None)

        current_time = time.time()

        metrics = runner.run()

        elapsed_time = time.time() - current_time

        assert set(metrics.keys()) == set(
            [
                "loss",
                "click/binary_output_loss",
                "follow/binary_output_loss",
                "click/binary_output/auc",
                "click/binary_output/prauc",
                "click/binary_output/logloss",
                "follow/binary_output/auc",
                "follow/binary_output/prauc",
                "follow/binary_output/logloss",
                "watching_times/regression_output_loss",
                "watching_times/regression_output/root_mean_squared_error",
                "regularization_loss",
                "loss_batch",
                "gate_click/binary_output_weight_0",
                "gate_click/binary_output_weight_1",
                "gate_click/binary_output_weight_2",
                "gate_click/binary_output_weight_3",
                "gate_follow/binary_output_weight_0",
                "gate_follow/binary_output_weight_1",
                "gate_follow/binary_output_weight_2",
                "gate_follow/binary_output_weight_3",
                "shared_gate_weight_0",
                "shared_gate_weight_1",
                "shared_gate_weight_2",
                "shared_gate_weight_3",
                "shared_gate_weight_4",
                "shared_gate_weight_5",
                "gate_watching_times/regression_output_weight_0",
                "gate_watching_times/regression_output_weight_1",
                "gate_watching_times/regression_output_weight_2",
                "gate_watching_times/regression_output_weight_3",
            ]
        )

        assert metrics["loss"] < 4
        assert metrics["click/binary_output_loss"] < 0.7
        assert metrics["follow/binary_output_loss"] < 0.1
        assert metrics["watching_times/regression_output_loss"] < 0.6
        assert metrics["click/binary_output/auc"] > 0.65
        assert metrics["click/binary_output/prauc"] > 0.5
        assert metrics["click/binary_output/logloss"] < 0.65
        assert metrics["follow/binary_output/auc"] > 0.35
        assert metrics["follow/binary_output/prauc"] > 0
        assert metrics["follow/binary_output/logloss"] < 0.1
        assert metrics["watching_times/regression_output/root_mean_squared_error"] < 0.8
        assert metrics["regularization_loss"] > 0.0
        assert metrics["loss_batch"] > 0.0
        # assert elapsed_time < 60
