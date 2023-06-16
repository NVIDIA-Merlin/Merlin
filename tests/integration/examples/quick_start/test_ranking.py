import os

import pytest
from examples.quick_start.scripts.ranking.ranking import RankingTrainEvalRunner

STANDARD_CI_TENREC_DATA_PATH = "/raid/data/tenrec_ci/"


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


def test_ranking_train_eval(tenrec_data_path):
    args = kwargs_to_cli_ags()
    # TODO: Load datasets from the pre-trained path preproc/train & preproc/eval
    train_ds, eval_ds, predict_ds = None, None, None
    runner = RankingTrainEvalRunner(args, train_ds, eval_ds, predict_ds, logger=None)
    metrics = runner.run()
