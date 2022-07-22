import os

from testbook import testbook

from tests.conftest import REPO_ROOT

import pytest

def test_func():
    with testbook(
        REPO_ROOT
        / "examples"
        / "scaling-criteo"
        / "02-ETL-with-NVTabular.ipynb",
        execute=False,
    ) as tb1:
        tb1.inject(
            """
            import os
            os.environ["BASE_DIR"] = "/tmp/input/criteo/"
            os.environ["INPUT_DATA_DIR"] = "/tmp/input/criteo/"
            os.environ["OUTPUT_DATA_DIR"] = "/tmp/output/criteo/"
            os.system("mkdir -p /tmp/input/criteo")
            os.system("mkdir -p /tmp/output/criteo")

            from merlin.datasets.synthetic import generate_data

            train, valid = generate_data("criteo", int(1000000), set_sizes=(0.7, 0.3))

            train.to_ddf().compute().to_parquet('/tmp/input/criteo/day_0.parquet')
            valid.to_ddf().compute().to_parquet('/tmp/input/criteo/day_1.parquet')
            """
        )
        tb1.execute()
        assert os.path.isfile("/tmp/output/criteo/train/part_0.parquet")
        assert os.path.isfile("/tmp/output/criteo/valid/part_0.parquet")
        assert os.path.isfile("/tmp/output/criteo/workflow/metadata.json")
        
    with testbook(
        REPO_ROOT
        / "examples"
        / "scaling-criteo"
        / "03-Training-with-Merlin-Models.ipynb",
        execute=False,
    ) as tb2:
        tb2.inject(
            """
            import os
            os.environ["INPUT_DATA_DIR"] = "/tmp/output/criteo/"
            """
        )
        tb2.execute()
        metrics = tb2.ref("eval_metrics")
        assert set(metrics.keys()) == set(
            [
                "auc",
                "binary_accuracy",
                "loss",
                "precision",
                "recall",
                "regularization_loss"
            ]
        )
        assert os.path.isfile("/tmp/output/criteo/dlrm/saved_model.pb")