import os

import pytest
from testbook import testbook
from tests.conftest import REPO_ROOT


def test_func():
    with testbook(
        REPO_ROOT / "examples" / "scaling-criteo" / "02-ETL-with-NVTabular.ipynb",
        execute=False,
        timeout=180,
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
        / "03-Training-with-Merlin-Models-TensorFlow.ipynb",
        execute=False,
        timeout=180,
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
                "regularization_loss",
            ]
        )
        assert os.path.isfile("/tmp/output/criteo/dlrm/saved_model.pb")

    with testbook(
        REPO_ROOT
        / "examples"
        / "scaling-criteo"
        / "04-Triton-Inference-with-Merlin-Models-TensorFlow.ipynb",
        execute=False,
        timeout=180,
    ) as tb3:
        tb3.inject(
            """
            import os
            os.environ["BASE_DIR"] = "/tmp/output/criteo/"
            os.environ["INPUT_FOLDER"] = "/tmp/input/criteo/"
            """
        )
        NUM_OF_CELLS = len(tb3.cells)
        tb3.execute_cell(list(range(0, NUM_OF_CELLS - 5)))
        tb3.inject(
            """
            import shutil
            
            from merlin.systems.triton.utils import run_ensemble_on_tritonserver
            outputs = ensemble.graph.output_schema.column_names
            response = run_ensemble_on_tritonserver(
                "/tmp/output/criteo/ensemble/", outputs, batch.fillna(0), "ensemble_model"
            )
            response = [x.tolist()[0] for x in response["label/binary_classification_task"]]
            shutil.rmtree("/tmp/input/criteo", ignore_errors=True)
            shutil.rmtree("/tmp/output/criteo", ignore_errors=True)
            """
        )
        tb3.execute_cell(NUM_OF_CELLS - 4)
        response = tb3.ref("response")
        assert len(response) == 3
