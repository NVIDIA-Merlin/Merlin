import os

import pytest
from testbook import testbook
from tests.conftest import REPO_ROOT

pytest.importorskip("hugectr")


def test_test_scaling_criteo_merlin_models_hugectr():
    with testbook(
        REPO_ROOT / "examples" / "scaling-criteo" / "02-ETL-with-NVTabular.ipynb",
        execute=False,
        timeout=180,
    ) as tb1:
        tb1.inject(
            """
            import os
            os.environ["BASE_DIR"] = "/tmp/test_merlin_criteo_hugectr/input/criteo/"
            os.environ["INPUT_DATA_DIR"] = "/tmp/test_merlin_criteo_hugectr/input/criteo/"
            os.environ["OUTPUT_DATA_DIR"] = "/tmp/test_merlin_criteo_hugectr/output/criteo/"
            os.environ["USE_HUGECTR"] = "True"
            
            os.system("mkdir -p /tmp/test_merlin_criteo_hugectr/input/criteo")
            os.system("mkdir -p /tmp/test_merlin_criteo_hugectr/output/criteo/")

            from merlin.datasets.synthetic import generate_data

            train, valid = generate_data("criteo", int(100000), set_sizes=(0.7, 0.3))

            train.to_ddf().compute().to_parquet('/tmp/test_merlin_criteo_hugectr/input/criteo/day_0.parquet')
            valid.to_ddf().compute().to_parquet('/tmp/test_merlin_criteo_hugectr/input/criteo/day_1.parquet')
            """
        )
        tb1.execute()
        assert os.path.isfile("/tmp/test_merlin_criteo_hugectr/output/criteo/train/part_0.parquet")
        assert os.path.isfile("/tmp/test_merlin_criteo_hugectr/output/criteo/valid/part_0.parquet")
        assert os.path.isfile("/tmp/test_merlin_criteo_hugectr/output/criteo/workflow/metadata.json")

    with testbook(
        REPO_ROOT
        / "examples"
        / "scaling-criteo"
        / "03-Training-with-HugeCTR.ipynb",
        execute=False,
        timeout=180,
    ) as tb2:
        tb2.inject(
            """
            import os
            os.environ["OUTPUT_DATA_DIR"] = "/tmp/test_merlin_criteo_hugectr/output/criteo/"
            """
        )
        tb2.execute()
        assert os.path.isfile(os.path.join('/tmp/test_merlin_criteo_hugectr/output/criteo/', "criteo_hugectr/1/", "criteo.json"))

    with testbook(
        REPO_ROOT
        / "examples"
        / "scaling-criteo"
        / "04-Triton-Inference-with-HugeCTR.ipynb",
        execute=False,
        timeout=180,
    ) as tb3:
        tb3.inject(
            """
            import os
            os.environ["OUTPUT_DATA_DIR"] = "/tmp/test_merlin_criteo_hugectr/output/criteo/"
            os.environ["INPUT_FOLDER"] = "/tmp/test_merlin_criteo_hugectr/input/criteo/"
            """
        )
        NUM_OF_CELLS = len(tb3.cells)
        tb3.execute_cell(list(range(0, NUM_OF_CELLS - 5)))
        tb3.inject(
            """
            import shutil
            from merlin.systems.triton.utils import run_triton_server, send_triton_request
            outputs = ["OUTPUT0"]
            
            with run_triton_server(
                        "/tmp/test_merlin_criteo_hugectr/output/criteo/model_inference/",
                        backend_config='hugectr,ps=/tmp/test_merlin_criteo_hugectr/output/criteo/model_inference/ps.json'
            ) as client:
                response = send_triton_request(
                    input_schema, batch.fillna(0), outputs, client=client, triton_model="criteo_ens"
                )
            
            response = response["OUTPUT0"]
            """
        )
        tb3.execute_cell(NUM_OF_CELLS - 4)
        response = tb3.ref("response")
        assert len(response) == 3
