import os

from testbook import testbook

from tests.conftest import REPO_ROOT

import pytest

pytest.importorskip("tensorflow")
pytest.importorskip("feast")
pytest.importorskip("faiss")

# flake8: noqa

def test_func():
    with testbook(
        REPO_ROOT
        / "examples"
        / "Building-and-deploying-multi-stage-RecSys"
        / "01-Building-Recommender-Systems-with-Merlin.ipynb",
        execute=False,
    ) as tb1:
        tb1.inject(
            """
            import os
            os.environ["DATA_FOLDER"] = "/tmp/data/"
            os.environ["NUM_ROWS"] = "100000"
            os.system("mkdir -p /tmp/examples")
            os.environ["BASE_DIR"] = "/tmp/examples/"
            """
        )
        tb1.execute()
        assert os.path.isdir("/tmp/examples/dlrm")
        assert os.path.isdir("/tmp/examples/feature_repo")
        assert os.path.isdir("/tmp/examples/query_tower")
        assert os.path.isfile("/tmp/examples/item_embeddings.parquet")
        assert os.path.isfile("/tmp/examples/feature_repo/user_features.py")
        assert os.path.isfile("/tmp/examples/feature_repo/item_features.py")

    with testbook(
        REPO_ROOT
        / "examples"
        / "Building-and-deploying-multi-stage-RecSys"
        / "02-Deploying-multi-stage-RecSys-with-Merlin-Systems.ipynb",
        execute=False,
        timeout=180,
    ) as tb2:
        tb2.inject(
            """
            import os
            os.environ["DATA_FOLDER"] = "/tmp/data/"
            os.environ["BASE_DIR"] = "/tmp/examples/"
            os.environ["topk_retrieval"] = "20"
            """
        )
        NUM_OF_CELLS = len(tb2.cells)
        tb2.execute_cell(list(range(0, NUM_OF_CELLS - 3)))
        top_k = tb2.ref("top_k")
        outputs = tb2.ref("outputs")
        assert outputs[0] == "ordered_ids"
        tb2.inject(
            """
            import shutil
            from merlin.core.dispatch import get_lib
            from merlin.dataloader.tf_utils import configure_tensorflow
            configure_tensorflow()
            df_lib = get_lib()
            batch = df_lib.read_parquet(
                os.path.join("/tmp/data/processed_nvt/", "train", "part_0.parquet"),
                num_rows=1,
                columns=["user_id_raw"],
            )
            from merlin.systems.triton.utils import run_ensemble_on_tritonserver
            response = run_ensemble_on_tritonserver(
                "/tmp/examples/poc_ensemble", ensemble.graph.input_schema, batch, outputs,  "ensemble_model"
            )
            response = [x.tolist()[0] for x in response["ordered_ids"]]
            shutil.rmtree("/tmp/examples/", ignore_errors=True)
            """
        )
        tb2.execute_cell(NUM_OF_CELLS - 2)
        response = tb2.ref("response")
        assert len(response) == top_k
