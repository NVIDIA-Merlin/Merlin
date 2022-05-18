import os

from testbook import testbook

from tests.conftest import REPO_ROOT


@testbook(
    REPO_ROOT
    / "examples/Building-and-deploying-multi-stage-RecSys/01-Building-Recommender-Systems-with-Merlin.ipynb",
    execute=False
)
def test_nb1(tb):
    tb.inject(
        """
        import os
        os.environ["DATA_FOLDER"] = "/tmp/data/"
        os.environ["NUM_ROWS"] = "40_000"
        os.system("mkdir -p /tmp/examples")
        os.environ["BASE_DIR"] = "/tmp/examples/"
        """
    )
    tb.execute()
    assert os.path.isdir("/tmp/examples/dlrm")
    assert os.path.isdir("/tmp/examples/feature_repo")
    assert os.path.isdir("/tmp/examples/query_tower")
    assert os.path.isfile("/tmp/examples/item_embeddings.parquet")
    assert os.path.isfile("/tmp/examples/feature_repo/user_features.py")
    assert os.path.isfile("/tmp/examples/feature_repo/item_features.py")

@testbook(
    REPO_ROOT
    / "examples/Building-and-deploying-multi-stage-RecSys/02-Deploying-multi-stage-RecSys-with-Merlin-Systems.ipynb",
    execute=False
)
def test_nb2(tb):
    tb.inject(
        """
        import os
        os.environ["DATA_FOLDER"] = "/tmp/data/"
        os.environ["BASE_DIR"] = "/tmp/examples/"
        """
    )
    NUM_OF_CELLS = len(tb.cells)
    tb.execute_cell(list(range(0, NUM_OF_CELLS - 3)))
    top_k = tb.ref("top_k")
    outputs = tb.ref("outputs")
    request = tb.ref("request")
    assert outputs[0] == "ordered_ids"
    tb.inject(
        """
        import shutil
        from merlin.models.loader.tf_utils import configure_tensorflow
        configure_tensorflow()
        from merlin.systems.triton.utils import run_ensemble_on_tritonserver
        response = run_ensemble_on_tritonserver(
            "/tmp/examples/poc_ensemble", outputs, request, "ensemble_model"
        )
        response = [x.tolist()[0] for x in response["ordered_ids"]]
        shutil.rmtree("/tmp/examples/", ignore_errors=True)
        """
    )
    tb.execute_cell(NUM_OF_CELLS - 2)
    response = tb.ref("response")
    assert len(response) == top_k
