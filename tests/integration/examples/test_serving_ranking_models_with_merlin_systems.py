import os
import shutil

import pytest
from testbook import testbook

from merlin.systems.triton.utils import run_triton_server

from tests.conftest import REPO_ROOT

pytest.importorskip("cudf")
pytest.importorskip("tensorflow")
pytest.importorskip("merlin.models")

TRITON_SERVER_PATH = shutil.which("tritonserver")


@pytest.mark.notebook
@pytest.mark.skipif(not TRITON_SERVER_PATH, reason="triton server not found")
def test_serving_ranking_models(tmp_path):
    with testbook(
        REPO_ROOT / "examples/ranking/TF/Serving-Ranking-Models-with-Merlin-Systems.ipynb",
        execute=False,
        timeout=180,
    ) as tb:
        tb.inject(
            f"""
            import os
            os.environ["DATA_FOLDER"] = "{tmp_path}"
            os.environ["NUM_ROWS"] = "2000"
            """
        )
        NUM_OF_CELLS = len(tb.cells)
        print("num_cells:", NUM_OF_CELLS)
        tb.execute_cell(list(range(0, NUM_OF_CELLS - 12)))
        assert os.path.isdir(f"{tmp_path}/dlrm")
        assert os.path.isdir(f"{tmp_path}/ensemble")
        assert os.listdir(f"{tmp_path}/ensemble")
        assert os.path.isdir(f"{tmp_path}/workflow")

        with run_triton_server(f"{tmp_path}/ensemble", grpc_port=8001):
            tb.execute_cell(list(range(50, NUM_OF_CELLS - 1)))
        
        preds = tb.ref("predictions")
        assert len(preds) == 3

   
