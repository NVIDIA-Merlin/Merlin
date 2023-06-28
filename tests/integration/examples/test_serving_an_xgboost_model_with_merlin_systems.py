import shutil

import pytest
from testbook import testbook

from merlin.systems.triton.utils import run_triton_server
from tests.conftest import REPO_ROOT

pytest.importorskip("tensorflow")
pytest.importorskip("merlin.models")
pytest.importorskip("xgboost")

TRITON_SERVER_PATH = shutil.which("tritonserver")


@pytest.mark.skipif(not TRITON_SERVER_PATH, reason="triton server not found")
@pytest.mark.notebook
def test_example_serving_xgboost(tmpdir):
    with testbook(
        REPO_ROOT / "examples/Serving-An-XGboost-Model-With-Merlin-Systems.ipynb",
        execute=False,
        timeout=180,
    ) as tb:
        tb.inject(
            f"""
            import os
            os.environ["OUTPUT_DATA_DIR"] = "{tmpdir}/ensemble"
            from unittest.mock import patch
            from merlin.datasets.synthetic import generate_data
            mock_train, mock_valid = generate_data(
                input="movielens-100k",
                num_rows=1000,
                set_sizes=(0.8, 0.2)
            )
            p1 = patch(
                "merlin.datasets.entertainment.get_movielens",
                return_value=[mock_train, mock_valid]
            )
            p1.start()
            """
        )
        NUM_OF_CELLS = len(tb.cells)

        tb.execute_cell(list(range(0, 14)))

        with run_triton_server(f"{tmpdir}/ensemble", grpc_port=8001):
            tb.execute_cell(list(range(14, NUM_OF_CELLS - 1)))
            pft = tb.ref("predictions_from_triton")
            lp = tb.ref("local_predictions")
            assert pft.shape == lp.shape
