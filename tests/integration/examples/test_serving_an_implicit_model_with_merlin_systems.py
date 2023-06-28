import shutil

import pytest
from testbook import testbook

from merlin.systems.triton.utils import run_triton_server
from merlin.core.compat import cudf
from tests.conftest import REPO_ROOT

pytest.importorskip("implicit")
pytest.importorskip("merlin.models")


if cudf:

    _TRAIN_ON_GPU = [True, False]
else:
    _TRAIN_ON_GPU = [False]

TRITON_SERVER_PATH = shutil.which("tritonserver")


@pytest.mark.notebook
@pytest.mark.skipif(not TRITON_SERVER_PATH, reason="triton server not found")
@pytest.mark.parametrize("gpu", _TRAIN_ON_GPU)
def test_example_serving_implicit(gpu, tmpdir):
    with testbook(
        REPO_ROOT / "examples/Serving-An-Implicit-Model-With-Merlin-Systems.ipynb",
        execute=False,
        timeout=180,
    ) as tb:
        tb.inject(
            f"""
            import os
            os.environ["OUTPUT_DATA_DIR"] = "{tmpdir}/ensemble"
            os.environ["USE_GPU"] = "{int(gpu)}"
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
            """,
            pop=True,
        )

        tb.execute_cell(list(range(0, 18)))

        with run_triton_server(f"{tmpdir}/ensemble", grpc_port=8001):
            tb.execute_cell(list(range(18, len(tb.cells) - 2)))
            pft = tb.ref("predictions_from_triton")
            lp = tb.ref("local_predictions")
            assert pft.shape == lp.shape
