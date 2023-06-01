import os
import shutil

import numpy as np
import pandas as pd
import pytest
from testbook import testbook

from tests.conftest import REPO_ROOT

pytest.importorskip("torch")
# flake8: noqa


@pytest.mark.multigpu
def test_func():
    INPUT_DATA_DIR = "/tmp/input/getting_started/"
    with testbook(
        REPO_ROOT / "examples/getting-started-movielens/01-Download-Convert.ipynb",
        execute=False,
    ) as tb1:
        tb1.cells.pop(7)
        tb1.inject(
            f"""
            import os
            os.environ["INPUT_DATA_DIR"] = "{INPUT_DATA_DIR}"
            """
        )
        if os.path.exists(INPUT_DATA_DIR) and os.path.isdir(INPUT_DATA_DIR):
            shutil.rmtree(INPUT_DATA_DIR)
        os.makedirs(f"{INPUT_DATA_DIR}ml-25m", exist_ok=True)
        pd.DataFrame(
            data={
                "movieId": list(range(56632)),
                "genres": ["abcdefghijkl"[i] for i in np.random.randint(0, 12, 56632)],
                "title": ["_"] * 56632,
            }
        ).to_csv(f"{INPUT_DATA_DIR}ml-25m/movies.csv", index=False)
        pd.DataFrame(
            data={
                "userId": np.random.randint(0, 162542, 1_000_000),
                "movieId": np.random.randint(0, 56632, 1_000_000),
                "rating": np.random.rand(1_000_000) * 5,
                "timestamp": ["_"] * 1_000_000,
            }
        ).to_csv(f"{INPUT_DATA_DIR}ml-25m/ratings.csv", index=False)
        tb1.execute()
        assert os.path.isfile("/tmp/input/getting_started/movies_converted.parquet")
        assert os.path.isfile("/tmp/input/getting_started/train.parquet")
        assert os.path.isfile("/tmp/input/getting_started/valid.parquet")

    with testbook(
        REPO_ROOT / "examples/getting-started-movielens/02-ETL-with-NVTabular.ipynb",
        execute=False,
    ) as tb2:
        tb2.inject(
            f"""
            import os
            os.environ["INPUT_DATA_DIR"] = "{INPUT_DATA_DIR}"
            """
        )
        tb2.execute()
        assert os.path.isdir("/tmp/input/getting_started/train")
        assert os.path.isdir("/tmp/input/getting_started/valid")
        assert os.path.isdir("/tmp/input/getting_started/workflow")

    with testbook(
        REPO_ROOT / "examples/getting-started-movielens/03-Training-with-PyTorch.ipynb",
        execute=False,
    ) as tb3:
        tb3.inject(
            f"""
            import os
            os.environ["INPUT_DATA_DIR"] = "{INPUT_DATA_DIR}"
            """
        )
        tb3.execute()
