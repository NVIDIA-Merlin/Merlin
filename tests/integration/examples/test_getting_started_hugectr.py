import os

from testbook import testbook
import pandas as pd
import os
import numpy as np

from tests.conftest import REPO_ROOT
import pytest

pytest.importorskip("hugectr")
# flake8: noqa


def test_func():
    INPUT_DATA_DIR = "/tmp/input/getting_started/"
    with testbook(
        REPO_ROOT
        / "examples/getting-started-movielens/01-Download-Convert.ipynb",
        execute=False,
        # timeout=450,
    ) as tb1:
        tb1.cells.pop(7)
        tb1.inject(
            f"""
            import os
            os.environ["INPUT_DATA_DIR"] = "{INPUT_DATA_DIR}"
            """
        )
        os.makedirs(f"{INPUT_DATA_DIR}ml-25m", exist_ok=True)
        pd.DataFrame(
            data={'movieId': list(range(5)), 'genres': ['a', 'a|b', 'b|c', 'c|a|b', 'b'], 'title': ['_'] * 5}
        ).to_csv(f'{INPUT_DATA_DIR}ml-25m/movies.csv', index=False)
        pd.DataFrame(
            data={
                'userId': np.random.randint(0, 10, 100_000),
                'movieId': np.random.randint(0, 5, 100_000),
                'rating': np.random.rand(100_000) * 5,
                'timestamp': ['_'] * 100_000
                }
            ).to_csv(f'{INPUT_DATA_DIR}ml-25m/ratings.csv', index=False)
        tb1.execute()
        assert os.path.isfile("/tmp/input/getting_started/movies_converted.parquet")
        assert os.path.isfile("/tmp/input/getting_started/train.parquet")
        assert os.path.isfile("/tmp/input/getting_started/valid.parquet")

    with testbook(
        REPO_ROOT
        / "examples/getting-started-movielens/02-ETL-with-NVTabular.ipynb",
        execute=False,
        # timeout=450,
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
        REPO_ROOT
        / "examples/getting-started-movielens/03-Training-with-HugeCTR.ipynb",
        execute=False,
        # timeout=450,
    ) as tb3:
        tb3.inject(
            f"""
            import os
            os.environ["INPUT_DATA_DIR"] = "{INPUT_DATA_DIR}"
            os.environ["MODEL_BASE_DIR"] = "{INPUT_DATA_DIR}"
            """
        )
        tb3.execute_cell(list(range(0, 27)))
        os.environ["INPUT_DATA_DIR"] = INPUT_DATA_DIR
        os.environ["MODEL_BASE_DIR"] = INPUT_DATA_DIR
        os.system('python train_hugeCTR.py')
        tb3.execute()
