import os

import pytest
from testbook import testbook
from tests.conftest import REPO_ROOT

pytest.importorskip("tensorflow")
pytest.importorskip("feast")
pytest.importorskip("faiss")
# flake8: noqa


def test_func(tmpdir):
    with testbook(
        REPO_ROOT
        / "examples/Building-and-deploying-multi-stage-RecSys/01-Building-Recommender-Systems-with-Merlin.ipynb",
        execute=False,
        timeout=450,
    ) as tb1:
        NUM_OF_CELLS = len(tb1.cells)
        tb1.inject(
            f"""
            import os
            os.system("mkdir -p {tmpdir / 'examples/'}")
            os.system("mkdir -p {tmpdir / 'data/'}")
            os.system("mkdir -p {tmpdir / 'feast_repo/feature_repo/data/'}")
            os.environ["DATA_FOLDER"] = "{tmpdir / 'data/'}"
            os.environ["BASE_DIR"] = "{tmpdir / 'examples/'}"
            """
        )
        tb1.execute_cell(list(range(0, 25)))
        tb1.inject(
            """
                from pathlib import Path
                from merlin.datasets.ecommerce import transform_aliccp
                from merlin.io.dataset import Dataset
                import glob
                train_min = Dataset(sorted(glob.glob('/raid/data/aliccp/train/*.parquet'))[0:2])
                valid_min = Dataset(sorted(glob.glob('/raid/data/aliccp/test/*.parquet'))[0:2])
                transform_aliccp(
                    (train_min, valid_min), output_path, nvt_workflow=outputs, workflow_name="workflow"
                )
            """
        )
        tb1.execute_cell(list(range(28, NUM_OF_CELLS)))
        assert os.path.isdir(f"{tmpdir / 'examples/dlrm'}")
        assert os.path.isdir(f"{tmpdir / 'examples/feast_repo/feature_repo'}")
        assert os.path.isdir(f"{tmpdir / 'examples/query_tower'}")
        assert os.path.isfile(f"{tmpdir / 'examples/item_embeddings.parquet'}")
        assert os.path.isfile(
            f"{tmpdir / 'examples/feast_repo/feature_repo/user_features.py'}"
        )
        assert os.path.isfile(
            f"{tmpdir / 'examples/feast_repo/feature_repo/item_features.py'}"
        )

    with testbook(
        REPO_ROOT
        / "examples/Building-and-deploying-multi-stage-RecSys/02-Deploying-multi-stage-RecSys-with-Merlin-Systems.ipynb",
        execute=False,
        timeout=2400,
    ) as tb2:
        tb2.inject(
            f"""
            import os
            os.environ["DATA_FOLDER"] = "{tmpdir / "data"}"
            os.environ["BASE_DIR"] = "{tmpdir / "examples"}"
            """
        )
        NUM_OF_CELLS = len(tb2.cells)
        tb2.execute_cell(list(range(0, NUM_OF_CELLS - 3)))
        top_k = tb2.ref("top_k")
        outputs = tb2.ref("outputs")
        assert outputs[0] == "ordered_ids"
        tb2.inject(
            f"""
            import shutil
            from merlin.core.dispatch import make_df
            from merlin.dataloader.tf_utils import configure_tensorflow
            from merlin.systems.triton.utils import run_ensemble_on_tritonserver
            import pandas as pd
            configure_tensorflow()
            user_features = pd.read_parquet("{tmpdir / 'examples/feast_repo/feature_repo/data/user_features.parquet'}")
            request = user_features[["user_id_raw"]].sample(1)
            request["user_id_raw"] = request["user_id_raw"].astype(np.int32)
            response = run_ensemble_on_tritonserver(
                "{tmpdir / "examples"}/poc_ensemble", ensemble.graph.input_schema, request, outputs,  "executor_model"
            )
            ordered_ids = [x.tolist() for x in response["ordered_ids"]]
            shutil.rmtree("/tmp/examples/", ignore_errors=True)
            """
        )
        ordered_ids = tb2.ref("ordered_ids")
        assert len(ordered_ids[0]) == top_k
