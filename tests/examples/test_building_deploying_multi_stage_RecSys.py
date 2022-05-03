import os

from testbook import testbook

from tests.conftest import REPO_ROOT
from merlin.systems.triton.utils import run_ensemble_on_tritonserver


@testbook(REPO_ROOT / "examples/Building-and-deploying-multi-stage-RecSys/01-Building-Recommender-Systems-with-Merlin.ipynb", execute=False)
#@testbook("/Merlin/examples/Building-and-deploying-multi-stage-RecSys/02-Deploying-multi-stage-RecSys-with-Merlin-Systems.ipynb", execute=False)

def test_func(tb1):
    tb1.inject(
        """
        import os
        os.environ["DATA_FOLDER"] = "/tmp/data/"
        os.system('mkdir -p /tmp/examples')
        os.environ["BASE_DIR"] = "/tmp/examples/"
        """
    )
    tb1.execute()
    assert os.path.isdir("dlrm")
    assert os.path.isdir("/tmp/examples/feature_repo")
    assert os.path.isdir("/tmp/examples/query_tower")
    assert os.path.isfile('/tmp/examples/item_embeddings.parquet')
#     tb2.inject(
#         """
#         import os
#         os.environ["DATA_FOLDER"] = "/tmp/data/"
#         os.environ["BASE_DIR"] = "/tmp/examples/"
#         """
#     )
    
#     NUM_OF_CELLS = len(tb2.cells)
#     tb2.execute_cell(list(range(0, NUM_OF_CELLS - 5)))
#     outputs = tb2.ref("outputs")
#     assert outputs[0] == "ordered_ids"

#     top_k = tb2.ref("top_k")
#     assert len(response[outputs[0]]) == top_k
#     tb2.execute_cell(NUM_OF_CELLS - 2)