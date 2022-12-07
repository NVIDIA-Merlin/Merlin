#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import itertools
import json
import os
import subprocess
import sys
from distutils.spawn import find_executable
from os.path import dirname, realpath

import pytest

pytest.importorskip("cudf")
import cudf  # noqa: E402

from tests.conftest import get_cuda_cluster  # noqa: E402

TEST_PATH = dirname(dirname(dirname(realpath(__file__))))

def test_optimize_criteo(tmpdir):
    input_path = str(tmpdir.mkdir("input"))
    _get_random_criteo_data(1000).to_csv(os.path.join(input_path, "day_0"), sep="\t", header=False)
    os.environ["INPUT_DATA_DIR"] = input_path
    os.environ["OUTPUT_DATA_DIR"] = str(tmpdir.mkdir("output"))
    with get_cuda_cluster() as cuda_cluster:
        scheduler_port = cuda_cluster.scheduler_address

        def _nb_modify(line):
            # Use cuda_cluster "fixture" port rather than allowing notebook
            # to deploy a LocalCUDACluster within the subprocess
            line = line.replace("download_criteo = True", "download_criteo = False")
            line = line.replace("cluster = None", f"cluster = '{scheduler_port}'")
            return line

        notebook_path = os.path.join(
            dirname(TEST_PATH),
            "examples/scaling-criteo/",
            "01-Download-Convert.ipynb",
        )
        _run_notebook(tmpdir, notebook_path, _nb_modify)

def _run_notebook(tmpdir, notebook_path, transform=None):
    # read in the notebook as JSON, and extract a python script from it
    notebook = json.load(open(notebook_path, encoding="utf-8"))
    source_cells = [cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code"]
    lines = [
        transform(line.rstrip()) if transform else line
        for line in itertools.chain(*source_cells)
        if not (line.startswith("%") or line.startswith("!"))
    ]

    # save the script to a file, and run with the current python executable
    # we're doing this in a subprocess to avoid some issues using 'exec'
    # that were causing a segfault with globals of the exec'ed function going
    # out of scope
    script_path = os.path.join(tmpdir, "notebook.py")
    with open(script_path, "w") as script:
        script.write("\n".join(lines))
    subprocess.check_output([sys.executable, script_path])


def _get_random_criteo_data(rows):
    dtypes = {col: float for col in [f"I{x}" for x in range(1, 14)]}
    dtypes.update({col: int for col in [f"C{x}" for x in range(1, 27)]})
    dtypes["label"] = bool
    ret = cudf.datasets.randomdata(rows, dtypes=dtypes)
    # binarize the labels
    ret.label = ret.label.astype(int)
    return ret