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

from __future__ import absolute_import

import contextlib
import os

from pathlib import Path

import pytest
import platform
import warnings
from pathlib import Path

import distributed
import psutil
import pytest
from asvdb import BenchmarkInfo, utils

from merlin.models.utils import ci_utils

REPO_ROOT = Path(__file__).parent.parent


try:
    import cudf

    try:
        import cudf.testing._utils

        assert_eq = cudf.testing._utils.assert_eq
    except ImportError:
        import cudf.tests.utils

        assert_eq = cudf.tests.utils.assert_eq
except ImportError:
    cudf = None

    def assert_eq(a, b, *args, **kwargs):
        if isinstance(a, pd.DataFrame):
            return pd.testing.assert_frame_equal(a, b, *args, **kwargs)
        elif isinstance(a, pd.Series):
            return pd.testing.assert_series_equal(a, b, *args, **kwargs)
        else:
            return np.testing.assert_allclose(a, b)

def pytest_sessionfinish(session, exitstatus):
    # if all the tests are skipped, lets not fail the entire CI run
    if exitstatus == 5:
        session.exitstatus = 0

@contextlib.contextmanager
def get_cuda_cluster():
    from dask_cuda import LocalCUDACluster

    CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    n_workers = min(2, len(CUDA_VISIBLE_DEVICES.split(",")))
    cluster = LocalCUDACluster(n_workers=n_workers)
    yield cluster
    cluster.close()

def get_benchmark_info():
    uname = platform.uname()
    (commitHash, commitTime) = utils.getCommitInfo()

    return BenchmarkInfo(
        machineName=uname.machine,
        cudaVer="na",
        osType="%s %s" % (uname.system, uname.release),
        pythonVer=platform.python_version(),
        commitHash=commitHash,
        commitTime=commitTime,
        gpuType="na",
        cpuType=uname.processor,
        arch=uname.machine,
        ram="%d" % psutil.virtual_memory().total,
    )
