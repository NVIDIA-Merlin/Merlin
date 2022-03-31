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

# pylint: skip-file

import argparse
import contextlib

from github import Github
from tomark import Tomark

import docker


@contextlib.contextmanager
def managed_container(img):
    client = docker.from_env()
    container = client.containers.run(
        img, detach=True, ipc_mode="host", runtime="nvidia", tty=True
    )
    try:
        yield container
    finally:
        container.stop()
        container.remove()


def get_cuda_version(container):
    output = container.exec_run("nvcc --version")
    return output[1].decode("utf-8").split()[19]


def get_pythonpkg_version(container, pkg):
    try:
        output = container.exec_run(
            "bash -c 'pip list | grep " + pkg + "'", stderr=False
        )
        return output[1].decode("utf-8").split()[1]
    except:  # noqa
        return "N/A"


def create_pr(repo, branch, filename, content, token, version):
    g = Github(token)
    r = g.get_repo(repo)
    r.create_git_ref(ref="refs/heads/" + branch, sha=r.get_branch("main").commit.sha)
    f = r.get_contents(filename, ref=branch)
    r.update_file(f.path, "release " + version, content, branch=branch, sha=f.sha)
    pr = r.create_pull(  # noqa
        title="Merlin Release " + version, body="", head=branch, base="main"
    )


def main(args):
    # Images information
    ngc_base = "nvcr.io/nvidia/merlin/"
    containers = [
        "merlin-training",
        "merlin-tensorflow-training",
        "merlin-pytorch-training",
        "merlin-inference",
        "merlin-tensorflow-inference",
        "merlin-pytorch-inference",
    ]
    # Information
    table_info = []
    # Itaretae images getting information
    for cont in containers:
        cont_info = {}
        img = ngc_base + cont + ":" + args.version
        cont_info["Container"] = img
        with managed_container(img) as container:
            # Get CUDA version
            cont_info["CUDA"] = get_cuda_version(container)
            # Get rmm version
            cont_info["RMM"] = get_pythonpkg_version(container, "rmm")
            # Get cuDF version
            cont_info["cuDF"] = get_pythonpkg_version(container, "cudf")
            # Get Merlin Core
            cont_info["Merlin-Core"] = get_pythonpkg_version(container, "merlin-core")
            # Get Merlin Systems
            cont_info["Merlin-Systems"] = get_pythonpkg_version(
                container, "merlin-systems"
            )
            # Get NVTabular
            cont_info["NVTabular"] = get_pythonpkg_version(container, "nvtabular")
            # Get Models
            cont_info["Merlin-Models"] = get_pythonpkg_version(container, "models")
            # Get Transformers4rec
            cont_info["Transformers4Rec"] = get_pythonpkg_version(
                container, "transformers4rec"
            )
            # Get HugeCTR
            cont_info["HugeCTR"] = get_pythonpkg_version(container, "hugectr")
            # Update table
            table_info.append(cont_info)
    # Generate markdown file and create PR
    filename = "Release.md"
    markdown = Tomark.table(table_info)
    create_pr(
        "Nvidia-Merlin/Merlin",
        "release-info-" + args.version,
        filename,
        markdown,
        args.token,
        args.version,
    )


def parse_args():
    """
    Use the versions script setting Merlin version to explore
    python versions.py -v 22.03
    """
    parser = argparse.ArgumentParser(description=("Merlin Versions Tool"))
    # Containers version
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        help="Merlin version (Required)",
    )

    # Github token
    parser.add_argument(
        "-t",
        "--token",
        type=str,
        help="GitHub token (Required)",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(parse_args())
