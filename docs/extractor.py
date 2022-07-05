#!/usr/bin/env python3

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

# Use a for-loop, to access
# multiple containers. The command-line argument
# is a release value, like "22.02".
# Between the release value and the container name,
# there is enough information to group the container
# information...
#
# {
#    "merlin-training": {
#       "22.02": {
#          "cuda": "11.6"
#       },
#       "22.01": {
#          "cuda": "11.5"
#       }
#    }
# }
#
# After all the data is gathered, it should be possible
# to construct a multi-release table.
# "merlin-training"
# |       | 22.02 | 22.01 |
# | ----- | ----- | ----- |
# | CUDA  |  11.6 |  11.5 |

import argparse
import json
import logging
import os
import subprocess
import sys
from collections import defaultdict
from datetime import date
from datetime import datetime as dt
from pathlib import Path

import yaml
from github import Github
from github.GithubException import GithubException
from github.GitRef import GitRef
from semver import VersionInfo

level = logging.DEBUG if os.environ.get("DEBUG") else logging.INFO
logging.basicConfig(level=level)
logger = logging.getLogger("extractor")


def get_yymm() -> str:
    return date.today().strftime("%y.%m")


# pylint: disable=too-many-locals
def open_pr(repo: str, path: str, release: str):
    token = os.environ.get("GH_TOKEN")
    if token is None:
        logger.info("Env var GH_TOKEN is not found. Cannot open PR.")
        return

    msg = "Updates from containers"
    pr_branch = "docs-smx-" + release.replace(".", "")

    content: str
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    g = Github(token)
    r = g.get_repo(repo)
    remote_ref: GitRef
    counter = 1
    while True:
        remote_branch = f"refs/heads/{pr_branch}-{counter}"
        try:
            remote_ref = r.create_git_ref(
                ref=remote_branch,
                sha=r.get_branch("main").commit.sha,
            )
        except GithubException:
            logger.info(
                "PR branch '%s' already exists. Incrementing the counter.",
                remote_branch,
            )
            counter += 1
            if counter > 25:
                logger.info("Failed to create a unique branch name. Giving up.")
                raise
        else:
            logger.info("Remote ref created: '%s'", remote_ref.ref)
            break

    f = r.get_contents(path, ref=remote_ref.ref)
    result = r.update_file(f.path, msg, content, branch=remote_ref.ref, sha=f.sha)
    diff = r.compare(r.get_branch("main").commit.sha, result["commit"].sha)
    if len(diff.files) == 0:
        logger.info("No changes to commit.")
        remote_ref.delete()
        return

    try:
        pr = r.create_pull(  # noqa
            title="Support matrix updates for " + release,
            body=msg,
            head=remote_ref.ref,
            base="main",
        )
        logger.info("Opened PR: '%s'", pr.html_url)
    except Exception as e:  # pylint: disable=broad-except
        logger.info("Failed to open PR: %s", e)


class SupportMatrixExtractor:

    contdata = {}
    data: defaultdict(dict)
    ERROR = "Not applicable"
    container_name: str
    release: str
    standard_snippets = ["dgx_system", "nvidia_driver", "gpu_model"]
    force = False

    def __init__(self, name: str, release: str, datafile: str, force: bool = False):
        self.container_name = name
        self.release = release
        self.contdata = {}
        self.data = {}
        self.data = defaultdict(dict)
        self.data[self.container_name][self.release] = self.contdata
        self.datafile = datafile
        self.force = force

    def get_from_envfile(self, path: str, lookup: str, key=None):
        if key is None:
            key = lookup
        self.contdata[key] = self.ERROR
        p = subprocess.run(  # nosec B602
            f"bash -c 'source {path}; echo ${{{lookup}}}'",
            shell=True,
            capture_output=True,
            check=False,
        )
        result = p.stdout.decode("utf-8")
        if p.returncode != 1 and not result.isspace():
            self.contdata[key] = result.replace('"', "").strip()
        else:
            logger.info("Failed to get env var '%s' from file '%s'", lookup, path)

    def get_from_env(self, lookup: str, key=None):
        if key is None:
            key = lookup
        self.contdata[key] = self.ERROR
        p = subprocess.run(  # nosec B602
            f"bash -c 'echo ${{{lookup}}}'",
            shell=True,
            capture_output=True,
            check=False,
        )
        result = p.stdout.decode("utf-8")
        if p.returncode != 1 and not result.isspace():
            self.contdata[key] = result.replace('"', "").strip()
            if lookup == "SMX_COMPRESSED_SIZE":
                # pylint: disable=C0209
                self.contdata[key] = "{} GB".format(
                    round(int(result) / 1024**3, 2)
                )  # noqa
        else:
            logger.info("Failed to get env var: '%s'", lookup)

    def get_from_pip(self, lookup: str, key=None):
        """Retrieves the version of a Python package from Pip. This function avoids importing
        the package which might not work on systems without a GPU.

        Returns `None` if the package isn't installed.
        """
        if key is None:
            key = lookup
        self.contdata[key] = self.ERROR
        p = subprocess.run(  # nosec B602
            f"python -m pip show '{lookup}'",
            shell=True,
            capture_output=True,
            check=False,
        )
        result = p.stdout.decode("utf-8")
        if p.returncode != 0:
            logger.info("Failed to get package version from pip: %s", lookup)
            return
        versions = [
            line.split()[-1]
            for line in result.split("\n")
            if line.startswith("Version:")
        ]
        if len(versions) == 1:
            self.contdata[key] = versions[0].strip()
        else:
            logger.info("Failed to extract version from pip output: %s", result)

    def get_from_python(self, lookup: str, key=None):
        """Retrieves the version of a Python package by importing the package
        and printing the ``__version__`` value.

        Returns `None` if the package version cannot be printed.
        """
        if key is None:
            key = lookup
        self.contdata[key] = self.ERROR
        p = subprocess.run(  # nosec B602
            f"python -c 'import {lookup} as x; print(x.__version__);'",
            shell=True,
            capture_output=True,
            check=False,
        )
        result = p.stdout.decode("utf-8")
        if p.returncode != 0:
            logger.info(
                "Failed to get '%s' package version from python: %s",
                lookup,
                result,
            )
            return
        for line in result.split("\n"):
            if VersionInfo.isvalid(line):
                result = line
                break
        self.contdata[key] = result.strip()

    def get_from_cmd(self, cmd: str, key: str):
        self.contdata[key] = self.ERROR
        p = subprocess.run(  # nosec B602
            f"bash -c '{cmd}'",
            shell=True,
            capture_output=True,
            check=False,
        )
        result = p.stdout.decode("utf-8")
        if p.returncode == 0:
            self.contdata[key] = result.strip()
            # Let the hacks begin...
            if key == "sm":
                smlist = result.split()
                self.contdata[key] = ", ".join(smlist)
            if key == "size":
                # pylint: disable=C0209
                self.contdata[key] = "{} GB".format(
                    round(int(result) / 1024**3, 2)
                )  # noqa
        else:
            logger.info("Command '%s' failed: %s", cmd, result)

    def insert_snippet(self, key: str, snip: str):
        self.contdata[key] = snip

    def to_json(self):
        return json.dumps(self.data, sort_keys=True)

    def from_json(self):
        if not os.path.exists(self.datafile):
            return

        with open(self.datafile, "r", encoding="utf-8") as f:
            self.data = json.load(f)

            if self.container_name not in self.data:
                self.data[self.container_name] = {}
            if self.release not in self.data[self.container_name] or self.force is True:
                self.data[self.container_name][self.release] = {}

        self.contdata = self.data[self.container_name][self.release]

    def to_json_file(self):
        logger.debug("Storing data to file: '%s'", self.datafile)
        with open(self.datafile, "w", encoding="utf-8") as f:
            json.dump(self.data, f, sort_keys=True, indent=2)
        logger.debug("...done.")

    def already_present(self) -> bool:
        if not os.path.exists(self.datafile):
            return False
        if self.container_name not in self.data.keys():
            return False
        if self.release not in self.data[self.container_name]:
            return False
        if len(self.data[self.container_name][self.release]) < 1:
            return False
        return True


# pylint: disable=too-many-locals, too-many-statements, too-many-branches
def main(args):
    # Images information
    ngc_base = "nvcr.io/nvidia/merlin/"

    scriptdir = Path(__file__).parent

    jsonfile = scriptdir / "data.json"
    snippetsfile = scriptdir / "snippets.yaml"
    version = args.version
    force = False

    if args.file:
        jsonfile = os.path.abspath(args.file)
    if args.snippets:
        snippetsfile = os.path.abspath(args.snippets)
    if args.force is True:
        force = True
    if not version:
        version = get_yymm()

    if args.pr:
        repo = os.environ.get("REPO", r"NVIDIA-Merlin/Merlin")
        open_pr(repo, str(jsonfile), version)
        sys.exit(0)

    if not args.container:
        logger.error("container is a required argument")
        sys.exit(1)

    sniptext = {}
    with open(snippetsfile, "r", encoding="utf-8") as f:
        sniptext = yaml.safe_load(f)
        for k in SupportMatrixExtractor.standard_snippets:
            assert sniptext[k]

    img = ngc_base + args.container + ":" + version

    logger.info("Extracting information from: %s", img)
    xtr = SupportMatrixExtractor(ngc_base + args.container, version, jsonfile, force)
    xtr.from_json()

    if xtr.already_present() and force is False:
        logger.info("...skipping because container is already in data.")
        return

    for k in xtr.standard_snippets:
        xtr.insert_snippet(k, sniptext[k])

    xtr.insert_snippet("release", version)

    xtr.get_from_env("SMX_COMPRESSED_SIZE", "compressedSize")
    xtr.get_from_cmd("du -sb / 2>/dev/null | cut -f1", "size")
    xtr.get_from_envfile("/etc/os-release", "PRETTY_NAME", "os")
    xtr.get_from_env("CUDA_VERSION", "cuda")
    xtr.get_from_pip("rmm")
    xtr.get_from_pip("cudf")
    xtr.get_from_env("CUDNN_VERSION", "cudnn")
    xtr.get_from_pip("nvtabular")
    xtr.get_from_pip("transformers4rec")
    xtr.get_from_pip("merlin.core")
    xtr.get_from_pip("merlin.systems")
    xtr.get_from_pip("merlin.models")
    xtr.get_from_python("hugectr2onnx")
    xtr.get_from_python("hugectr")
    xtr.get_from_python("sparse_operation_kit")
    xtr.get_from_pip("tensorflow", "tf")
    xtr.get_from_pip("torch", "pytorch")
    xtr.get_from_env("CUBLAS_VERSION", "cublas")
    xtr.get_from_env("CUFFT_VERSION", "cufft")
    xtr.get_from_env("CURAND_VERSION", "curand")
    xtr.get_from_env("CUSOLVER_VERSION", "cusolver")
    xtr.get_from_env("CUSPARSE_VERSION", "cusparse")
    xtr.get_from_env("CUTENSOR_VERSION", "cutensor")
    xtr.get_from_env("NVIDIA_TENSORFLOW_VERSION", "nvidia_tensorflow")
    xtr.get_from_env("NVIDIA_PYTORCH_VERSION", "nvidia_pytorch")
    xtr.get_from_env("OPENMPI_VERSION", "openmpi")
    xtr.get_from_env("TRT_VERSION", "tensorrt")
    xtr.get_from_env("TRTOSS_VERSION", "base_container")
    # xtr.get_from_cmd("cuobjdump /usr/local/hugectr/lib/libhuge_ctr_shared.so
    # | grep arch | sed -e \'s/.*sm_//\' | sed -e \'H;${x;s/\\n/, /g;s/^, //;p};d\'", "sm")
    # flake8: noqa
    xtr.get_from_cmd(
        "if [ ! -f /usr/local/hugectr/lib/libhuge_ctr_shared.so ]; then exit 1; fi; cuobjdump /usr/local/hugectr/lib/libhuge_ctr_shared.so | grep arch | sed -e 's/.*sm_//'",
        "sm",
    )
    xtr.get_from_cmd("cat /opt/tritonserver/TRITON_VERSION", "triton")
    xtr.get_from_cmd(
        'python -c "import sys;print(sys.version_info[0]);"', "python_major"
    )

    # Some hacks for the base container image
    if args.container == "merlin-training":
        xtr.insert_snippet("base_container", "Not applicable")
    elif args.container == "merlin-tensorflow-training":
        tf2_img = xtr.contdata["nvidia_tensorflow"]
        py_maj = xtr.contdata["python_major"]
        xtr.insert_snippet(
            "base_container",
            f"nvcr.io/nvidia/tensorflow:{tf2_img}-py{py_maj}",
        )
    elif args.container == "merlin-pytorch-training":
        pt_img = xtr.contdata["nvidia_pytorch"]
        py_maj = xtr.contdata["python_major"]
        xtr.insert_snippet(
            "base_container",
            f"nvcr.io/nvidia/pytorch:{pt_img}-py{py_maj}",
        )
    else:
        trtoss = xtr.contdata["base_container"]
        xtr.insert_snippet("base_container", f"Triton version {trtoss}")

    xtr.insert_snippet("timestamp_utc", dt.utcnow().isoformat())
    xtr.to_json_file()

    logger.info(xtr.contdata)

    logger.info(xtr.data)


def parse_args():
    """
    Use the versions script setting Merlin version to explore
    python extractor.py -v 22.03
    """
    parser = argparse.ArgumentParser(description=("Container Extraction Tool"))
    # Containers version
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        help="Version in YY.MM format",
    )

    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="JSON data file",
    )

    parser.add_argument(
        "-s",
        "--snippets",
        type=str,
        help="YAML snippets file",
    )

    parser.add_argument(
        "-c",
        "--container",
        type=str,
        help="Single container name",
    )

    parser.add_argument(
        "--force",
        type=bool,
        default=False,
        help="When True, specifies to get data for a container that is already in data.json",
    )

    parser.add_argument(
        "--pr",
        type=bool,
        default=False,
        help="When True, open a PR for the data.json file",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(parse_args())
