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

import argparse
import contextlib
import docker

@contextlib.contextmanager
def managed_container(img):
    client = docker.from_env()
    container = client.containers.run(img, detach=True, ipc_mode="host", runtime="nvidia", tty=True)
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
        output = container.exec_run("bash -c 'pip list | grep " + pkg + "'", stderr=False)
        return output[1].decode("utf-8").split()[1]
    except:
        return "N/A"

def main(args):
  # Images information
  ngc_base = "nvcr.io/nvidia/merlin/"
  containers = ["merlin-training",  "merlin-tensorflow-training",  "merlin-pytorch-training", 
                "merlin-inference", "merlin-tensorflow-inference", "merlin-pytorch-inference"]
  # Information
  info = {}
  # Itaretae images getting information
  for cont in containers:
    info[cont] = {}
    img = ngc_base + cont + ":" + args.version
    with managed_container(img) as container:
       # Get CUDA version
       info[cont]["CUDA"] = get_cuda_version(container)
       # Get rmm version
       info[cont]["rmm"] = get_pythonpkg_version(container, "rmm")
       # Get cuDF version
       info[cont]["cudf"] = get_pythonpkg_version(container, "cudf")
       # Get Merlin Core
       info[cont]["merlin-core"] = get_pythonpkg_version(container, "merlin-core")
       # Get NVTabular
       info[cont]["nvtabular"] = get_pythonpkg_version(container, "nvtabular")
       # Get Transformers4rec
       info[cont]["transformers4rec"] = get_pythonpkg_version(container, "transformers4rec")
       # Get Models
       info[cont]["models"] = get_pythonpkg_version(container, "models")
       # Get HugeCTR
       info[cont]["hugectr"] = get_pythonpkg_version(container, "hugectr")
  print(info)

def parse_args():
    """
    Use the versions script setting Merlin version to explore
    python versions.py -v 22.03
    """
    parser = argparse.ArgumentParser(description=("Merlin Versions Tool"))
    # Config file
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        help="Merlin version (Required)",
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main(parse_args())
