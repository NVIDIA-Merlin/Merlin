# Installation

```{contents}
---
depth: 2
local: true
backlinks: none
---
```

Many users have different preferences on how to run their machine learning pipeline. We want to provide a guide for the most common platforms. Merlin can be easily installed via `pip`, but resolving the dependencies can be sometimes challenging (GPU Driver, TensorFlow/PyTorch, RAPIDs, Triton Inference Server).

## Docker (Preferred Option)

Our recommendation is to use our hosted docker containers, which are accessible on NVIDIA GPU Cloud (NCG) catalog at [https://catalog.ngc.nvidia.com/containers](https://catalog.ngc.nvidia.com/containers).

| Container Name | Key Merlin Components |
| ------------- | ------------- | 
| [merlin-hugectr](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-hugectr) | Merlin Libraries - in particular HugeCTR, Triton |
| [merlin-tensorflow](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-tensorflow) | Merlin Libraries - in particular Merlin Models and SOK, Triton, TensorFlow |
| [merlin-pytorch](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-pytorch) | Merlin Libraries - in particular Transformer4Rec, Triton, PyTorch |

To use these containers, you must install the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) to provide GPU support for Docker.

You can find the Dockerfiles in our [GitHub repository](https://github.com/NVIDIA-Merlin/Merlin/tree/main/docker)

You can use the NGC links referenced in the preceding table for more information about how to launch and run these containers.

## Using PIP

In some cases, you need to build your own docker container or build on top of another base image. You can install Merlin and its dependencies via `pip` for training your models with TensorFlow or PyTorch. Review the [Triton Documentation](https://github.com/triton-inference-server/server#documentation) to install Triton Inference Server without docker or use our docker container.

Pre-Requirements:
CUDA>=11.8 for [RAPIDs cuDF](https://rapids.ai/pip.html). An example installation can be found [here](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=18.04&target_type=runfile_local)
Python3.8
docker Image `nvcr.io/nvidia/cuda:11.8.0-devel-ubuntu22.04` or equivalent
docker with [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)

1. Installation of basic requirements, Python3.8 and other libraries via apt

```shell
apt update && \
    apt install software-properties-common curl git -y && \
    add-apt-repository ppa:deadsnakes/ppa -y

apt update && \
    TZ=Etc/UTC apt install -y \
        python3.8 \
        python3.8-dev \
        python3.8-distutils \
        graphviz \
        libcudnn8
```

2. Create symbolic link to use python3.8
   
```shell
ln -s /usr/bin/python3.8 /usr/bin/python
```

3. Install Pip

```shell
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.8 get-pip.py
```

4. Install RAPIDs cuDF

```shell
pip install cudf-cu11 dask-cudf-cu11 --extra-index-url=https://pypi.nvidia.com
```

5. Install preferred deep learning framework. It is possible to install both TensorFlow and PyTorch. 

```shell
pip install tensorflow-gpu==2.9.2
pip install torchmetrics==0.10.0 torch --extra-index-url https://download.pytorch.org/whl/cu117
```

6. Install libraries via pip

```shell
pip install pytest fiddle wandb nest-asyncio testbook git+https://github.com/rapidsai/asvdb.git@main && \
pip install scikit-learn tritonclient[all] && \ 
pip install protobuf==3.20.3 pynvml ipython ipykernel graphviz && \
pip install merlin-core nvtabular merlin-dataloader merlin-systems merlin-models transformers4rec
```



[7. Optional: Clone the Merlin, Merlin Models or Transformer4Rec repositories to download the examples. **Important You need to checkout the version (tag) corresponding to the pip install**.]


## Using Conda

In some cases, you need to build your own docker container or build on top of another base image. You can install Merlin and its dependencies via `conda` for training your models with TensorFlow or PyTorch. Review the [Triton Documentation](https://github.com/triton-inference-server/server#documentation) to install Triton Inference Server without docker or use our docker container.

Requirements:
Docker Image: Ubuntu18.04


## Hosted on NVIDIA LaunchPad

[NVIDIA LaunchPad](https://www.nvidia.com/en-us/launchpad/) provides free, short-term access to many hands-on labs. We provide a hands-on tutorial for session-based recommender systems to predict the next item using Transformer4Rec in PyTorch.

Sign up and check it out for free: [Build Session-Based Recommenders on NVIDIA LaunchPad](https://www.nvidia.com/en-us/launchpad/ai/build-session-based-recommenders/)

After sign-up, it will take ˜3-5 business days until the request is granted. You will receive an email.

## Hosted on Google Colab (experimental)

As Colab runs on Python3.8 and RAPIDs supports `pip` installation, we successfully were able to run our examples on the Google Colab environment. We cannot automate regular testing of our installation scripts and examples. Therefore, we share our instructions for the Merlin 22.12 version, but cannot guarantee that it will continuously work.

You can read how to run Merlin on Colab in [our blog](https://medium.com/nvidia-merlin/how-to-run-merlin-on-google-colab-83b5805c63e0).
