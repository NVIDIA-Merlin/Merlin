FROM nvcr.io/nvidia/tritonserver:23.02-py3

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN apt-get update
# Packages required for cudf with cuda 11.8
RUN apt-get install -y libcusparse-11-8 libcufft-11-8 cuda-nvcc-11-8

WORKDIR /workspace

COPY merlin merlin

# cudf required for NVTabular workflows generated on GPU devices
RUN python -m pip install --no-cache-dir cudf-cu11==23.02.0 dask-cudf-cu11==23.02.0 --extra-index-url=https://pypi.nvidia.com

RUN python -m pip install -e merlin/core

RUN python -m pip install -e merlin/systems --no-deps
RUN python -m pip install tritonclient[grpc] scipy

# first install to get nvtabular_cpp installed
RUN python -m pip install merlin/NVTabular --no-deps
# second editable install for python version
RUN python -m pip install -e merlin/NVTabular --no-deps

