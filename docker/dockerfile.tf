# syntax=docker/dockerfile:1
ARG IMAGE=nvcr.io/nvidia/tensorflow:21.11-tf2-py3
FROM ${IMAGE} AS phase1
ENV CUDA_SHORT_VERSION=11.4

SHELL ["/bin/bash", "-c"]
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/lib:/repos/dist/lib

ENV DEBIAN_FRONTEND=noninteractive

ARG RELEASE=false
ARG NVTAB_VER=vnightly
ARG TF4REC_VER=vnightly
ARG HUGECTR_VER=vnightly
ARG SM="60;61;70;75;80"

ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_PATH=$CUDA_HOME
ENV CUDA_CUDA_LIBRARY=${CUDA_HOME}/lib64/stubs
ENV PATH=${CUDA_HOME}/lib64/:${PATH}:${CUDA_HOME}/bin

RUN apt update -y --fix-missing && \
    apt upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        gdb \
        valgrind \
        zlib1g-dev lsb-release clang-format libboost-serialization-dev \
        openssl \
        libssl-dev \
        protobuf-compiler \
        libaio-dev \
        slapd && \
      apt install -y --no-install-recommends software-properties-common && \
      add-apt-repository -y ppa:deadsnakes/ppa && \
      apt update -y --fix-missing

RUN pip install git+git://github.com/gevent/gevent.git@21.8.0#egg=gevent

# Install cmake
RUN apt remove --purge cmake -y && wget http://www.cmake.org/files/v3.21/cmake-3.21.1.tar.gz && \
    tar xf cmake-3.21.1.tar.gz && cd cmake-3.21.1 && ./configure && make && make install

# Install spdlog from source
RUN git clone --branch v1.9.2 https://github.com/gabime/spdlog.git build-env && \
    pushd build-env && \
      mkdir build && cd build && cmake .. && make -j && make install && \
    popd && \
    rm -rf build-env

FROM phase1 as phase2

ARG RELEASE=false
ARG NVTAB_VER=vnightly
ARG TF4REC_VER=vnightly

ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python'

RUN pip install pandas sklearn ortools pydot && \
    pip cache purge

RUN pip install pybind11
SHELL ["/bin/bash", "-c"]

# Install NVTabular
RUN git clone https://github.com/NVIDIA-Merlin/NVTabular.git /nvtabular/ && \
    cd /nvtabular/; if [ "$RELEASE" == "true" ] && [ ${NVTAB_VER} != "vnightly" ] ; then git fetch --all --tags && git checkout tags/${NVTAB_VER}; else git checkout main; fi; \
    python setup.py develop;

# Install Transformers4Rec
RUN git clone https://github.com/NVIDIA-Merlin/Transformers4Rec.git /transformers4rec && \
    cd /transformers4rec/;  if [ "$RELEASE" == "true" ] && [ ${TF4REC_VER} != "vnightly" ] ; then git fetch --all --tags && git checkout tags/${TF4REC_VER}; else git checkout main; fi; \
    pip install -e .[tensorflow,nvtabular]

RUN pip install pynvml pytest graphviz sklearn scipy matplotlib 
RUN pip install nvidia-pyindex; pip install tritonclient[all] grpcio-channelz
RUN pip install nvtx cupy-cuda114 cachetools typing_extensions fastavro

RUN apt-get update; apt-get install -y graphviz

ENV LD_LIBRARY_PATH=/usr/local/hugectr/lib:$LD_LIBRARY_PATH \
    LIBRARY_PATH=/usr/local/hugectr/lib:$LIBRARY_PATH \
    PYTHONPATH=/usr/local/hugectr/lib:$PYTHONPATH

RUN git clone https://github.com/rapidsai/asvdb.git build-env && \
    pushd build-env && \
      python setup.py install && \
    popd && \
    rm -rf build-env

RUN pip install dask==2021.07.1 distributed==2021.07.1 dask[dataframe]==2021.07.1 dask-cuda
FROM phase2 as phase3

ARG RELEASE=false
ARG HUGECTR_VER=vnightly
ARG SM="60;61;70;75;80"
ARG USE_NVTX=OFF

# Arguments "_XXXX" are only valid when $HUGECTR_DEV_MODE==false
ARG HUGECTR_DEV_MODE=false
ARG _HUGECTR_BRANCH=master
ARG _HUGECTR_REPO="github.com/NVIDIA-Merlin/HugeCTR.git"
ARG _CI_JOB_TOKEN=""

RUN mkdir -p /usr/local/nvidia/lib64 && \
    ln -s /usr/local/cuda/lib64/libcusolver.so /usr/local/nvidia/lib64/libcusolver.so.10

RUN ln -s /usr/lib/x86_64-linux-gnu/libibverbs.so.1 /usr/lib/x86_64-linux-gnu/libibverbs.so

RUN if [ "$HUGECTR_DEV_MODE" == "false" ]; then \
        git clone https://${_CI_JOB_TOKEN}${_HUGECTR_REPO} build-env && pushd build-env && git fetch --all; \
        if [ "$RELEASE" == "true" ] && [ ${HUGECTR_VER} != "vnightly" ]; then \
            git fetch --all --tags && git checkout tags/${HUGECTR_VER}; \
        else \
            git checkout ${_HUGECTR_BRANCH}; \
        fi; \
        cd sparse_operation_kit && \
        python setup.py install && \
        popd && \
        rm -rf build-env; \
    fi

RUN pip install pybind11
RUN pip install numba numpy --upgrade

RUN rm -rf /usr/local/share/jupyter/lab/staging/node_modules/fast-json-patch

SHELL ["/bin/bash", "-c"]

RUN echo $(du -h --max-depth=1 /)

HEALTHCHECK NONE
ENTRYPOINT []
CMD ["/bin/bash"]
