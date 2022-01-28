# syntax=docker/dockerfile:1
ARG IMAGE=nvcr.io/nvidia/tensorflow:22.01-tf2-py3
FROM ${IMAGE}

# Args
ARG RELEASE=false
ARG NVTAB_VER=vnightly
ARG HUGECTR_VER=vnightly
ARG TF4REC_VER=vnightly

# Envs
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/lib:/repos/dist/lib
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_PATH=$CUDA_HOME
ENV CUDA_CUDA_LIBRARY=${CUDA_HOME}/lib64/stubs
ENV PATH=${CUDA_HOME}/lib64/:${PATH}:${CUDA_HOME}/bin

# Install packages
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update -y --fix-missing && \
    apt upgrade -y && \
    apt install -y --no-install-recommends software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt update -y --fix-missing && \
    apt-get install -y --no-install-recommends \
        gdb \
        valgrind \
        zlib1g-dev lsb-release clang-format libboost-serialization-dev \
        openssl \
        graphviz \
        libssl-dev \
        protobuf-compiler \
        libaio-dev \
        slapd && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install cmake
RUN apt remove --purge cmake -y && wget http://www.cmake.org/files/v3.21/cmake-3.21.1.tar.gz && \
    tar xf cmake-3.21.1.tar.gz && cd cmake-3.21.1 && ./configure && make && make install

# Install multiple packages
RUN pip cache purge
RUN pip uninstall cupy-cuda114 -y
RUN pip install nvtx pandas cupy-cuda115 cachetools typing_extensions fastavro
RUN pip install pynvml pytest graphviz scipy matplotlib tqdm pydot nvidia-pyindex
RUN pip install tritonclient[all] grpcio-channelz
RUN pip install pybind11 jupyterlab gcsfs
RUN pip3 install --no-cache-dir mpi4py ortools sklearn onnx onnxruntime
RUN pip install dask==2021.09.1 distributed==2021.09.1 dask[dataframe]==2021.09.1 dask-cuda
RUN pip install gevent==21.8.0
RUN git clone https://github.com/rapidsai/asvdb.git /repos/asvdb && cd /repos/asvdb && python setup.py install

ARG INSTALL_NVT=true
# Install NVTabular
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python'
RUN if [ "$INSTALL_NVT" == "true" ]; then \
        git clone https://github.com/NVIDIA-Merlin/NVTabular.git /nvtabular/ && \
        cd /nvtabular/; if [ "$RELEASE" == "true" ] && [ ${NVTAB_VER} != "vnightly" ] ; then git fetch --all --tags && git checkout tags/${NVTAB_VER}; else git checkout main; fi; \
        python setup.py develop; \
    fi

# Install Transformers4Rec
RUN if [ "$INSTALL_NVT" == "true" ]; then \
        git clone https://github.com/NVIDIA-Merlin/Transformers4Rec.git /transformers4rec && \
        cd /transformers4rec/;  if [ "$RELEASE" == "true" ] && [ ${TF4REC_VER} != "vnightly" ] ; then git fetch --all --tags && git checkout tags/${TF4REC_VER}; else git checkout main; fi; \
        pip install -e .[tensorflow,nvtabular] && python setup.py develop; \
    fi

# Install HugeCTR
ENV LD_LIBRARY_PATH=/usr/local/hugectr/lib:$LD_LIBRARY_PATH \
    LIBRARY_PATH=/usr/local/hugectr/lib:$LIBRARY_PATH \
    PYTHONPATH=/usr/local/hugectr/lib:$PYTHONPATH

RUN git clone https://github.com/rapidsai/asvdb.git build-env && \
    pushd build-env && \
      python setup.py install && \
    popd && \
    rm -rf build-env

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

# Clean up
RUN rm -rf /repos
RUN pip install numba numpy --upgrade
RUN pip install dask==2021.09.1 distributed==2021.09.1 dask[dataframe]==2021.09.1 dask-cuda
RUN rm -rf /usr/local/share/jupyter/lab/staging/node_modules/fast-json-patch

RUN echo $(du -h --max-depth=1 /)

HEALTHCHECK NONE
CMD ["/bin/bash"]
