# syntax=docker/dockerfile:1
ARG IMAGE=nvcr.io/nvidia/tensorflow:22.01-tf2-py3
FROM ${IMAGE}

# Args
ARG HUGECTR_VER=master
ARG NVTAB_VER=main
ARG MODELS_VER=main
ARG TF4REC_VER=main

# Envs
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/lib:/repos/dist/lib
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_PATH=$CUDA_HOME
ENV CUDA_CUDA_LIBRARY=${CUDA_HOME}/lib64/stubs
ENV PATH=${CUDA_HOME}/lib64/:${PATH}:${CUDA_HOME}/bin

# Install packages
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update -y --fix-missing && \
    apt install -y --no-install-recommends software-properties-common && \
    apt-get install -y --no-install-recommends \
    #    gdb \
    #    valgrind \
    #    zlib1g-dev lsb-release clang-format libboost-serialization-dev \
    #    openssl \
        graphviz \
    #    libssl-dev \
        protobuf-compiler && \
    #    libaio-dev \
    #    slapd && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install multiple packages
RUN pip install pytest
RUN pip install graphviz
RUN pip install pydot
RUN pip install nvidia-pyindex
#RUN pip install nvtx pandas cupy-cuda115 cachetools typing_extensions fastavro
#RUN pip install pynvml pytest graphviz scipy matplotlib tqdm pydot nvidia-pyindex
RUN pip install tritonclient[all] grpcio-channelz
#RUN pip install pybind11 jupyterlab gcsfs
RUN pip install pybind11
#RUN pip install --no-cache-dir mpi4py ortools sklearn onnx onnxruntime
#RUN pip install dask==2021.11.2 distributed==2021.11.2 dask[dataframe]==2021.11.2 dask-cuda
RUN pip install betterproto
#RUN pip install gevent==21.8.0
RUN pip install --no-cache-dir git+https://github.com/rapidsai/asvdb.git@main
#RUN pip install transformers

ARG INSTALL_NVT=true
# Install NVTabular
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python'
RUN if [ "$INSTALL_NVT" == "true" ]; then \
      git clone https://github.com/NVIDIA-Merlin/NVTabular.git /nvtabular/ && \
      cd /nvtabular/ && git checkout ${NVTAB_VER} && pip install . --no-deps && python setup.py develop --no-deps; \
    fi

# Install Transformers4Rec
RUN if [ "$INSTALL_NVT" == "true" ]; then \
      git clone https://github.com/NVIDIA-Merlin/Transformers4Rec.git /transformers4rec && \
      cd /transformers4rec/ && git checkout ${TF4REC_VER} && pip install .[tensorflow,nvtabular] --no-deps; \
    fi

# Install Models
RUN git clone https://github.com/NVIDIA-Merlin/Models.git /models/ && \
    cd /models/ && git checkout ${MODELS_VER} && pip install . --no-deps

# Install HugeCTR
ENV LD_LIBRARY_PATH=/usr/local/hugectr/lib:$LD_LIBRARY_PATH \
    LIBRARY_PATH=/usr/local/hugectr/lib:$LIBRARY_PATH \
    PYTHONPATH=/usr/local/hugectr/lib:$PYTHONPATH

# Arguments "_XXXX" are only valid when $HUGECTR_DEV_MODE==false
ARG HUGECTR_DEV_MODE=false
ARG _HUGECTR_REPO="github.com/NVIDIA-Merlin/HugeCTR.git"
ARG _CI_JOB_TOKEN=""

RUN mkdir -p /usr/local/nvidia/lib64 && \
    ln -s /usr/local/cuda/lib64/libcusolver.so /usr/local/nvidia/lib64/libcusolver.so.10

RUN ln -s /usr/lib/x86_64-linux-gnu/libibverbs.so.1 /usr/lib/x86_64-linux-gnu/libibverbs.so

RUN if [ "$HUGECTR_DEV_MODE" == "false" ]; then \
        git clone https://${_CI_JOB_TOKEN}${_HUGECTR_REPO} build-env && \
        pushd build-env && \ 
          git checkout ${HUGECTR_VER} && \
          cd sparse_operation_kit && \
          python setup.py install && \
        popd && \
        rm -rf build-env; \
    fi

# Clean up
RUN rm -rf /repos

HEALTHCHECK NONE
CMD ["/bin/bash"]
