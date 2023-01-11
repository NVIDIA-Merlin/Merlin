# syntax=docker/dockerfile:1.2
ARG MERLIN_VERSION=22.12
ARG TRITON_VERSION=22.11
ARG TENSORFLOW_VERSION=22.11

ARG DLFW_IMAGE=nvcr.io/nvidia/tensorflow:${TENSORFLOW_VERSION}-tf2-py3
ARG FULL_IMAGE=nvcr.io/nvidia/tritonserver:${TRITON_VERSION}-py3
ARG BASE_IMAGE=nvcr.io/nvstaging/merlin/merlin-base:${MERLIN_VERSION}

FROM ${DLFW_IMAGE} as dlfw
FROM ${FULL_IMAGE} as triton
FROM ${BASE_IMAGE} as base

# Triton TF backends
COPY --chown=1000:1000 --from=triton /opt/tritonserver/backends/tensorflow2 backends/tensorflow2/

# Tensorflow dependencies (only)
# Pinning to pass hugectr sok tests
RUN pip install tensorflow-gpu==2.9.2 transformers==4.23.1\
    && pip uninstall tensorflow-gpu keras -y

# DLFW Tensorflow packages
COPY --chown=1000:1000 --from=dlfw /usr/local/lib/python3.8/dist-packages/tensorflow /usr/local/lib/python3.8/dist-packages/tensorflow/
COPY --chown=1000:1000 --from=dlfw /usr/local/lib/python3.8/dist-packages/tensorflow-*.dist-info /usr/local/lib/python3.8/dist-packages/tensorflow.dist-info/
COPY --chown=1000:1000 --from=dlfw /usr/local/lib/python3.8/dist-packages/keras /usr/local/lib/python3.8/dist-packages/keras/
COPY --chown=1000:1000 --from=dlfw /usr/local/lib/python3.8/dist-packages/keras-*.dist-info /usr/local/lib/python3.8/dist-packages/keras.dist-info/
COPY --chown=1000:1000 --from=dlfw /usr/local/bin/saved_model_cli /usr/local/bin/saved_model_cli
COPY --chown=1000:1000 --from=dlfw /usr/local/lib/tensorflow/ /usr/local/lib/tensorflow/
COPY --chown=1000:1000 --from=dlfw /usr/local/lib/python3.8/dist-packages/horovod /usr/local/lib/python3.8/dist-packages/horovod/
COPY --chown=1000:1000 --from=dlfw /usr/local/lib/python3.8/dist-packages/horovod-*.dist-info /usr/local/lib/python3.8/dist-packages/horovod.dist-info/
COPY --chown=1000:1000 --from=dlfw /usr/local/bin/horovodrun /usr/local/bin/horovodrun

# Install dependencies for hps tf plugin
RUN apt update -y --fix-missing && \
    apt install -y --no-install-recommends \
        #   Required to build RocksDB.
            libgflags-dev \
            zlib1g-dev libbz2-dev libsnappy-dev liblz4-dev libzstd-dev \
        #   Required to build RdKafka.
            zlib1g-dev libzstd-dev \
            libssl-dev libsasl2-dev && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*


# Install HugeCTR
ENV LD_LIBRARY_PATH=/usr/local/hugectr/lib:$LD_LIBRARY_PATH \
    LIBRARY_PATH=/usr/local/hugectr/lib:$LIBRARY_PATH \
    SOK_COMPILE_UNIT_TEST=ON

# Arguments "_XXXX" are only valid when $HUGECTR_DEV_MODE==false
ARG HUGECTR_DEV_MODE=false
ARG _HUGECTR_REPO="github.com/NVIDIA-Merlin/HugeCTR.git"
ARG _CI_JOB_TOKEN=""
ARG HUGECTR_VER=main

RUN mkdir -p /usr/local/nvidia/lib64 && \
    ln -s /usr/local/cuda/lib64/libcusolver.so /usr/local/nvidia/lib64/libcusolver.so.10

RUN ln -s /usr/lib/x86_64-linux-gnu/libibverbs.so.1 /usr/lib/x86_64-linux-gnu/libibverbs.so

# Install distributed-embeddings and sok
ARG INSTALL_DISTRIBUTED_EMBEDDINGS=true
ARG TFDE_VER=v0.2
RUN if [ "$HUGECTR_DEV_MODE" == "false" ]; then \
        git clone --branch ${HUGECTR_VER} --depth 1 https://${_CI_JOB_TOKEN}${_HUGECTR_REPO} /hugectr && \
        pushd /hugectr && \
	pip install ninja && \
	git submodule update --init --recursive && \
        # Install SOK
        cd sparse_operation_kit && \
        python setup.py install && \
        # Install HPS TF plugin
        cd ../hps_tf && \
        python setup.py install && \
        popd; \
    fi; \
    if [ "$INSTALL_DISTRIBUTED_EMBEDDINGS" == "true" ]; then \
        git clone https://github.com/NVIDIA-Merlin/distributed-embeddings.git /distributed_embeddings/ && \
        cd /distributed_embeddings && git checkout ${TFDE_VER} && git submodule update --init --recursive && \
        make pip_pkg && pip install artifacts/*.whl && make clean; \
    fi; \
    rm -rf /hugectr
    
