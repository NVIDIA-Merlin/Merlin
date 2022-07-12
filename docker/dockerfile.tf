# syntax=docker/dockerfile:1.2
ARG MERLIN_VERSION=22.06
ARG TRITON_VERSION=22.05
ARG TENSORFLOW_VERSION=22.05

ARG DLFW_IMAGE=nvcr.io/nvidia/tensorflow:${TENSORFLOW_VERSION}-tf2-py3
ARG FULL_IMAGE=nvcr.io/nvidia/tritonserver:${TRITON_VERSION}-py3
ARG BASE_IMAGE=nvcr.io/nvstaging/merlin/merlin-base:${MERLIN_VERSION}

FROM ${DLFW_IMAGE} as dlfw
FROM ${FULL_IMAGE} as triton
FROM ${BASE_IMAGE} as base

# Triton TF backends
COPY --chown=1000:1000 --from=triton /opt/tritonserver/backends/tensorflow2 backends/tensorflow2/

# Tensorflow dependencies (only)
RUN pip install tensorflow-gpu \
    && pip uninstall tensorflow-gpu keras -y

# DLFW Tensorflow packages
COPY --chown=1000:1000 --from=dlfw /usr/local/lib/python3.8/dist-packages/tensorflow /usr/local/lib/python3.8/dist-packages/tensorflow/
COPY --chown=1000:1000 --from=dlfw /usr/local/lib/python3.8/dist-packages/keras /usr/local/lib/python3.8/dist-packages/keras/
COPY --chown=1000:1000 --from=dlfw /usr/local/lib/tensorflow/ /usr/local/lib/tensorflow/
COPY --chown=1000:1000 --from=dlfw /usr/local/lib/python3.8/dist-packages/horovod /usr/local/lib/python3.8/dist-packages/horovod/
COPY --chown=1000:1000 --from=dlfw /usr/local/bin/horovodrun /usr/local/bin/horovodrun

# Install cmake
RUN wget http://www.cmake.org/files/v3.21/cmake-3.21.1.tar.gz && \
    tar xf cmake-3.21.1.tar.gz && cd cmake-3.21.1 && ./configure && make && make install


# Install HugeCTR
ENV LD_LIBRARY_PATH=/usr/local/hugectr/lib:$LD_LIBRARY_PATH \
    LIBRARY_PATH=/usr/local/hugectr/lib:$LIBRARY_PATH \
    SOK_COMPILE_UNIT_TEST=ON

# Arguments "_XXXX" are only valid when $HUGECTR_DEV_MODE==false
ARG HUGECTR_DEV_MODE=false
ARG _HUGECTR_REPO="github.com/NVIDIA-Merlin/HugeCTR.git"
ARG _CI_JOB_TOKEN=""

RUN mkdir -p /usr/local/nvidia/lib64 && \
    ln -s /usr/local/cuda/lib64/libcusolver.so /usr/local/nvidia/lib64/libcusolver.so.10

RUN ln -s /usr/lib/x86_64-linux-gnu/libibverbs.so.1 /usr/lib/x86_64-linux-gnu/libibverbs.so

RUN if [ "$HUGECTR_DEV_MODE" == "false" ]; then \
        git clone https://${_CI_JOB_TOKEN}${_HUGECTR_REPO} /hugectr && \
        pushd /hugectr && \
          git checkout ${HUGECTR_VER} && \
          cd sparse_operation_kit && \
          python setup.py install && \
        popd; \
    fi

# Install distributed-embeddings
ARG INSTALL_DISTRIBUTED_EMBEDDINGS=true
RUN if [ "$INSTALL_DISTRIBUTED_EMBEDDINGS" == "true" ]; then \
        git clone https://github.com/NVIDIA-Merlin/distributed-embeddings.git /distributed_embeddings/ && \
        cd /distributed_embeddings && git checkout ${TFDE_VER} && \
        make pip_pkg && pip install artifacts/*.whl && make clean; \
    fi
