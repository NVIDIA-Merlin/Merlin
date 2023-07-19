# syntax=docker/dockerfile:1.2
ARG MERLIN_VERSION=23.06
ARG TRITON_VERSION=23.06
ARG TENSORFLOW_VERSION=23.06

ARG DLFW_IMAGE=nvcr.io/nvidia/tensorflow:${TENSORFLOW_VERSION}-tf2-py3
ARG FULL_IMAGE=nvcr.io/nvidia/tritonserver:${TRITON_VERSION}-py3
ARG BASE_IMAGE=nvcr.io/nvstaging/merlin/merlin-base:${MERLIN_VERSION}

FROM ${DLFW_IMAGE} as dlfw
FROM ${FULL_IMAGE} as triton
FROM ${BASE_IMAGE} as base

# Triton TF backends
COPY --chown=1000:1000 --from=triton /opt/tritonserver/backends/tensorflow backends/tensorflow/

# Tensorflow dependencies (only)
# Pinning to pass hugectr sok tests
# wrapt 1.5.0 introduce hugectr test failures, so downgrade to 1.14.0
RUN pip install --no-cache-dir tensorflow protobuf==3.20.3 wrapt==1.14.0 \
    && pip uninstall tensorflow keras -y

# DLFW Tensorflow packages
COPY --chown=1000:1000 --from=dlfw /usr/local/lib/python${PYTHON_VERSION}/dist-packages/tensorflow /usr/local/lib/python${PYTHON_VERSION}/dist-packages/tensorflow/
COPY --chown=1000:1000 --from=dlfw /usr/local/lib/python${PYTHON_VERSION}/dist-packages/tensorflow-*.dist-info /usr/local/lib/python${PYTHON_VERSION}/dist-packages/tensorflow.dist-info/
COPY --chown=1000:1000 --from=dlfw /usr/local/lib/python${PYTHON_VERSION}/dist-packages/keras /usr/local/lib/python${PYTHON_VERSION}/dist-packages/keras/
COPY --chown=1000:1000 --from=dlfw /usr/local/lib/python${PYTHON_VERSION}/dist-packages/keras-*.dist-info /usr/local/lib/python${PYTHON_VERSION}/dist-packages/keras.dist-info/
COPY --chown=1000:1000 --from=dlfw /usr/local/bin/saved_model_cli /usr/local/bin/saved_model_cli
COPY --chown=1000:1000 --from=dlfw /usr/local/lib/tensorflow/ /usr/local/lib/tensorflow/
COPY --chown=1000:1000 --from=dlfw /usr/local/lib/python${PYTHON_VERSION}/dist-packages/horovod /usr/local/lib/python${PYTHON_VERSION}/dist-packages/horovod/
COPY --chown=1000:1000 --from=dlfw /usr/local/lib/python${PYTHON_VERSION}/dist-packages/horovod-*.dist-info /usr/local/lib/python${PYTHON_VERSION}/dist-packages/horovod.dist-info/
COPY --chown=1000:1000 --from=dlfw /usr/local/bin/horovodrun /usr/local/bin/horovodrun

# Need to install transformers after tensorflow has been pulled in, so it builds artifacts correctly.
RUN pip install --no-cache-dir transformers==4.26.0

# Install HugeCTR
# Arguments "_XXXX" are only valid when $HUGECTR_DEV_MODE==false
ARG HUGECTR_DEV_MODE=false
ARG _HUGECTR_REPO="github.com/NVIDIA-Merlin/HugeCTR.git"
ARG _CI_JOB_TOKEN=""
ARG HUGECTR_VER=main

ENV LD_LIBRARY_PATH=/usr/local/lib/python${PYTHON_VERSION}/dist-packages/tensorflow:$LD_LIBRARY_PATH \
    LIBRARY_PATH=${HUGECTR_HOME}/lib:$LIBRARY_PATH \
    SOK_COMPILE_UNIT_TEST=ON

RUN mkdir -p /usr/local/nvidia/lib64 && \
    ln -s /usr/local/cuda/lib64/libcusolver.so /usr/local/nvidia/lib64/libcusolver.so

RUN ln -s libibverbs.so.1 $(find /usr/lib/*-linux-gnu/libibverbs.so.1 | sed -e 's/\.1$//g')

# Install distributed-embeddings and sok
ARG INSTALL_DISTRIBUTED_EMBEDDINGS=false
ARG TFDE_VER=v23.03.00

RUN if [ "$HUGECTR_DEV_MODE" == "false" ]; then \
        git clone --branch ${HUGECTR_VER} --depth 1 --recurse-submodules --shallow-submodules https://${_CI_JOB_TOKEN}${_HUGECTR_REPO} /hugectr && \
        pushd /hugectr && \
        rm -rf .git/modules && \
        pip --no-cache-dir install ninja tf2onnx && \
        # Install SOK
        cd sparse_operation_kit && \
        python setup.py install && \
        # Install HPS TF plugin
        cd ../hps_tf && \
        python setup.py install && \
        popd && \
        mv /hugectr/ci ~/hugectr-ci && mv /hugectr/sparse_operation_kit ~/hugectr-sparse_operation_kit && \
    	rm -rf /hugectr && mkdir -p /hugectr && \
        mv ~/hugectr-ci /hugectr/ci && mv ~/hugectr-sparse_operation_kit /hugectr/sparse_operation_kit \
    ; fi && \
    if [ "$INSTALL_DISTRIBUTED_EMBEDDINGS" == "true" ]; then \
        git clone --branch ${TFDE_VER} --depth 1 https://github.com/NVIDIA-Merlin/distributed-embeddings.git /distributed_embeddings/ && \
        cd /distributed_embeddings && git submodule update --init --recursive && \
        make pip_pkg && pip install --no-cache-dir artifacts/*.whl && make clean \
    ; fi

