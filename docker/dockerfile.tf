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


HEALTHCHECK NONE
CMD ["/bin/bash"]