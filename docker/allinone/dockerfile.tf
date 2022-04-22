# syntax=docker/dockerfile:1.2
ARG MERLIN_VERSION=22.03
ARG BASE_IMAGE=nvcr.io/nvidia/merlin/merlin-base:${MERLIN_VERSION}

FROM ${BASE_IMAGE} as base

COPY --chown=1000:1000 --from=build /opt/tritonserver/backends/tensorflow2 backends/tensorflow2/
RUN pip install tensorflow-gpu
COPY --chown=1000:1000 --from=dlfw /usr/local/lib/python3.8/dist-packages/tensorflow /usr/local/lib/python3.8/dist-packages/tensorflow/
COPY --chown=1000:1000 --from=dlfw /usr/local/lib/tensorflow/ /usr/local/lib/tensorflow/
COPY --chown=1000:1000 --from=dlfw /usr/local/lib/python3.8/dist-packages/horovod /usr/local/lib/python3.8/dist-packages/horovod/
COPY --chown=1000:1000 --from=dlfw /usr/local/bin/horovodrun /usr/local/bin/horovodrun

HEALTHCHECK NONE
CMD ["/bin/bash"]