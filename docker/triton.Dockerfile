FROM nvcr.io/nvidia/tritonserver:23.02-py3

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /workspace

COPY merlin merlin

RUN python -m pip install -e merlin/core
# first install to get nvtabular_cpp installed
RUN python -m pip install merlin/NVTabular --no-deps
# second editable install for python version
RUN python -m pip install -e merlin/NVTabular --no-deps
RUN python -m pip install -e merlin/systems --no-deps
RUN python -m pip install tritonclient[grpc] scipy
