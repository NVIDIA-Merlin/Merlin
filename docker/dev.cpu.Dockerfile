FROM ubuntu:22.04

# -----------------------------------------------------------------------------
# Install system packages
# -----------------------------------------------------------------------------

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    software-properties-common \
    graphviz `# For visualizing merlin graphs` \
    ripgrep

RUN add-apt-repository ppa:deadsnakes/ppa -y

RUN apt-get update && TZ=Etc/UTC apt install -y \
    python3.8 \
    python3.8-dev \
    python3.8-distutils

RUN ln -s /usr/bin/python3.8 /usr/bin/python

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py

# Update pip
RUN pip install --no-cache-dir --upgrade \
    pip \
    virtualenv

# -----------------------------------------------------------------------------
# Setup Virtualenv
# -----------------------------------------------------------------------------

# The virtual env is required to allow us to install the merlin packages
# in development (editable) mode succesfully

ENV VIRTUALENV_PATH=/venv
RUN python -m virtualenv $VIRTUALENV_PATH
ENV PATH="${VIRTUALENV_PATH}/bin:${PATH}"

WORKDIR /workspace

# install common dev requirements
COPY requirements/dev.cpu.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt

# -----------------------------------------------------------------------------
# Install Merlin Repos
# -----------------------------------------------------------------------------

COPY merlin merlin

# install merlin packages in development mode
RUN python -m pip install -e merlin/models --no-deps
RUN python -m pip install -e merlin/core --no-deps
RUN python -m pip install -e merlin/NVTabular --no-deps
RUN python -m pip install -e merlin/systems --no-deps
RUN python -m pip install -e merlin/dataloader --no-deps
RUN python -m pip install -e merlin/Transformers4Rec --no-deps
