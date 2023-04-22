FROM python:3.8.16-slim-buster

RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential `For building NVTabular C++ code` \
    graphviz `# For visualizing merlin graphs` \
    ripgrep

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
