FROM nvcr.io/nvidia/merlin/merlin-tensorflow:22.12

# -----------------------------------------------------------------------------
# Install system packages
# -----------------------------------------------------------------------------

RUN apt-get update
RUN apt-get install -y \
    graphviz `# For visualizing merlin graphs` \
    sudo `# for installing deps from git repos`

# Update pip
RUN pip install --upgrade \
    pip \
    virtualenv

# remove default installation of merlin packages
RUN pip uninstall -y \
    merlin-core \
    merlin-models \
    merlin-systems \
    nvtabular \
    transformers4rec \
    ipython \
    jupyter

# -----------------------------------------------------------------------------
# Setup User
# -----------------------------------------------------------------------------

ARG UID
ARG GID
ARG USER

RUN addgroup --gid $GID $USER
RUN adduser --disabled-password --gecos '' --uid $UID --gid $GID $USER
RUN adduser $USER sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER $USER

# -----------------------------------------------------------------------------
# Setup Virtualenv
# -----------------------------------------------------------------------------

# The virtual env is required to allow us to install the merlin packages
# in development (editable) mode succesfully

ENV VIRTUALENV_PATH=/home/$USER/venv
RUN python -m virtualenv $VIRTUALENV_PATH
ENV PATH="${VIRTUALENV_PATH}/bin:${PATH}"

# Extending the PYTHONPATH so that we can continue to find packages already installed
# e.g 'cudf', 'cuda-python' and others in /usr/local/lib/
ENV PYTHONPATH="${VIRTUALENV_PATH}/lib/python3.8/site-packages:/usr/local/lib/python3.8/dist-packages"

WORKDIR /workspace/dev

# install common dev requirements
COPY requirements.in /tmp/requirements.in
RUN python -m pip install -r /tmp/requirements.in

# -----------------------------------------------------------------------------
# Install Merlin Repos
# -----------------------------------------------------------------------------

COPY --chown=$USER merlin merlin

# # # install merlin packages in development mode
RUN python -m pip install -e merlin/models --no-deps
RUN python -m pip install -e merlin/core --no-deps
RUN python -m pip install -e merlin/NVTabular --no-deps
RUN python -m pip install -e merlin/systems --no-deps
RUN python -m pip install -e merlin/dataloader --no-deps
