FROM nvcr.io/nvidia/tensorflow:21.03-tf2-py3

ARG RELEASE=false
ARG NVTAB_VER=v0.5.0
ARG HUGECTR_VER=v3.0.1
ARG SM="60;61;70;75;80"

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        vim gdb git wget unzip tar python3.8-dev \
        zlib1g-dev lsb-release clang-format libboost-all-dev \
        openssl libssl1.1 curl zip\
       	slapd=2.4.49+dfsg-2ubuntu1.7 && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /var/tmp && wget -q -nc --no-check-certificate -P /var/tmp http://repo.anaconda.com/miniconda/Miniconda3-4.7.12-Linux-x86_64.sh && \
    bash /var/tmp/Miniconda3-4.7.12-Linux-x86_64.sh -b -p /opt/conda && \
    /opt/conda/bin/conda init && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    . /opt/conda/etc/profile.d/conda.sh && \
    conda activate base && \
    conda update -n base -c defaults conda && \
    conda config --add channels conda-forge --add channels nvidia --add channels rapidsai --add channels anaconda && \
    conda install -y cmake=3.19.6 pip rmm=0.18 cudatoolkit=11.0 && \
    /opt/conda/bin/conda clean -afy && \
    rm -rf /var/tmp/Miniconda3-4.7.12-Linux-x86_64.sh && \
    rm -rfv /opt/conda/include/nccl.h /opt/conda/lib/libnccl.so /opt/conda/include/google /opt/conda/include/*cudnn* /opt/conda/lib/*cudnn*
ENV CPATH=/opt/conda/include:$CPATH \
    LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH \
    LIBRARY_PATH=/opt/conda/lib:$LIBRARY_PATH \
    PATH=/opt/conda/bin:$PATH \
    CONDA_PREFIX=/opt/conda \
    NCCL_LAUNCH_MODE=PARALLEL

ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python'

RUN conda install -y cmake=3.19.6 pip rmm=0.18 cudatoolkit=11.0

RUN pip3 install numpy==1.19.2 pandas sklearn ortools nvtx-plugins jupyter tensorflow==2.4.0 && \
    pip3 cache purge

RUN mkdir -p /var/tmp && cd /var/tmp && source activate ${CONDA_ENV} && git clone https://github.com/NVIDIA/HugeCTR.git HugeCTR && cd - && \
      cd /var/tmp/HugeCTR && if [[ "$RELEASE" == "true" ]]; then git fetch --all --tags && git checkout tags/${HUGECTR_VER}; else git checkout master; fi && \
      git submodule update --init --recursive && \
      mkdir -p build && cd build && \
      cmake -DCMAKE_BUILD_TYPE=Release -DSM=$SM -DONLY_EMB_PLUGIN=ON .. && make -j$(nproc) && make install && \
      rm -rf /var/tmp/HugeCTR;

RUN conda config --set auto_activate_base false

RUN pip uninstall jupyterlab jupyter -y

SHELL ["/bin/bash", "-c"]


ARG CONDA_ENV=merlin
RUN source deactivate; conda create --name ${CONDA_ENV}
ARG RAPIDS_VER=0.18.0
RUN source activate ${CONDA_ENV}; conda install -c rapidsai -c nvidia -c numba -c conda-forge cudf=${RAPIDS_VER} rmm=${RAPIDS_VER} dask=2021.02.0 cmake cudatoolkit=11.0 pandas=1.1.5
RUN source activate ${CONDA_ENV}; conda install -c rapidsai -c nvidia -c numba -c conda-forge dask-cudf=${RAPIDS_VER} dask-cuda=${RAPIDS_VER} cudnn nvtx 
RUN source activate ${CONDA_ENV}; git clone https://github.com/NVIDIA/NVTabular.git /nvtabular/; cd /nvtabular/; if [[ "$RELEASE" == "true" ]] ; then git fetch --all --tags && git checkout tags/${NVTAB_VER}; else git checkout main; fi; pip install -e .;
RUN source activate ${CONDA_ENV}; pip install pynvml pytest graphviz sklearn scipy matplotlib 
RUN source activate ${CONDA_ENV}; pip install nvidia-pyindex; pip install tritonclient[all] grpcio-channelz
RUN source activate ${CONDA_ENV}; conda install -c rapidsai asvdb

RUN source activate ${CONDA_ENV}; conda env config vars set PYTHONPATH=$PYTHONPATH:/opt/conda/envs/merlin/lib/python3.8/site-packages:/hugectr/tools/embedding_plugin/python:/opt/conda/lib/python3.8/site-packages:/usr/local/lib/python3.8/dist-packages
RUN source activate ${CONDA_ENV}; apt update; apt install -y graphviz ;
RUN source activate ${CONDA_ENV}; conda clean --all -y
RUN echo $(du -h --max-depth=1 /)

ENV LD_LIBRARY_PATH=/usr/local/hugectr/lib:$LD_LIBRARY_PATH \
    LIBRARY_PATH=/usr/local/hugectr/lib:$LIBRARY_PATH \
    PYTHONPATH=/usr/local/hugectr/lib:$PYTHONPATH
    
HEALTHCHECK NONE
ENTRYPOINT []
CMD ["/bin/bash"]

