# syntax=docker/dockerfile:1.2
ARG TRITON_VERSION=22.02
ARG IMAGE=nvcr.io/nvidia/tritonserver:${TRITON_VERSION}-tf2-python-py3
FROM ${IMAGE}

# Args
ARG CORE_VER=main
ARG RMM_VER=v21.12.00
ARG CUDF_VER=v21.12.02
ARG NVTAB_VER=main
ARG NVTAB_BACKEND_VER=main
ARG MODELS_VER=main
ARG HUGECTR_VER=master
ARG HUGECTR_BACKEND_VER=main
ARG TF4REC_VER=main
#
# Envs
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/lib:/repos/dist/lib
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_PATH=$CUDA_HOME
ENV CUDA_CUDA_LIBRARY=${CUDA_HOME}/lib64/stubs
ENV PATH=${CUDA_HOME}/lib64/:${PATH}:${CUDA_HOME}/bin
ENV PYTHONPATH=/usr/lib/python3.8/site-packages:$PYTHONPATH

# Install packages
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update -y --fix-missing && \
    apt install -y --no-install-recommends \
        clang-format \
        libboost-serialization-dev \
	libexpat1-dev \
	libsasl2-2 \
        libssl-dev \
        libtbb-dev \
	policykit-1 \
        protobuf-compiler \
        rapidjson-dev \
	software-properties-common \
        zlib1g-dev && \
    apt autoremove -y && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

# Install multiple packages
RUN pip install cupy-cuda115 nvidia-pyindex pybind11 pytest protobuf transformers==4.12 tensorflow-metadata 
RUN pip install betterproto cachetools graphviz nvtx scipy sklearn
RUN pip install numba --no-deps
RUN pip install tritonclient[all] grpcio-channelz
RUN pip install dask==2021.11.2 distributed==2021.11.2 dask[dataframe]==2021.11.2 dask-cuda
RUN pip install git+https://github.com/rapidsai/asvdb.git@main

# Install cmake
RUN apt remove --purge cmake -y && wget http://www.cmake.org/files/v3.21/cmake-3.21.1.tar.gz && \
    tar xf cmake-3.21.1.tar.gz && cd cmake-3.21.1 && ./configure && make && make install

# Install spdlog
RUN git clone --branch v1.9.2 https://github.com/gabime/spdlog.git build-env && \
    pushd build-env && \
      mkdir build && cd build && cmake .. && make -j && make install && \
    popd && \
    rm -rf build-env
    
# Install arrow
ENV ARROW_HOME=/usr/local
RUN git clone --branch apache-arrow-5.0.0 --recurse-submodules https://github.com/apache/arrow.git build-env && \
    pushd build-env && \
      export PARQUET_TEST_DATA="${PWD}/cpp/submodules/parquet-testing/data" && \
      export ARROW_TEST_DATA="${PWD}/testing/data" && \
      pip install -r python/requirements-build.txt && \
      mkdir cpp/release && \
      pushd cpp/release && \
        cmake -DCMAKE_INSTALL_PREFIX=${ARROW_HOME} \
              -DCMAKE_INSTALL_LIBDIR=lib \
              -DCMAKE_LIBRARY_PATH=${CUDA_CUDA_LIBRARY} \
              -DARROW_FLIGHT=ON \
              -DARROW_GANDIVA=OFF \
              -DARROW_ORC=ON \
              -DARROW_WITH_BZ2=ON \
              -DARROW_WITH_ZLIB=ON \
              -DARROW_WITH_ZSTD=ON \
              -DARROW_WITH_LZ4=ON \
              -DARROW_WITH_SNAPPY=ON \
              -DARROW_WITH_BROTLI=ON \
              -DARROW_PARQUET=ON \
              -DARROW_PYTHON=ON \
              -DARROW_PLASMA=ON \
              -DARROW_BUILD_TESTS=ON \
              -DARROW_CUDA=ON \
              -DARROW_DATASET=ON \
              -DARROW_HDFS=ON \
              -DARROW_S3=ON \ 
              .. && \
        make -j$(nproc) && \
        make install && \
      popd && \
      pushd python && \
        export PYARROW_WITH_PARQUET=ON && \
        export PYARROW_WITH_CUDA=ON && \
        export PYARROW_WITH_ORC=ON && \
        export PYARROW_WITH_DATASET=ON && \
        export PYARROW_WITH_S3=ON && \
        export PYARROW_WITH_HDFS=ON && \
        python setup.py build_ext --build-type=release bdist_wheel && \
        pip install dist/*.whl --no-deps --force-reinstall && \
      popd && \
    popd && \
    rm -rf build-env

# Install rmm
ENV INSTALL_PREFIX=/usr
RUN git clone https://github.com/rapidsai/rmm.git build-env && cd build-env/ && \
    git checkout ${RMM_VER} && \
    cd ..; \
    pushd build-env && \
    ./build.sh librmm && \
    pip install python/. --no-deps && \
    popd && \
    rm -rf build-env

# Install CUDF
RUN git clone https://github.com/rapidsai/cudf.git build-env && cd build-env/ && \
    git checkout ${CUDF_VER} && \
    git submodule update --init --recursive && \
    cd .. && \
    pushd build-env && \
      export CUDF_HOME=${PWD} && \
      export CUDF_ROOT=${PWD}/cpp/build/ && \
      export CMAKE_LIBRARY_PATH=${CUDA_CUDA_LIBRARY} && \
      export CUDAFLAGS=-Wno-error=unknown-pragmas && \
      ./build.sh libcudf cudf dask_cudf --allgpuarch --cmake-args=\"-DCUDF_ENABLE_ARROW_S3=OFF\" && \
    popd && \
    rm -rf build-env

# Install Merlin Core
RUN git clone https://github.com/NVIDIA-Merlin/core.git /core/ && \
    cd /core/ && git checkout ${CORE_VER} && pip install -e . --no-deps
ENV PYTHONPATH=/core:$PYTHONPATH

# Install NVTabular
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python'
RUN git clone https://github.com/NVIDIA-Merlin/NVTabular.git /nvtabular/ && \
    cd /nvtabular/ && git checkout ${NVTAB_VER} && pip install -e . --no-deps
ENV PYTHONPATH=/nvtabular:$PYTHONPATH

# Install Transformers4Rec
RUN pip install transformers==4.12
RUN git clone https://github.com/NVIDIA-Merlin/Transformers4Rec.git /transformers4rec && \
    cd /transformers4rec/ && git checkout ${TF4REC_VER} && pip install -e . --no-deps
ENV PYTHONPATH=/transformers4rec:$PYTHONPATH

# Install Models
RUN git clone https://github.com/NVIDIA-Merlin/Models.git /models/ && \
    cd /models/ && git checkout ${MODELS_VER} && pip install -e . --no-deps;
ENV PYTHONPATH=/models:$PYTHONPATH

# Install NVTabular Triton Backend
ARG TRITON_VERSION
RUN git clone https://github.com/NVIDIA-Merlin/nvtabular_triton_backend.git build-env && \
    cd build-env && git checkout ${NVTAB_BACKEND_VER} && cd .. && \
    pushd build-env && \
      mkdir build && \
      cd build && \
      cmake -Dpybind11_DIR=/usr/local/lib/python3.8/dist-packages/pybind11/share/cmake/pybind11 \
        -D TRITON_COMMON_REPO_TAG="r$TRITON_VERSION"    \
        -D TRITON_CORE_REPO_TAG="r$TRITON_VERSION"      \
        -D TRITON_BACKEND_REPO_TAG="r$TRITON_VERSION" .. \
      && make -j && \
      mkdir /opt/tritonserver/backends/nvtabular && \
      cp libtriton_nvtabular.so /opt/tritonserver/backends/nvtabular/ && \
    popd && \
    rm -rf build-env 

# Clean up
RUN rm -rf /repos

HEALTHCHECK NONE
CMD ["/bin/bash"]
