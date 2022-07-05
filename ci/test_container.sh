#!/bin/bash
set -e

container=$1
devices=$2



echo "##################"
echo "# Software check #"
echo "##################"

regex="merlin(.)*-inference"
if [[ ! "$container" =~ $regex ]]; then
    echo "Check tritonserver for inference containers"
    which tritonserver
fi

if [ "$container" == "merlin-hugectr" ]; then
    echo "Check HugeCTR for ctr-training container"
    python -c "import hugectr; print(hugectr.__version__)"

    # TODO: remove this block once
    # https://github.com/NVIDIA-Merlin/HugeCTR/pull/328
    # is in the hugectr release
    cd /hugectr && \
    checker_test && \
    device_map_test && \
    loss_test && \
    optimizer_test && \
    regularizers_test
fi

if [ "$container" == "merlin-tensorflow" ]; then
    echo "Check TensorFlow for merlin-tensorflow container"
    python -c "import tensorflow; print(tensorflow.__version__)"
    echo "Check merlin-sok for tf-training container"
    python -c "import sparse_operation_kit; print(sparse_operation_kit.__version__)"
    echo "Check distributed-embeddings for tf-training container"
    python -c "import distributed_embeddings as tfde; print(tfde.__doc__)"

    # TODO: remove this block once
    # https://github.com/NVIDIA-Merlin/HugeCTR/pull/328
    # is in the hugectr release
    pushd /hugectr/sparse_operation_kit/unit_test/test_scripts/tf2 && \
    bash sok_test_unit.sh && \
    popd
fi

if [ "$container" == "merlin-pytorch" ]; then
    echo "Check PyTorch for torch-training container"
    python -c "import torch; print(torch.__version__)"
fi

echo "##############"
echo "# Unit tests #"
echo "##############"

## Test Core
echo "Run unit tests for Core"
cd /core && ci/test_unit.sh $container $devices

## Test NVTabular
echo "Run unit tests for NVTabular"
cd /nvtabular && ci/test_unit.sh $container $devices

## Test Transformers4Rec
echo "Run unit tests for Transformers4Rec"
cd /transformers4rec/ && ci/test_unit.sh $container $devices

## Test Models
echo "Run unit tests for Models"
pip install coverage
cd /models/ && ci/test_unit.sh $container $devices

## Test Systems
echo "Run unit tests for Systems"
cd /systems && pytest -rxs tests/unit

## Test HugeCTR
if [ "$container" == "merlin-hugectr" ]; then
    echo "Run unit tests for HugeCTR"
    /hugectr/ci/test_unit.sh $container $devices
fi

## Test distributed-embeddings
if [ "$container" == "merlin-tensorflow" ]; then
    echo "Run unit tests for merlin-sok"
    /hugectr/ci/test_unit.sh $container $devices

    echo "Run unit tests for distributed-embeddings"
    pytest -rxs /distributed_embeddings/tests
fi

echo "#####################"
echo "# Integration tests #"
echo "#####################"

# Test NVTabular 
## Not shared storage in blossom yet, inference testing cannot be run
echo "Run integration tests for NVTabular"
/nvtabular/ci/test_integration.sh $container $devices --report 1

# Test Transformers4Rec
echo "Run integration tests for Transformers4Rec"
/transformers4rec/ci/test_integration.sh $container $devices
