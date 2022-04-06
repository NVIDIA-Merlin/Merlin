#!/bin/bash

container=$1
devices=$2



echo "##################"
echo "# Software check #"
echo "##################"

regex="merlin(.)*-inference"
if [[ ! "$container" =~ $regex ]]; then
    echo "Check tritonserver for inference containers"
    whereis tritonserver
fi

if [ "$container" == "merlin-training" ]; then
    echo "Check HugeCTR for ctr-training container"
    python -c "import hugectr; print(hugectr.__version.__)"
fi

if [ "$container" == "merlin-tensorflow-training" ]; then
    echo "Check TensorFlow for tf-training container"
    python -c "import tensorflow; print(tensorflow.__version__)"
fi

if [ "$container" == "merlin-pytorch-training" ]; then
    echo "Check PyTorch for torch-training container"
    python -c "import torch; print(torch.__version__)"
fi

echo "##############"
echo "# Unit tests #"
echo "##############"

## Test Core
echo "Run unit tests for Core"
/core/ci/test_unit.sh $container $devices

## Test NVTabular
echo "Run unit tests for NVTabular"
/nvtabular/ci/test_unit.sh $container $devices

## Test Transformers4Rec
echo "Run unit tests for Transformers4Rec"
/transformers4rec/ci/test_unit.sh $container $devices

## Test Models
echo "Run unit tests for Models"
pip install coverage
/models/ci/test_unit.sh $container $devices

## Test Systems
echo "Run unit tests for Systems"
pytest -rxs /systems/tests/unit

## Test HugeCTR
if [ "$container" == "merlin-training" ]; then
    echo "Run unit tests for HugeCTR"
    /hugectr/ci/test_unit.sh $container $devices
fi

echo "#####################"
echo "# Integration tests #"
echo "#####################"

# Test NVTabular 
## Not shared storage in blossom yet, inference testing cannot be run
regex="merlin(.)*-inference"
if [[ ! "$container" =~ $regex ]]; then
    echo "Run instegration tests for NVTabular"
    /nvtabular/ci/test_integration.sh $container $devices --report 1
fi

# Test Transformers4Rec
echo "Run integration tests for Transformers4Rec"
/transformers4rec/ci/test_integration.sh $container $devices
