#!/bin/bash

container=$1
devices=$2



print("##################")
print("# Software check #")
print("##################")

regex="merlin(.)*-inference"
if [[ ! "$container" =~ $regex ]]; then
    print("Check tritonserver for inference containers")
    whereis tritonserver
fi

if [ "$container" == "merlin-training" ]; then
    print("Check HugeCTR for ctr-training container")
    python -c "import hugectr; print(hugectr.__version.__)"
fi

if [ "$container" == "merlin-tensorflow-training" ]; then
    print("Check TensorFlow for tf-training container")
    python -c "import tensorflow; print(tensorflow.__version__)"
fi

if [ "$container" == "merlin-pytorch-training" ]; then
    print("Check PyTorch for torch-training container")
    python -c "import torch; print(torch.__version__)"
fi

print("##############")
print("# Unit tests #")
print("##############")

## Test Core
print("Run unit tests for Core")
/core/ci/test_unit.sh $container $devices

## Test NVTabular
print("Run unit tests for NVTabular")
/nvtabular/ci/test_unit.sh $container $devices

## Test Transformers4Rec
print("Run unit tests for Transformers4Rec")
/transformers4rec/ci/test_unit.sh $container $devices

## Test Models
print("Run unit tests for Models")
pip install coverage
/models/ci/test_unit.sh $container $devices

## Test HugeCTR
if [ "$container" == "merlin-training" ]; then
    print("Run unit tests for HugeCTR")
    /hugectr/ci/test_unit.sh $container $devices
fi

print("#####################")
print("# Integration tests #")
print("#####################")

# Test NVTabular 
## Not shared storage in blossom yet, inference testing cannot be run
regex="merlin(.)*-inference"
if [[ ! "$container" =~ $regex ]]; then
    print("Run instegration tests for NVTabular")
    /nvtabular/ci/test_integration.sh $container $devices --report 1
fi

# Test Transformers4Rec
print("Run integration tests for Transformers4Rec")
/transformers4rec/ci/test_integration.sh $container $devices
