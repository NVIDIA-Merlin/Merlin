#!/bin/bash
set -e

container=$1

##############
# Unit tests #
##############

## Test NVTabular - All containers
pytest /nvtabular/tests/unit

## Test HugeCTR - Training container
if [ "$container" == "merlin-training" ]; then
    # Running oom in blossom
    # layers_test && \
    checker_test && \
    data_reader_test && \
    device_map_test && \
    loss_test && \
    optimizer_test && \
    regularizers_test && \
    model_oversubscriber_test && \
    parser_test && \
    auc_test
## Test Transformers4Rec - Tensorflow container
elif [ "$container" == "merlin-tensorflow-training" ]; then
    pytest /transformers4rec/tests/tf
# Test Transformers4Rec - Pytorch container
elif [ "$container" == "merlin-pytorch-training" ]; then
    pytest /transformers4rec/tests/torch
# Test HugeCTR & Transformers4Rec - Inference container
elif [ "$container" == "merlin-inference" ]; then
    # HugeCTR - Deactivated until it is self-contained and it runs
    # inference_test
    # Transformers4Rec
    pytest /transformers4rec/tests
fi

#####################
# Integration tests #
#####################

## Test NVTabular 
# /nvtabular/ci/test_integration.sh $container 0

## Test HugeCTR
# Waiting to sync integration tests with them

## Test Transformers4Rec
# Waiting for integration tests to be developed
