#!/bin/bash
set -e

container=$1

##############
# Unit tests #
##############

## Test NVTabular
pytest /nvtabular/tests/unit

## Test HugeCTR
### Training container
if [ "$container" == "merlin-training" ]; then
    # layers_test && \ Running oom in blossom
    checker_test && \
    # data_reader_test && \ Need Multi-GPU
    device_map_test && \
    loss_test && \
    optimizer_test && \
    regularizers_test # && \
    # parser_test && \ Needs Multi-GPU
    # auc_test Needs Multi-GPU
### Inference container
# elif [ "$container" == "merlin-inference" ]; then
    # HugeCTR - Deactivated until it is self-contained and it runs
    # inference_test
# fi

## Test Transformers4Rec
/transformers4rec/ci/test_unit.sh $container 0


#####################
# Integration tests #
#####################

## Test NVTabular 
# /nvtabular/ci/test_integration.sh $container 0

## Test HugeCTR
# Waiting to sync integration tests with them

## Test Transformers4Rec
/transformers4rec/ci/test_integration.sh $container 0
