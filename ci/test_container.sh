#!/bin/bash
set -e

container=$1

# Test NVTabular - All containers
pytest /nvtabular/tests/unit

# Test HugeCTR - Training container
if [ "$container" == "merlin-training" ]; then
    layers_test && \
    checker_test && \
    data_reader_test && \
    device_map_test && \
    loss_test && \
    optimizer_test && \
    regularizers_test && \
    model_oversubscriber_test && \
    parser_test && \
    auc_test
# Test Transformers4Rec - Tensorflow container
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
