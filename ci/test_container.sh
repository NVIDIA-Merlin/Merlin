#!/bin/bash
set -e

container=$1

# Test NVTabular
pytest /nvtabular/tests/unit

# Test HugeCTR
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
elif [ "$container" == "merlin-tensorflow-training" ]; then
    embedding_test
elif [ "$container" == "merlin-inference" ]; then
    inference_test
fi

# Test Transformers4Rec
if [ "$container" != "merlin-training" ]; then
    pytest /transformers4rec/tests
fi

