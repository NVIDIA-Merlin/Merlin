#!/bin/bash

hugectr=$1
tf4rec=$2

# Test NVTabular
pytest /nvtabular/tests/unit

# Test HugeCTR
if [ "$hugectr" == "train" ]; then
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
elif [ "$hugectr" == "embedding" ]; then
    embedding_test
elif [ "$hugectr" == "inference" ]; then
    inference_test
fi

# Test Transformers4Rec
if [ "$tf4rec" == "true" ]; then
    pytest /transformers4rec/tests/unit
fi

