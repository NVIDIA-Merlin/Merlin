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
elif [ "$container" == "merlin-inference" ]; then
    /usr/local/hugectr/bin/inference_test
fi

# Test Transformers4Rec
if [ "$container" != "merlin-training" ]; then
    sh -c 'pytest /transformers4rec/tests; ret=$?; [ $ret = 5 ] && exit 0 || exit $ret'
fi
