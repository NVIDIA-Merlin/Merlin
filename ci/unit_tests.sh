#!/bin/bash

container=$1
devices=$2

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
