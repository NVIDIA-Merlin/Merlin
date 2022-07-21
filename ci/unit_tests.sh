#!/bin/bash

container=$1
devices=$2

echo "##############"
echo "# Unit tests #"
echo "##############"

exit_code=0

## Test Core
echo "Run unit tests for Core"
cd /core && ci/test_unit.sh $container $devices || exit_code=1

## Test NVTabular
echo "Run unit tests for NVTabular"
cd /nvtabular && ci/test_unit.sh $container $devices || exit_code=1

## Test Transformers4Rec
echo "Run unit tests for Transformers4Rec"
cd /transformers4rec/ && ci/test_unit.sh $container $devices || exit_code=1

## Test Models
echo "Run unit tests for Models"
pip install coverage || exit_code=1
cd /models/ && ci/test_unit.sh $container $devices || exit_code=1

## Test Systems
echo "Run unit tests for Systems"
cd /systems && pytest -rxs tests/unit || exit_code=1

## Test HugeCTR
if [ "$container" == "merlin-hugectr" ]; then
    echo "Run unit tests for HugeCTR"
    /hugectr/ci/test_unit.sh $container $devices || exit_code=1
fi

## Test distributed-embeddings
if [ "$container" == "merlin-tensorflow" ]; then
    echo "Run unit tests for merlin-sok"
    /hugectr/ci/test_unit.sh $container $devices || exit_code=1

    echo "Run unit tests for distributed-embeddings"
    pytest -rxs /distributed_embeddings/tests || exit_code=1
fi

## Test Merlin
echo "Run unit tests for Merlin"
cd /Merlin && pytest -rxs tests/unit || exit_code=1

exit $exit_code
