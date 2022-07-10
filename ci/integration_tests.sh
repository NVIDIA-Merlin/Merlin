#!/bin/bash

container=$1
devices=$2

echo "#####################"
echo "# Integration tests #"
echo "#####################"

exit_code=0

# Test NVTabular 
## Not shared storage in blossom yet, inference testing cannot be run
echo "Run integration tests for NVTabular"
/nvtabular/ci/test_integration.sh $container $devices --report 1 || exit_code=1

# Test Transformers4Rec
echo "Run integration tests for Transformers4Rec"
/transformers4rec/ci/test_integration.sh $container $devices || exit_code=1

exit $exit_code