#!/bin/bash

container=$1
devices=$2
suppress_failures="${3:-1}"

echo "#####################"
echo "# Integration tests #"
echo "#####################"

exit_code=0

# Test NVTabular 
## Not shared storage in blossom yet, inference testing cannot be run
echo "Run integration tests for NVTabular"
/nvtabular/ci/test_integration.sh $container $devices --report 1 || exit_code=1

# Test Merlin Models
echo "Run integration tests for Merlin Models"
/models/ci/test_integration.sh $container $devices || exit_code=1

# Test Transformers4Rec
echo "Run integration tests for Transformers4Rec"
/transformers4rec/ci/test_integration.sh $container $devices || exit_code=1

## Test Merlin
echo "Run integration tests for Merlin"
/Merlin/ci/test_integration.sh $container $devices || exit_code=1

if [[ "$suppress_failures" -eq 0 ]]
then
    exit 0
else
    exit $exit_code
fi
