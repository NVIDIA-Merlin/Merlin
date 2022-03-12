#!/bin/bash

container=$1
devices=$2

##############
# Unit tests #
##############

## Test Core
/core/ci/test_unit.sh $container $devices

## Test NVTabular
/nvtabular/ci/test_unit.sh $container $devices

if [ "$container" != "merlin-training" ]; then
  ## Test Transformers4Rec
  /transformers4rec/ci/test_unit.sh $container $devices

  ## Test Models
  pip install coverage
  /models/ci/test_unit.sh $container $devices
fi

## Test HugeCTR
if [ "$container" == "merlin-training" ]; then
    /hugectr/ci/test_unit.sh $container $devices
fi

#####################
# Integration tests #
#####################

# Test NVTabular 
## Not shared storage in blossom yet
#regex="merlin(.)*-inference"
#if [[ ! "$container" =~ $regex ]]; then
#    /nvtabular/ci/test_integration.sh $container $devices --report 1
#fi

# Test Transformers4Rec
if [ "$container" != "merlin-training" ]; then
    /transformers4rec/ci/test_integration.sh $container $devices
fi
