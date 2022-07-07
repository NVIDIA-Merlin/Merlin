#!/bin/bash
set -e

container=$1
devices=$2
suppress_failures="${3:-1}"

# Use the dirname directly, without changing directories
if [[ $BASH_SOURCE = */* ]]; then
    ci_script_dir=${BASH_SOURCE%/*}/
else
    ci_script_dir=./
fi

<<<<<<< HEAD
${ci_script_dir}software_check.sh $container $devices
${ci_script_dir}unit_tests.sh $container $devices
${ci_script_dir}integration_tests.sh $container $devices $suppress_failures
=======
echo "#####################"
echo "# Integration tests #"
echo "#####################"

# Test NVTabular 
## Not shared storage in blossom yet, inference testing cannot be run
echo "Run integration tests for NVTabular"
/nvtabular/ci/test_integration.sh $container $devices --report 1

# Test Transformers4Rec
echo "Run integration tests for Merlin Models"
/models/ci/test_integration.sh $container $devices

# Test Transformers4Rec
echo "Run integration tests for Transformers4Rec"
/transformers4rec/ci/test_integration.sh $container $devices
>>>>>>> Added integration tests to Merlin Models
