#!/bin/bash

container=$1
devices=$2

echo "##############"
echo "# Unit tests #"
echo "##############"

exit_code=0

## Test HugeCTR
if [ "$container" == "merlin-hugectr" ]; then
    echo "Run unit tests for HugeCTR"
    /hugectr/ci/test_unit.sh $container $devices || exit_code=1
    echo "Run unit tests for merlin-sok"
    /hugectr/ci/test_unit.sh "merlin-tensorflow" $devices || exit_code=1
fi

exit $exit_code
