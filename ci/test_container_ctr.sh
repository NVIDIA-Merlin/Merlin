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

if [ $container != 'merlin-ci-runner' ]; then
    ${ci_script_dir}container_size.sh $container $devices
fi

${ci_script_dir}container_software.sh $container $devices

echo "##############"
echo "# Unit tests #"
echo "##############"
exit_code=0
## Test HugeCTR
if [ "$container" == "merlin-hugectr" ]; then
    echo "Run unit tests for HugeCTR"
    /hugectr/ci/test_unit.sh $container $devices || exit_code=1
fi

## Test SOK
if [ "$container" == "merlin-tensorflow" ]; then
    echo "Run unit tests for merlin-sok"
    /hugectr/ci/test_unit.sh $container $devices || exit_code=1
fi

exit $exit_code
