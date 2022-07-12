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

${ci_script_dir}software_check.sh $container $devices
${ci_script_dir}unit_tests.sh $container $devices
${ci_script_dir}integration_tests.sh $container $devices $suppress_failures
