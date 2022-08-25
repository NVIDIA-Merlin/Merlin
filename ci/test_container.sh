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

${ci_script_dir}container_software.sh $container $devices
${ci_script_dir}container_unit.sh $container $devices
${ci_script_dir}container_integration.sh $container $devices $suppress_failures
