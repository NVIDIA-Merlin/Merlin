#!/usr/bin/env bash

docker compose run --rm requirements sh -c "pip-compile --output-file=$1.txt --resolver=backtracking $1.in"
