#!/bin/bash

container=$1
devices=$2

echo "##################"
echo "# Container size #"
echo "##################"

cd / && NUMGB=$(du -sh 2> /dev/null | grep -oE '[0-9]*G' | grep -oE '[0-9]*') 
if [ $NUMGB -ge 15  ]; then echo "Size of container exceeds 15GB, failed build." && exit 1 ; fi;
