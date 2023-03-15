#!/bin/bash

container=$1
devices=$2

echo "##################"
echo "# Container size #"
echo "##################"

cd / && NUMGB=$(du -sh --exclude "raid" 2> /dev/null | grep -oE '[0-9]*G' | grep -oE '[0-9]*') 
echo "Size of container is: $NUMGB GB"
if [ $NUMGB -ge 16  ]; then echo "Size of container exceeds 16GB, failed build." && exit 1 ; fi;
