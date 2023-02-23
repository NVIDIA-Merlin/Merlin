#!/bin/bash

container=$1
devices=$2

exit_code=0

echo "##################"
echo "# Container size #"
echo "##################"

cd / && NUMGB=$(du -sh --exclude "raid" 2> /dev/null | grep -oE '[0-9]*G' | grep -oE '[0-9]*') 
echo "Size of container is: $NUMGB GB"
if [ $NUMGB -ge 15  ]; then echo "Size of container exceeds 15GB, failed build." && exit 1 ; fi;


echo "##################"
echo "# Software check #"
echo "##################"

regex="merlin-(.)*"
if [[ ! "$container" =~ $regex ]]; then
    echo "Check tritonserver for all Merlin containers"
    which tritonserver || exit_code=1
fi

if [ "$container" == "merlin-hugectr" ]; then
    echo "Check HugeCTR for ctr-training container"
    python -c "import hugectr; print(hugectr.__version__)" || exit_code=1

    # TODO: remove this block once
    # https://github.com/NVIDIA-Merlin/HugeCTR/pull/328
    # is in the hugectr release
    cd /hugectr && \
    checker_test && \
    device_map_test && \
    loss_test && \
    optimizer_test && \
    regularizers_test || exit_code=1
fi

if [ "$container" == "merlin-tensorflow" ]; then
    echo "Check TensorFlow for merlin-tensorflow container"
    python -c "import tensorflow; print(tensorflow.__version__)" || exit_code=1
    echo "Check merlin-sok for tf-training container"
    python -c "import sparse_operation_kit; print(sparse_operation_kit.__version__)" || exit_code=1
    echo "Check distributed-embeddings for tf-training container"
    python -c "import distributed_embeddings as tfde; print(tfde.__doc__)" || exit_code=1

    # TODO: remove this block once
    # https://github.com/NVIDIA-Merlin/HugeCTR/pull/328
    # is in the hugectr release
    pushd /hugectr/sparse_operation_kit/unit_test/test_scripts/tf2 && \
    bash sok_test_unit.sh && \
    popd || exit_code=1
fi

if [ "$container" == "merlin-pytorch" ]; then
    echo "Check PyTorch for torch-training container"
    python -c "import torch; print(torch.__version__)" || exit_code=1
fi

exit $exit_code
