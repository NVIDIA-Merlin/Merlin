# NVIDIA Merlin Dockerfiles

We built and host dockerfiles for NVIDIA Merlin on [NGC](https://ngc.nvidia.com/catalog/containers/). The containers can be pulled and started out of the box. In this repository, we provide the Dockerfiles used for the builts. Userse can clone and adjust the containers, if necessary.

| Container name             | Dockerfile       | Container location                                                             | Functionality                                         |
|----------------------------|------------------|--------------------------------------------------------------------------------|-------------------------------------------------------|
| Merlin-training            | dockerfile.ctr   |  https://ngc.nvidia.com/containers/nvstaging:merlin:merlin-training            | NVTabular and HugeCTR                                 |
| Merlin-tensorflow-training | dockerfile.tf    |  https://ngc.nvidia.com/containers/nvstaging:merlin:merlin-tensorflow-training | NVTabular, TensorFlow and Tensorflow Embedding plugin |
| Merlin-pytorch-training    | dockerfile.torch |  https://ngc.nvidia.com/containers/nvstaging:merlin:merlin-pytorch-training    | NVTabular and PyTorch                                 |
| Merlin-inference           | dockerfile.tri   |  https://ngc.nvidia.com/containers/nvstaging:merlin:merlin-inference           | NVTabular, HugeCTR and Triton Inference               |