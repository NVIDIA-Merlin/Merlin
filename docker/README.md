# NVIDIA Merlin Dockerfiles and Containers

All NVIDIA Merlin components are available as open source projects. However, a more convenient way to make use of these components is by using our Merlin NGC containers. We have created Docker containers for NVIDIA Merlin that are hosted on [NGC](https://ngc.nvidia.com/catalog/containers/). 

Containers allow you to package your software application, libraries, dependencies, and runtime compilers in a self-contained environment. These containers can be pulled and launched right out of the box. You can clone and adjust these containers if necessary. 

The following table provides a list of Dockerfiles that you can use to build the corresponding Docker container:

| Container Name       | Dockerfile         | Container Location                                                                     | Functionality                                                  |
|----------------------|--------------------|----------------------------------------------------------------------------------------|----------------------------------------------------------------|
| `merlin-hugectr`     | `dockerfile.ctr`   | <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-hugectr>    | NVTabular and HugeCTR                                          |
| `merlin-tensorflow`  | `dockerfile.tf`    | <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-tensorflow> | NVTabular, TensorFlow, and HugeCTR Tensorflow Embedding plugin |
| `merlin-pytorch`     | `dockerfile.torch` | <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-pytorch>    | NVTabular and PyTorch                                          |
