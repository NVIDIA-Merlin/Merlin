# NVIDIA Merlin Dockerfiles and Containers

All NVIDIA Merlin components are available as open source projects. However, a more convenient way to make use of these components is by using our Merlin NGC containers. We have created Docker containers for NVIDIA Merlin that are hosted on [NGC](https://ngc.nvidia.com/catalog/containers/). 

Containers allow you to package your software application, libraries, dependencies, and runtime compilers in a self-contained environment. These containers can be pulled and launched right out of the box. You can clone and adjust these containers if necessary. 

The following table provides a list of Dockerfiles that you can use to build the corresponding Docker container:

| Container Name       | Dockerfile         | Container Location                                                                     | Functionality                                                  |
|----------------------|--------------------|----------------------------------------------------------------------------------------|----------------------------------------------------------------|
| `merlin-hugectr`     | `dockerfile.ctr`   | <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-hugectr>    | NVTabular and HugeCTR                                          |
| `merlin-tensorflow`  | `dockerfile.tf`    | <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-tensorflow> | NVTabular, TensorFlow, and HugeCTR Tensorflow Embedding plugin |
| `merlin-pytorch`     | `dockerfile.torch` | <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-pytorch>    | NVTabular and PyTorch                                          |


# Building these containers locally

Building our containers is a two-step process. We first build the Merlin `BASE_IMAGE` using the `dockerfile.merlin` file. This container depends on two upstream containers: `nvcr.io/nvidia/tritonserver` and `nvcr.io/nvidia/tensorflow`, from which it pulls the necessary dependencies for Triton Inference Server and RAPIDS tools. It also builds and installs other Merlin requirements, such as scikit-learn, XGBoost, etc.

At the time of this writing, the two-stage build process takes roughly 1 hour. Running all of the tests for all Merlin libraries can take a couple of additional hours, depending on which framework you're building.

## Building the `BASE_IMAGE`

We tag this image as `nvcr.io/nvstaging/merlin/merlin-base:${MERLIN_VERSION}` and it is used to create the framework-specific containers. There are `ARG`s in the Dockerfile to define which version of the containers to use. You can override the defaults when building the image like below.

```bash
docker build . -f dockerfile.merlin -t nvcr.io/mycompany/merlin-base:${MERLIN_VERSION} --build-arg DLFW_IMAGE=22.12
```

In this example we are tagging the base image as `nvcr.io/mycompany/merlin-base:${MERLIN_VERSION}`. The tag Merlin uses when building this image in our own build pipeline is `nvcr.io/nvstaging/merlin/merlin-base:${MERLIN_VERSION}`. _Note that this intermediate image is not made available publicly, but the framework-specific containers based on it are._

## Building a framework-specific container

We also provide Dockerfiles for creating framework-specific containers: `dockerfile.tf`, `dockerfile.torch`, and `dockerfile.ctr`. These are all based on the `BASE_IMAGE` created in the previous step and install the associciated deep learning frameworks.

To build the Pytorch container, we will specify the `BASE_IMAGE` build arg to use the base image we just created.

```bash
docker build . -f dockerfile.torch -t ngcr.io/mycompany/merlin-torch:${MERLIN_VERSION} --build-arg BASE_IMAGE=nvcr.io/mycompany/merlin-base:${MERLIN_VERSION}
```

## Default Arguments

Each of the Dockerfiles have many `ARG`s defined, most of which have defaults set. Sometimes the defaults fall out of date, because the Merlin team overrides them in our build process as demonstrated above. To see the latest versions used in each of our containers, see the [Merlin Support Matrix](https://nvidia-merlin.github.io/Merlin/main/support_matrix/index.html)
