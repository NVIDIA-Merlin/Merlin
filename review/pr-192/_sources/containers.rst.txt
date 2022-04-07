Merlin Containers
=================

Merlin and the Merlin component libraries are available in Docker containers from the NVIDIA GPU Cloud (NCG) catalog.
Access the catalog of containers at http://ngc.nvidia.com/catalog/containers.

Training Containers
--------------------

The following table identifies the container names, catalog URL, and key Merlin components.

.. list-table::
   :widths: 25 50 25
   :header-rows: 1

   * - Training Container
     - NGC Catalog URL
     - Key Merlin Components
   * - merlin-training
     - https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-training
     - NVTabular and HugeCTR
   * - merlin-tensorflow-training
     - https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-tensorflow-training
     - NVTabular, TensorFlow, and HugeCTR Tensorflow Embedding plugin
   * - merlin-pytorch-training
     - https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-pytorch-training
     - NVTabular and PyTorch

To use these containers, you must install the `NVIDIA Container Toolkit <https://github.com/NVIDIA/nvidia-docker>`_ to provide GPU support for Docker.
You can use the NGC links referenced in the preceding table for more information about how to launch and run these containers.


Inference Containers
--------------------

The following table identifies the container names, catalog URL, and key Merlin components.

.. list-table::
   :widths: 25 50 25
   :header-rows: 1

   * - Inference Container
     - NGC Catalog URL
     - Key Merlin Components
   * - merlin-inference
     - https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-inference
     - NVTabular, HugeCTR, and Triton Inference Server
   * - merlin-tensorflow-inference
     - https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-tensorflow-inference
     - NVTabular, Tensorflow, and Triton Inference Server
   * - merlin-pytorch-inference
     - https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-pytorch-inference
     - NVTabular, PyTorch, and Triton Inference Server

To use these containers, you must install the `NVIDIA Container Toolkit <https://github.com/NVIDIA/nvidia-docker>`_ to provide GPU support for Docker.
You can use the NGC links referenced in the preceding table for more information about how to launch and run these containers.
