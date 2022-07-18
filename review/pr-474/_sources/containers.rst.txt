Merlin Containers
=================

Merlin and the Merlin component libraries are available in Docker containers from the NVIDIA GPU Cloud (NCG) catalog.
Access the catalog of containers at http://ngc.nvidia.com/catalog/containers.

The following table identifies the container names, catalog URL, and key Merlin components.

.. list-table::
   :header-rows: 1

   * - Container Name
     - NGC Catalog URL
     - Key Merlin Components
   * - merlin-hugectr
     - https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-hugectr
     - Merlin libraries, including HugeCTR
   * - merlin-tensorflow
     - https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-tensorflow
     - Merlin libraries, TensorFlow, and HugeCTR Tensorflow Embedding plugin
   * - merlin-pytorch
     - https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-pytorch
     - Merlin libraries and PyTorch

To use these containers, you must install the `NVIDIA Container Toolkit <https://github.com/NVIDIA/nvidia-docker>`_ to provide GPU support for Docker.
You can use the NGC links referenced in the preceding table for more information about how to launch and run these containers.


Structural Changes Beginning with the 22.06 Releases
----------------------------------------------------

With the 22.06 release of the Merlin containers, each container can perform model training as well as inference.
Before the 22.06 release, the NGC catalog included one container for training and a separate container for inference.
