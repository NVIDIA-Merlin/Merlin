# NVIDIA Merlin Example Notebooks

We have a collection of Jupyter example notebooks that are based on different datasets to provide end-to-end examples for NVIDIA Merlin.
These example notebooks demonstrate how to use NVTabular with TensorFlow, PyTorch, and [HugeCTR](https://github.com/NVIDIA/HugeCTR).
Each example provides additional details about the end-to-end workflow, such as includes ETL, training, and inference.

## Inventory

### [Building and Deploying a multi-stage RecSys](./Building-and-deploying-multi-stage-RecSys)

Recommender system pipelines are often based on multiple stages: Retrieval, Filtering, Scoring and Ordering.
This example provides an end-to-end pipeline that leverages the Merlin framework:

- Processing the dataset using NVTabular.
- Training a scoring model using Merlin Models.
- Training a retrieval model using Merlin Models.
- Building a feature store with Feast and ANN index with Fiass.
- Deploying an end-to-end pipeline of retrieval, scoring, and ANN search to Triton Inference Server.

### [Getting Started with MovieLens](./getting-started-movielens)

The MovieLens25M is a popular dataset for recommender systems and is used in academic publications.
Many users are familiar with this dataset, so the notebooks focus primarily on the basic concepts of NVTabular:

- Learning NVTabular to GPU-accelerate ETL (Preprocess and Feature Engineering).
- Getting familiar with NVTabular's high-level API.
- Using single-hot and multi-hot categorical input features with NVTabular.
- Using the NVTabular dataloader with the TensorFlow Keras model.
- Using the NVTabular dataloader with PyTorch.

### [Scaling Large Datasets with Criteo](./scaling-criteo)

[Criteo](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/) provides the largest publicly available dataset for recommender systems with a size of 1TB of uncompressed click logs that contain 4 billion examples.
These notebooks demonstrate how to scale NVTabular:

- Use multiple GPUs and nodes with NVTabular for ETL.
- Train recommender system models with the NVTabular dataloader for PyTorch.
- Train recommender system models with the NVTabular dataloader for TensorFlow.
- Train recommender system models with HugeCTR using a multi-GPU.
- Inference with the Triton Inference Server and TensorFlow or HugeCTR.

## Running the Example Notebooks

You can run the examples with Docker containers.
Docker containers are available from the NVIDIA GPU Cloud catalog.
Access the catalog of containers at <http://ngc.nvidia.com/catalog/containers>.

Depending on which example you want to run, you should use any one of these Docker containers:

- [`merlin-hugectr`](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-hugectr) (contains Merlin Core, Merlin Models, Merlin Systems, NVTabular, HugeCTR)
- [`merlin-tensorflow`](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-tensorflow) (contains Merlin Core, Merlin Models, Merlin Systems, NVTabular and TensorFlow)
- [`merlin-pytorch`](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-pytorch) (contains Merlin Core, Merlin Models, Merlin Systems, NVTabular and PyTorch)

All the containers include Triton Inference Server and are capable of training models and performing inference.

There are example docker-compose files referenced in [Scaling to large Datasets with Criteo](https://github.com/NVIDIA-Merlin/Merlin/tree/main/examples/scaling-criteo) that you can get from our GitHub repository.

To run the example notebooks using Docker containers, perform the following steps:

1. Pull and start the container by running the following command:

   ```shell
   docker run --gpus all --rm -it \
     -p 8888:8888 -p 8797:8787 -p 8796:8786 --ipc=host \
     <docker container> /bin/bash
   ```

   The container opens a shell when the run command execution is completed.
   Your shell prompt should look similar to the following example:

   ```shell
   root@2efa5b50b909:
   ```

1. Install JupyterLab with `pip` by running the following command:

   ```shell
   pip install jupyterlab
   ```

   For more information, see the JupyterLab [Installation Guide](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html).

1. Start the JupyterLab server by running the following command:

   ```shell
   jupyter-lab --allow-root --ip='0.0.0.0'
   ```

   View the messages in your terminal to identify the URL for JupyterLab.
   The messages in your terminal show similar lines to the following example:

   ```shell
   Or copy and paste one of these URLs:
   http://2efa5b50b909:8888/lab?token=9b537d1fda9e4e9cadc673ba2a472e247deee69a6229ff8d
   or http://127.0.0.1:8888/lab?token=9b537d1fda9e4e9cadc673ba2a472e247deee69a6229ff8d
   ```

1. Open a browser and use the `127.0.0.1` URL provided in the messages by JupyterLab.

1. After you log in to JupyterLab, navigate to the `/Merlin/examples` directory to try out the example notebooks.
