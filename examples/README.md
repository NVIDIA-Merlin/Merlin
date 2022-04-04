# NVIDIA Merlin Example Notebooks

We have created a collection of Jupyter example notebooks based on different datasets to provide end-to-end examples for NVIDIA Merlin. These example notebooks demonstrate how to use NVTabular with TensorFlow, PyTorch, and [HugeCTR](https://github.com/NVIDIA/HugeCTR). Each example provides additional details about the end-to-end workflow, which includes ETL, Training, and Inference.

## Inventory

### 1. [Deploying multi stage RecSys](./Deploying-multi-stage-RecSys/)

Recommender systems pipelines are often based on 4 stages: Retrieval, Filtering, Scoring and Ordering. This example leverages the Merlin Framework to train and deploy the different stages:
- Process the data with NVTabular
- Training a retrieal model with Merlin Models
- Training a scoring model with Merlin Models
- Deploying the stages to Triton Inference Server with Merlin Systems

### 2. [Getting Started with MovieLens](https://github.com/NVIDIA-Merlin/Merlin/tree/main/examples/getting-started-movielens)

The MovieLens25M is a popular dataset for recommender systems and is used in academic publications. Most users are familiar with this dataset, so we're focusing primarily on the basic concepts of NVTabular, which includes:
- Learning NVTabular to GPU-accelerate ETL (Preprocess and Feature Engineering)
- Getting familiar with NVTabular's high-level API
- Using single-hot/multi-hot categorical input features with NVTabular
- Using the NVTabular dataloader with the TensorFlow Keras model
- Using the NVTabular dataloader with PyTorch

### 3. [Scaling Large Datasets with Criteo](https://github.com/NVIDIA-Merlin/Merlin/tree/main/examples/scaling-criteo)

[Criteo](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/) provides the largest publicly available dataset for recommender systems with a size of 1TB of uncompressed click logs that contain 4 billion examples. We demonstrate how to scale NVTabular, as well as:
- Use multiple GPUs and nodes with NVTabular for ETL
- Train recommender system models with the NVTabular dataloader for PyTorch
- Train recommender system models with the NVTabular dataloader for TensorFlow
- Train recommender system models with HugeCTR using a multi-GPU
- Inference with the Triton Inference Server and TensorFlow or HugeCTR

## Running the Example Notebooks

You can run the example notebooks by [installing NVTabular](https://github.com/NVIDIA/NVTabular#installation) and other required libraries. Alternatively, Docker containers are available on http://ngc.nvidia.com/catalog/containers/ with pre-installed versions. Depending on which example you want to run, you should use any one of these Docker containers:
- Merlin-Tensorflow-Training (contains NVTabular with TensorFlow)
- Merlin-Pytorch-Training (contains NVTabular with PyTorch)
- Merlin-Training (contains NVTabular with HugeCTR)
- Merlin-Inference (contains NVTabular with TensorFlow and Triton Inference support)

There are example docker-compose files referenced in [Scaling to large Datasets with Criteo](https://github.com/NVIDIA-Merlin/Merlin/tree/main/examples/scaling-criteo).

To run the example notebooks using Docker containers, do the following:

1. Pull the container by running the following command:
   ```
   docker run --gpus all --rm -it -p 8888:8888 -p 8797:8787 -p 8796:8786 -p 8000:8000 -p 8001:8001 -p 8002:8002 --ipc=host --cap-add SYS_PTRACE <docker container> /bin/bash
   ```

   The container will open a shell when the run command execution is completed. You will have to start JupyterLab on the Docker container. It should look similar to this:
   ```
   root@2efa5b50b909:
   ```
   
2. Some containers require to install jupyter-lab with `conda` or `pip` by running the following command:
   ```
   pip install jupyterlab
   ```
   
   For more information, see [Installation Guide](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html).

3. Start the jupyter-lab server by running the following command:
   ```
   jupyter-lab --allow-root --ip='0.0.0.0' --NotebookApp.token='<password>' --notebook-dir=/
   ```

4. Open any browser to access the jupyter-lab server using <MachineIP>:8888.

5. Once in the server, navigate to the ```/Merlin/examples``` directory and try out the example notebooks.
