# NVIDIA Merlin Examples

We created a collection of jupyter notebooks based on different datasets to provide end-to-end examples for NVIDIA Merlin. The examples cover how to use NVTabular in combination with TensorFlow, PyTorch and [HugeCTR](https://github.com/NVIDIA/HugeCTR). Each example explains an end-to-end workflow from ETL to Training to Inference.

## Structure

Each example is structured in multiple notebooks:
- 01-Download-Convert.ipynb: Instruction to download the dataset and convert it into the correct format to consume it in the next notebooks
- 02-ETL-with-NVTabular.ipynb: Execute preprocessing and feature engineering pipeline (ETL) with **NVTabular** on GPU
- 03a-Training-with-TF.ipynb: Training a model with **TensorFlow** based on the ETL output
- 03b-Training-with-PyTorch.ipynb: Training a model with **PyTorch** based on the ETL output
- 03c-Training-with-HugeCTR.ipynb: Training a model with **HugeCTR** based on the ETL output
- 03d-Training-with-FastAI.ipynb: Training a model with **FastAI** based on the ETL output
- 04: Inference with Triton Inference server (depending on which deep learning framework)
- 05: containing a range of additional notebooks for one dataset.

## Examples

### 1. [Getting Started with MovieLens](https://github.com/NVIDIA-Merlin/Merlin/tree/main/examples/getting-started-movielens)

The MovieLens25M is a popular dataset for recommender systems and is used in academic publications. Most users are familiar with the dataset and we will teach the **basic concepts of NVTabular**:
- Learning **NVTabular** for using GPU-accelerated ETL (Preprocess and Feature Engineering)
- Getting familiar with **NVTabular's high-level API**
- Using single-hot/multi-hot categorical input features with **NVTabular**
- Using **NVTabular dataloader** with TensorFlow Keras model
- Using **NVTabular dataloader** with PyTorch

### 2. [Scaling to large Datasets with Criteo](https://github.com/NVIDIA-Merlin/Merlin/tree/main/examples/scaling-criteo)

[Criteo](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/) provides the largest publicly available dataset for recommender systems, having a size of 1TB uncompressed click logs of 4 billion examples. We will teach to **scale NVTabular**:
- Using **multiple GPUs and multiple nodes** with NVTabular for ETL
- Training recommender system model with NVTabular dataloader for PyTorch
- Training recommender system model with NVTabular dataloader for TensorFlow
- Training recommender system model with HugeCTR using multi GPU
- Inference with Triton Inference Server for TensorFlow or HugeCTR

## Start Examples

You can run the examples by [installing NVTabular](https://github.com/NVIDIA/NVTabular#installation) and other required libraries. Alternativly, docker conatiners are available on http://ngc.nvidia.com/catalog/containers/ with pre-installed versions. Depending which example you want to run, you should select the docker container:
- Merlin-Tensorflow-Training contains NVTabular with TensorFlow
- Merlin-Pytorch-Training contains NVTabular with PyTorch
- Merlin-Training contains NVTabular with HugeCTR
- Merlin-Inference contains NVTabular with TensorFlow and Triton Inference support

There are example docker-compose files in [Scaling to large Datasets with Criteo](https://github.com/NVIDIA-Merlin/Merlin/tree/main/examples/scaling-criteo)


### Start Examples with Docker Container

You can pull the container by running the following command.
```
docker run --runtime=nvidia --rm -it -p 8888:8888 -p 8797:8787 -p 8796:8786 --ipc=host --cap-add SYS_PTRACE <docker container> /bin/bash
```

**NOTE**: If you are running on Docker version 19 and higher, change ```--runtime=nvidia``` to ```--gpus all```.

The container will open a shell when the run command execution is completed. You'll have to start jupyter lab on the Docker container. It should look similar to this:
```
root@2efa5b50b909:
```

1. Activate the ```rapids``` conda environment by running the following command:
   ```
   root@2efa5b50b909: source activate rapids
   ```

   You should receive the following response, indicating that the environment has been activated:
   ```
   (rapids)root@2efa5b50b909:
   ```
2. Install jupyter-lab with `conda` or `pip`: [Installation Guide](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html)

   ```
   pip install jupyterlab
   ```

3. Start the jupyter-lab server by running the following command:
   ```
   jupyter-lab --allow-root --ip='0.0.0.0' --NotebookApp.token='<password>'
   ```

4. Open any browser to access the jupyter-lab server using <MachineIP>:8888.

5. Once in the server, navigate to the ```/nvtabular/``` directory and try out the examples.


