# Merlin Systems Example Notebook

These Jupyter notebooks demonstrate how to use Merlin Systems to deploy ranking models on [Triton Inference Server](https://github.com/triton-inference-server/server). Currently we support models built with TensorFlow framework, and traditional-ml models like XGBoost and python-based models with implicit datasets. Examples built with PyTorch framework are being developed and will be added here soon. 

## Running the Example Notebooks

Docker containers are available from the NVIDIA GPU Cloud.
We use the latest stable version of the [merlin-tensorflow](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-tensorflow/tags) container to run the example notebooks. To run the example notebooks using Docker containers, perform the following steps:


1. Pull and start the container by running the following command:

   ```shell
   docker run --gpus all --rm -it \
     -p 8888:8888 -p 8797:8787 -p 8796:8786 --ipc=host \
     nvcr.io/nvidia/merlin/merlin-tensorflow:23.XX /bin/bash
   ```

   > You can find the release tags and more information on the [merlin-tensorflow](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-tensorflow) container page.

   The container opens a shell when the run command execution is completed.
   Your shell prompt should look similar to the following example:

   ```shell
   root@2efa5b50b909:
   ```

2. Start the JupyterLab server by running the following command:

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

3. Open a browser and use the `127.0.0.1` URL provided in the messages by JupyterLab.

4. After you log in to JupyterLab, navigate to the `/Merlin/examples/ranking` directory to try out the example notebooks.
