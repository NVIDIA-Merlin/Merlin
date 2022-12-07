# Deploying a Multi-Stage Recommender System

We created two Jupyter notebooks that demonstrate two different stages of recommender systems.
The notebooks show how to deploy a multi-stage recommender system and serve recommendations with Triton Inference Server.
The notebooks demonstrate how to use the NVTabular, Merlin Models, and Merlin Systems libraries for feature engineering, training, and then inference.

The two example notebooks are structured as follows:

- [Building the Recommender System](01-Building-Recommender-Systems-with-Merlin.ipynb):
  - Execute the preprocessing and feature engineering pipeline (ETL) with NVTabular on the GPU/CPU.
  - Train a ranking and retrieval model with TensorFlow based on the ETL output.
  - Export the saved models, user and item features, and item embeddings.

- [Deploying the Recommender System with Triton](02-Deploying-multi-stage-RecSys-with-Merlin-Systems.ipynb):
  - Set up a Feast feature store for feature storing and a Faiss index for similarity search.
  - Build a multi-stage recommender system ensemble pipeline with Merlin Systems operators.
  - Perform inference with the Triton Inference Server using the Merlin Systems library.

## Running the Example Notebooks

Containers with the Merlin libraries are available from the NVIDIA NGC catalog.
To run the sample notebooks, use the `merlin-tensorflow` container.

You can pull and run the `nvcr.io/nvidia/merlin/merlin-tensorflow:nightly` container.

> In production, instead of using the `nightly` tag, specify a release tag.
> You can find the release tags and more information on the [Merlin TensorFlow](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-tensorflow) container page.

To run the example notebooks using a container, do the following:

1. After you pull the container, launch it by running the following command:

   ```shell
   docker run -it --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 -p 8888:8888 \
     -v <path to your data>:/workspace/data/ --ipc=host \
     nvcr.io/nvidia/merlin/merlin-tensorflow:nightly /bin/bash
   ```

   You can remove the `--gpus all` flag to run the example on CPU.

   The container opens a shell when the run command execution is complete.
   Your shell prompt should look similar to the following example:

   ```text
   root@2efa5b50b909:
   ```

1. Start JupyterLab by running the following command:

   ```shell
   jupyter-lab --allow-root --ip='0.0.0.0' --NotebookApp.token='<password>'
   ```

   View the messages in your terminal to identify the URL for JupyterLab.
   The messages in your terminal should show lines like the following example:

   ```text
   Or copy and paste one of these URLs:
   http://2efa5b50b909:8888/lab?token=9b537d1fda9e4e9cadc673ba2a472e247deee69a6229ff8d
   or http://127.0.0.1:8888/lab?token=9b537d1fda9e4e9cadc673ba2a472e247deee69a6229ff8d
   ```

1. Open a browser and use the `127.0.0.1` URL provided in the messages from JupyterLab.

1. After you log in to JupyterLab, navigate to the ```/Merlin/examples/Building-and-deploying-multi-stage-RecSys/``` directory and execute the example notebooks.
