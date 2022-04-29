# Deploying a Multi-Stage Recommender System

We created two Jupyter notebooks that demonstrate two different stages of a Recommender Systems.
The goal of the notebooks is to show how to deploy a multi-stage Recommender System and serve recommendations with Triton Inference Server.
The notebooks demonstrate how to use the NVTabular, Merlin Models, and Merlin Systems libraries for feature engineering, training, and then inference.

The two example notebooks are structured as follows:

- [Building the Recommender System](01-Building-Recommender-Systems-with-Merlin.ipynb): 
  - Execute the preprocessing and feature engineering pipeline (ETL) with NVTabular on the GPU.
  - Train a ranking and retrieval model with TensorFlow based on the ETL output.
  - Export the saved models, user and item features, and item embeddings.

- [Deploying the Recommender System with Triton](02-Deploying-multi-stage-RecSys-with-Merlin-Systems.ipynb): 
  - Set up a Feast feature store for feature storing and a Faiss index for similarity search.
  - Build a multi-stage recommender system ensemble pipeline with Merlin Systems operators.
  - Perform inference with the Triton Inference Server using the Merlin Systems library.

## Running the Example Notebooks

Merlin docker containers are available on http://ngc.nvidia.com/catalog/containers/ with pre-installed versions. For `
Deploying-multi-stage-RecSys` example notebooks we used `merlin-tensorflow-inference` container that has NVTabular with TensorFlow and Triton Inference support.

To run the example notebooks using Docker containers, do the following:

1. Once you pull the inference container, launch it by running the following command:
   ```
   docker run -it --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 -p 8888:8888 -v <path to your data>:/workspace/data/ --ipc=host <docker container> /bin/bash
   ```
The container will open a shell when the run command execution is completed.
  
2. Install Tensorflow

    ```
    pip install tensorflow-gpu
    ```

3. Install feast and faiss libraries

    ```
    pip install feast
    pip install faiss-gpu
    ```

4. You will have to start JupyterLab on the Docker container. First, install jupyter-lab with the following command:
   ```
   pip install jupyterlab
   ```
   
   For more information, see [Installation Guide](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html).

5. Start the jupyter-lab server by running the following command:
   ```
   jupyter-lab --allow-root --ip='0.0.0.0' --NotebookApp.token='<password>'
   ```

4. Open any browser to access the jupyter-lab server using `localhost:8888`.

5. Once in the server, navigate to the ```/Merlin/examples/Deploying-multi-stage-RecSys``` directory and execute the example notebooks.
