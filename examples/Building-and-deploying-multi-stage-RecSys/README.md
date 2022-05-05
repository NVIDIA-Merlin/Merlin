# NVIDIA Merlin Example Notebooks

We created two jupyter notebook examples based on different stages of a Recommender Systems to demonstrate how to deploy a multi-stage Recommender Systems on Triton Inference Server. The notebooks demonstrate how to use NVTabular, Merlin Models and Merlin Systems libraries for ETL, training, and inference, respectively.

## Structure

Two example notebooks are structured as follows:
- 01-Building-Recommender-Systems-with-Merlin: 
    - Execute the preprocessing and feature engineering pipeline (ETL) with NVTabular on the GPU
    - train a ranking and retrieval model with TensorFlow based on the ETL output
    - export saved models, user and item features and item embeddings.
- 02-Deploying-multi-stage-RecSys-with-Merlin-Systems: 
    - set up Feast feature store for feature storing and Faiss index for similarity search
    - build multi-stage recommender systems ensemble pipeline with Merlin Systems operators
    - perform Inference with the Triton Inference Server using Merlin Systems library.

## Running the Example Notebooks

Merlin docker containers are available on http://ngc.nvidia.com/catalog/containers/ with pre-installed versions. For `Building-and-deploying-multi-stage-RecSys` example notebooks we used `merlin-tensorflow-inference` container that has NVTabular with TensorFlow and Triton Inference support.

To run the example notebooks using Docker containers, do the following:

1. Once you pull the inference container, launch it by running the following command:
   ```
   docker run -it --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 -p 8888:8888 -v <path to your data>:/workspace/data/ --ipc=host <docker container> /bin/bash
   ```
The container will open a shell when the run command execution is completed.

1. You will have to start JupyterLab on the Docker container. First, install jupyter-lab with the following command if it is missing:
   ```
   pip install jupyterlab
   ```
   
   For more information, see [Installation Guide](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html).

2. Start the jupyter-lab server by running the following command:
   ```
   jupyter-lab --allow-root --ip='0.0.0.0' --NotebookApp.token='<password>'
   ```

3. Open any browser to access the jupyter-lab server using `localhost:8888`.

4. Once in the server, navigate to the ```/Merlin/examples/Building-and-deploying-multi-stage-RecSys/``` directory and execute the example notebooks.
