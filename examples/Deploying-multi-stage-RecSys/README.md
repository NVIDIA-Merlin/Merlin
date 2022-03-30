# NVIDIA Merlin Example Notebooks

We created two jupyter notebook examples based on different stages of a Recommender Systems to demonstrate how to deploy a Recommender Systems. The notebook demonstrate how to use NVTabular, Merlin Models and Merlin Systems libraries for ETL, training, and inference, respectively.

## Structure

Each example notebook is structured as follows:
- execute the preprocessing and feature engineering pipeline (ETL) with NVTabular on the GPU
- train a ranking and retrieval model with TensorFlow based on the ETL output
- perform Inference with the Triton Inference Server using Merlin Systems library.


## Running the Example Notebooks

You can run the example notebooks by [installing NVTabular](https://github.com/NVIDIA/NVTabular#installation) and other required libraries. Alternatively, Docker conatiners are available on http://ngc.nvidia.com/catalog/containers/ with pre-installed versions. For the PoC example notebooks we used `merlin-tensorflow-inference:nightly` container.

- merlin-tensorflow-inference contains NVTabular with TensorFlow and Triton Inference support

To run the example notebooks using Docker containers, do the following:

1. Once you pull the inference container, launch it by running the following command:
   ```
   docker run -it --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 -p 8888:8888 -v <path to your data>:/workspace/data/ --ipc=host <docker container> /bin/bash
   ```

   **NOTE**: If you are running on Docker version 19 and higher, change ```--runtime=nvidia``` to ``````.

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

4. Open any browser to access the jupyter-lab server using localhost:8888.

5. Once in the server, navigate to the ```/Merlin/examples``` directory and execute the example notebooks.