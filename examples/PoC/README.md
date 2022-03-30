# NVIDIA Merlin Example Notebooks

We created a jupyter notebook example based on different stages of a Recommender Systems to provide an end-to-end example. The notebook demonstrate how to use NVTabular, Merlin Models and Merlin Systems for ETL, training, and inference, respectively.

## Structure

Each example notebook is structured as follows:
- execute the preprocessing and feature engineering pipeline (ETL) with NVTabular on the GPU
- train a model with TensorFlow based on the ETL output
- perform Inference with the Triton Inference Server using Merlin Systems library.


## Running the Example Notebooks

You can run the example notebooks by [installing NVTabular](https://github.com/NVIDIA/NVTabular#installation) and other required libraries. Alternatively, Docker conatiners are available on http://ngc.nvidia.com/catalog/containers/ with pre-installed versions. For the PoC example notebooks we used `merlin-tensorflow-inference:nightly` container.

- merlin-tensorflow-inference contains NVTabular with TensorFlow and Triton Inference support

To run the example notebooks using Docker containers, do the following:

1. Pull the container by running the following command:
   ```
   docker run --runtime=nvidia --rm -it -p 8888:8888 -p 8797:8787 -p 8796:8786 --ipc=host --cap-add SYS_PTRACE <docker container> /bin/bash
   ```

   **NOTE**: If you are running on Docker version 19 and higher, change ```--runtime=nvidia``` to ```--gpus all```.

   The container will open a shell when the run command execution is completed. You will have to start JupyterLab on the Docker container. It should look similar to this:
   ```
    root@f9b2754d5741:/opt/tritonserver# 
   ```
   
2. Install Tensorflow

    ```
    pip install tensorflow-gpu
    ```

3. Install feast and faiss libraries

    ```
    pip install feast
    pip install faiss-gpu
    ```

4. Install jupyter-lab and matplotlib by running the following command:
   ```
   pip install jupyterlab
   pip install matplotlib
   ```
   
   For more information, see [Installation Guide](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html).

5. Start the jupyter-lab server by running the following command:
   ```
   jupyter-lab --allow-root --ip='0.0.0.0' --NotebookApp.token='<password>'
   ```

4. Open any browser to access the jupyter-lab server using <MachineIP>:8888.

5. Once in the server, navigate to the ```/Merlin/examples``` directory and try out the example notebooks.
