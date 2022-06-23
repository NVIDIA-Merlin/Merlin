## What is Merlin for Recommender Systems?

NVIDIA Merlin is a framework for accelerating the entire recommender systems pipeline on the GPU: from data ingestion and training to deployment. Merlin empowers data scientists, machine learning engineers, and researchers to build high-performing recommenders at scale. Merlin includes tools that democratize building deep learning recommenders by addressing common ETL, training, and inference challenges.  Each stage of the Merlin pipeline offers an easy-to-use API and is optimized to support hundreds of terabytes of data.

The Merlin PyTorch container allows users to do preprocessing and feature engineering with NVTabular, and then train a deep-learning based recommender system model with PyTorch, and serve the trained model on Triton Inference Server.

## About the Merlin PyTorch Container

The Merlin PyTorch container includes the following key components to simplify developing and deploying your recommender system:

* NVTabular performs data preprocessing and feature engineering for tabular data. The library can operate on small and large datasets--scaling to manipulate terabyte-scale datasets that are used to train deep learning recommender systems.

* Triton Inference Server to provide GPU-accelerated inference. Triton Inference Server simplifies the deployment of AI models at scale in production. The server is an open source inference serving software that enables teams to deploy trained AI models from any framework: TensorFlow, TensorRT, PyTorch, ONNX Runtime, or a custom framework. The server can serve models from local storage or Google Cloud Platform or AWS S3 on any GPU- or CPU-based infrastructure (cloud, data center, or edge). The NVTabular ETL workflow and trained deep learning models (TensorFlow or HugeCTR) can be deployed easily with only a few steps to production.

## Getting Started

### Launch the Merlin PyTorch Container

You can launch the Merlin PyTorch container with the following command:

```
docker run --gpus all  --rm -it -p 8888:8888 -p 8797:8787 -p 8796:8786 --ipc=host --cap-add SYS_NICE nvcr.io/nvidia/merlin/merlin-pytorch:latest /bin/bash
```

If you have a Docker version less than 19.03, change `--gpus all` to `--runtime=nvidia`.

The container will open a shell when the run command completes execution, you will be responsible for starting the jupyter lab on the docker container. Should look similar to below:

Start the jupyter-lab server:

```
cd / ; jupyter-lab --allow-root --ip='0.0.0.0' --NotebookApp.token=''
```

Now you can use any browser to access the jupyter-lab server, via :8888
Once in the server, navigate to the /nvtabular/ directory and explore the code base or try out some of the examples.

Within the container is the codebase, along with all of our dependencies, particularly RAPIDS Dask-cuDF. The easiest way to get started is to simply launch the container above and explore the examples within.

### Other NVIDIA Merlin containers

Merlin containers are available in the NVIDIA container repository at the following locations:
Table 1: Merlin Containers

| Container name | Container location | Functionality |
|----------------|--------------------|---------------|
| merlin-hugectr | https://ngc.nvidia.com/catalog/containers/nvidia:merlin:merlin-hugectr | Merlin and HugeCTR |
| merlin-pytorch | https://ngc.nvidia.com/catalog/containers/nvidia:merlin:merlin-pytorch | Merlin and PyTorch |
| merlin-tensorflow | https://ngc.nvidia.com/catalog/containers/nvidia:merlin:merlin-tensorflow | Merlin and TensorFlow |

## Examples and Tutorials

We provide a collection of examples, use cases, and tutorials for [NVTabular](https://github.com/NVIDIA/NVTabular/tree/main/examples) and [HugeCTR](https://github.com/NVIDIA/HugeCTR/tree/master/notebooks) as Jupyter notebooks in our repository. These Jupyter notebooks are based on the following datasets:
- MovieLens
- Outbrain Click Prediction
- Criteo Click Ads Prediction
- RecSys2020 Competition Hosted by Twitter
- Rossmann Sales Prediction
  With the example notebooks we cover the following:
- Preprocessing and feature engineering with NVTabular
- Advanced workflows with NVTabular
- Accelerated dataloaders for TensorFlow and PyTorch
- Scaling to multi-GPU and multi nodes systems
- Integrating NVTabular with HugeCTR
- Deploying to inference with Triton

## Learn More

* [NVTabular Documentation](https://nvidia-merlin.github.io/NVTabular/main/Introduction.html)
* [HugeCTR Documentation](https://nvidia-merlin.github.io/HugeCTR/master/hugectr_user_guide.html)
* [NVTabular](https://github.com/nvidia-merlin/nvtabular)
* [HugeCTR](https://github.com/nvidia-merlin/hugectr)
* [Triton Inference Server](https://github.com/triton-inference-server/server)
* [NVIDIA Developer Site](https://developer.nvidia.com/nvidia-merlin#getstarted)
* [NVIDIA Developer Blog](https://medium.com/nvidia-merlin)

## License

By pulling and using the container, you accept the terms and conditions of this [End User License Agreement.](https://developer.download.nvidia.com/licenses/NVIDIA_Deep_Learning_Container_License.pdf)
