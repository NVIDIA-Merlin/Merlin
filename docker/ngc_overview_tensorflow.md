## What is Merlin for Recommender Systems?

NVIDIA Merlin is a framework for accelerating the entire recommender systems pipeline on the GPU: from data ingestion and training to deployment. Merlin empowers data scientists, machine learning engineers, and researchers to build high-performing recommenders at scale. Merlin includes tools that democratize building deep learning recommenders by addressing common ETL, training, and inference challenges. Each stage of the Merlin pipeline is optimized to support hundreds of terabytes of data, all accessible through easy-to-use APIs. With Merlin, better predictions than traditional methods and increased click-through rates are within reach.

The Merlin ecosystem has four main components: Merlin ETL, Merlin Dataloaders and Training, and Merlin Inference.

## Merlin TensorFlow

The merlin-tensorflow container allows users to do preprocessing and feature engineering with NVTabular, and then train a deep-learning based recommender system model with TensorFlow, and serve the trained model on Triton Inference Server.

As the ETL component of the Merlin ecosystem, NVTabular is a feature engineering and preprocessing library for tabular data designed to quickly and easily manipulate terabyte scale datasets used to train deep learning based recommender systems. The core features are explained in the API documentation and additional information can be found in the GitHub repository.

Dataloading is a bottleneck in training deep learning recommender systems models. NVIDIA Merlin accelerates training deep learning recommender systems in two ways: 1) Customized dataloaders speed-up existing TensorFlow training pipelines or 2) using HugeCTR, a dedicated framework written in CUDA C++. This container provides the environment to use NVTabular dataloaders for TensorFlow to accelerate deep learning training for existing TensorFlow pipelines.

NVTabular and HugeCTR supports Triton Inference Server to provide GPU-accelerated inference. Triton Inference Server simplifies the deployment of AI models at scale in production. It is an open source inference serving software that lets teams deploy trained AI models from any framework (TensorFlow, TensorRT, PyTorch, ONNX Runtime, or a custom framework), from local storage or Google Cloud Platform or AWS S3 on any GPU- or CPU-based infrastructure (cloud, data center, or edge). The NVTabular ETL workflow and trained deep learning models (TensorFlow or HugeCTR) can be deployed easily with only a few steps to production. Both NVTabular and HugeCTR provide end-to-end examples for deployment: NVTabular examples and HugeCTR examples.

## Getting Started

### Launch Merlin TensorFlow Container

You can pull the training containers with the following command:

```
docker run --runtime=nvidia --rm -it -p 8888:8888 -p 8797:8787 -p 8796:8786 --ipc=host --cap-add SYS_NICE nvcr.io/nvidia/merlin/merlin-tensorflow:latest /bin/bash
```

If you are running on a docker version 19+, change --runtime=nvidia to --gpus all.
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
| merlin-hugectr | https://ngc.nvidia.com/catalog/containers/nvidia:merlin:merlin-tensorflow | Merlin and TensorFlow |

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

For more sample models and their end-to-end instructions for HugeCTR visit the link: https://github.com/NVIDIA/HugeCTR/tree/master/samples

## Learn More

If you are interested in learning more about how NVTabular works under the hood, we have API documentation that outlines in detail the specifics of the calls available within the library.
The following are the suggested readings for those who want to learn more about HugeCTR.

HugeCTR User Guide: https://github.com/NVIDIA/HugeCTR/blob/master/docs/hugectr_user_guide.md

Questions and Answers: https://github.com/NVIDIA/HugeCTR/blob/master/docs/QAList.md

Sample models and their end-to-end instructions: https://github.com/NVIDIA/HugeCTR/tree/master/samples

NVIDIA Developer Site: https://developer.nvidia.com/nvidia-merlin#getstarted

NVIDIA Developer Blog: https://medium.com/nvidia-merlin

## Contributing

If you wish to contribute to the Merlin library directly please see [Contributing.md](https://github.com/NVIDIA/NVTabular/blob/main/CONTRIBUTING.md). We are particularly interested in contributions or feature requests for feature engineering or preprocessing operations that you have found helpful in your own workflows.

## License

By pulling and using the container, you accept the terms and conditions of this [End User License Agreement.](https://developer.download.nvidia.com/licenses/NVIDIA_Deep_Learning_Container_License.pdf)
