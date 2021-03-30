## [NVIDIA Merlin](https://github.com/NVIDIA-Merlin) 

[NVIDIA Merlin](https://developer.nvidia.com/nvidia-merlin) is an open source library designed to accelerate recommender systems on NVIDIA GPUs. It enables data scientists, machine learning engineers, and researchers to build high-performing recommenders at scale. Merlin includes tools to address common ETL, training, and inference challenges. Each stage of the Merlin pipeline is optimized to support hundreds of terabytes of data, which is all accessible through easy-to-use APIs. With Merlin, better predictions and increased click-through rates are within reach.

### Benefits

NVIDIA Merlin is an open source library dedicated to accelerate (deep learning) recommender system pipelines with GPUs. Merlin enables its users to
* GPU-accelerating data transformation (ETL) for preprocessing and engineering features, which scales beyond larger than memory datasets sizes
* Accelerating existing training pipelines in TensorFlow, PyTorch or FastAI by leveraging custom-built, optimized dataloaders
* Scaling large deep learning recommender models by enabling larger than memory embedding tables
* Deploying data transformation and trained models to production with only few lines of code

The goal is to provide a scalable, accelerated and easy-to-use solution to build recommender systems end-to-end. 

### Examples

We provide a collection of [end-to-end examples](./examples/) in this repository as Jupyter notebooks. The examples cover
- Download and prepare the dataset
- Preprocessing and engineering features
- Training deep learning recommendation models with TensorFlow, PyTorch, FastAI or HugeCTR
- Deploying the models to production

The exampels are based on different datasets to provide a wide range of real-world use cases.

### Components of NVIDIA Merlin

NVIDIA Merlin is a collection of open source libraries: [NVTabular](https://github.com/NVIDIA/NVTabular), [HugeCTR](https://github.com/NVIDIA/HugeCTR) and [Triton Inference Server](https://github.com/triton-inference-server/server)

<p align="center">
<img src='https://developer.nvidia.com/sites/default/files/akamai/merlin/recommender-systems-dev-web-850.svg' width="65%">
</p>

**[NVTabular](https://github.com/NVIDIA/NVTabular):**

