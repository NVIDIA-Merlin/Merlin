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
As the ETL component of the Merlin ecosystem, NVTabular is a feature engineering and preprocessing library for tabular data. It is designed to quickly and easily manipulate terabyte scale datasets that are used to train deep learning based recommender systems. The high-level API is easy to define complex data transformation workflows. We observed 100x-1000x speed-ups in comparison to the same transformation on optimized CPU clusters. Users can
- prepare datasets quickly and easily for experimentation so that more models can be trained
- process datasets that exceed GPU and CPU memory without having to worry about scale
- focus on what to do with the data and not how to do it by using abstraction at the operation level

**[NVTabular dataloaders](https://github.com/NVIDIA/NVTabular):**
NVTabular provides seamless integration into common deep learning frameworks, such as TensorFlow, PyTorch and HugeCTR. When training deep learning recommender system models, data loading can be a bottleneck. We developed custom, highlu-optimized dataloader to accelerate existing TensorFlow or PyTorch training pipelines. Replacing only the dataloader shows a 9x speed-ups in comparison to the same training pipeline with GPU. Users can
- remove bottlenecks from dataloading by processing large chunks of data at a time instead of item by item
- process datasets that don’t fit within the GPU or CPU memory by streaming from the disk
- prepare batch asynchronously into the GPU to avoid CPU-GPU communication
- integrate easily into existing TensorFlow or PyTorch training pipelines by using a similar API

**[HugeCTR](https://github.com/NVIDIA/HugeCTR):**
HugeCTR is a custom deep learning framework dedicated for recommendation systems written in CUDA C++. HugeCTR contains the same optimized dataloader for preparing batches with GPU-acceleration. In addition, HugeCTR scales to large deep learning recommendation models. The neural network architectures contain often large embedding tables to represent hunders of millions of users or items. These embedding tables can easily exceed the CPU/GPU memory. HugeCTR provides strategies to scale large embedding tables beyond available memory. Users can
- Scale embedding tables over multiple GPUs or multi nodes
- Proficiency in oversubscribing models to train embedding tables with single nodes that don’t fit within the GPU or CPU memory (only required embeddings are prefetched from a parameter server per batch).
- Asynchronous and multithreaded data pipelines.
- A highly optimized data loader.

**[Triton](https://github.com/triton-inference-server/server):**
NVTabular and HugeCTR both support the Triton Inference Server to provide GPU-accelerated inference. The Triton Inference Server simplifies the deployment of AI models to production at scale. It is an inference serving software that is open source and lets teams deploy trained AI models from any framework. Users can:
- deploy NVTabular ETL workflows and trained deep learning models to production in a few lines of code
- deploy an ensemble of NVTabular ETL and trained deep learning model to ensure that same data transformation are applied in production
- deplypy models concurrently on GPUs maximizing utilization
- enable low latency real time inferencing or batch inferencing to maximize GPU/CPU utilization 
- scale production environemtn with Kubernetes for orchestration, metrics, and auto-scaling (as docker container)

### More Resources
Check out our [end-to-end examples](./examples/) for NVIDIA Merlin. You can find more information about NVIDIA Merlin and its components on:
- [NVTabular GitHub](https://github.com/NVIDIA/NVTabular)
- [HugeCTR GitHub](https://github.com/NVIDIA/HugeCTR)
- [NVTabular API Documentation](https://nvidia.github.io/NVTabular/main/Introduction.html)
- [HugeCTR UserGuide](https://github.com/NVIDIA/HugeCTR/blob/master/docs/hugectr_user_guide.md)
- [HugeCTR Python API](https://github.com/NVIDIA/HugeCTR/blob/master/docs/python_interface.md)
- [NVTabular Accelerated Training Documentation](https://nvidia.github.io/NVTabular/main/training/index.html)
