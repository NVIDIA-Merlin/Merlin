## [NVIDIA Merlin](https://github.com/NVIDIA-Merlin) 

NVIDIA Merlin is an open source library designed to accelerate recommender systems on NVIDIA’s GPUs. It enables data scientists, machine learning engineers, and researchers to build high-performing recommenders at scale. Merlin includes tools to address common ETL, training, and inference challenges. Each stage of the Merlin pipeline is optimized to support hundreds of terabytes of data, which is all accessible through easy-to-use APIs. With Merlin, better predictions and increased click-through rates (CTRs) are within reach. For more information, see [NVIDIA Merlin](https://developer.nvidia.com/nvidia-merlin).

### Benefits

NVIDIA Merlin is a scalable and accelerated solution, making it easy to build recommender systems from end to end. With NVIDIA Merlin, you can:
* transform GPU-accelerated data (ETL) for preprocessing and engineering features, which scales beyond larger than memory datasets sizes.
* accelerate existing training pipelines in TensorFlow, PyTorch, or FastAI by leveraging optimized, custom-built data loaders.
* scale large deep learning recommender models by enabling larger than memory embedding tables.
* deploy data transformations and trained models to production with only a few lines of code.

The goal is to provide a scalable, accelerated and easy-to-use solution to build recommender systems end-to-end. 

### Components of NVIDIA Merlin

NVIDIA Merlin is a collection of open source libraries:
* NVTabular
* HugeCTR
* Triton Inference Server

<p align="center">
<img src='https://developer.nvidia.com/sites/default/files/akamai/merlin/recommender-systems-dev-web-850.svg' width="65%">
</p>

**[NVTabular](https://github.com/NVIDIA/NVTabular)**:<br>
NVTabular is a feature engineering and preprocessing library for tabular data. NVTabular is essentially the ETL component of the Merlin ecosystem. It is designed to quickly and easily manipulate terabyte-size datasets that are used to train deep learning based recommender systems. NVTabular offers a high-level API that can be used to define complex data transformation workflows. NVTabular is also capable of transformation speed-ups that can be 100 times to 1,000 times faster than transformations taking place on optimized CPU clusters. With NVTabular, you can:
- prepare datasets quickly and easily for experimentation so that more models can be trained.
- process datasets that exceed GPU and CPU memory without having to worry about scale.
- focus on what to do with the data and not how to do it by using abstraction at the operation level.

**[NVTabular data loaders](https://github.com/NVIDIA/NVTabular)**:<br>
NVTabular provides seamless integration with common deep learning frameworks, such as TensorFlow, PyTorch, and HugeCTR. When training deep learning recommender system models, data loading can be a bottleneck. Therefore, we’ve developed custom, highly-optimized data loaders to accelerate existing TensorFlow and PyTorch training pipelines. The NVTabular data loaders can lead to a speed-up that is 9 times faster than the same training pipeline used with the GPU. With the NVTabular data loaders, you can:
- remove bottlenecks from data loading by processing large chunks of data at a time instead of item by item.
- process datasets that don’t fit within the GPU or CPU memory by streaming from the disk.
- prepare batches asynchronously into the GPU to avoid CPU-GPU communication.
- integrate easily into existing TensorFlow or PyTorch training pipelines by using a similar API.

**[HugeCTR](https://github.com/NVIDIA/HugeCTR)**:<br>
HugeCTR is a GPU-accelerated framework designed to distribute training across multiple GPUs and nodes and estimate click-through rates. HugeCTR contains optimized data loaders that can be used to prepare batches with GPU-acceleration. In addition, HugeCTR is capable of scaling large deep learning recommendation models. The neural network architectures often contain large embedding tables that represent hundreds of millions of users and items. These embedding tables can easily exceed the CPU/GPU memory. HugeCTR provides strategies for scaling large embedding tables beyond available memory. With HugeCTR, you can:
- scale embedding tables over multiple GPUs or multi nodes.
- load a subset of an embedding table into the GPU in a coarse grained, on-demand manner during the training stage.

**[Triton](https://github.com/triton-inference-server/server):**<br>
NVTabular and HugeCTR both support the Triton Inference Server to provide GPU-accelerated inference. The Triton Inference Server is open source inference serving software that can be used to simplify the deployment of trained AI models from any framework to production. With Triton, you can:
- deploy NVTabular ETL workflows and trained deep learning models to production with a few lines of code.
- deploy an ensemble of NVTabular ETL and trained deep learning models to ensure that same data transformations are applied in production.
- deploy models concurrently on GPUs to maximize utilization.
- enable low latency inferencing in real time or batch inferencing to maximize GPU/CPU utilization. 
- scale the production environment with Kubernetes for orchestration, metrics, and auto-scaling using a Docker container.

### Examples

A collection of [end-to-end examples](./examples/) is available within this repository in the form of Jupyter notebooks. The examples demonstrate how to:
- download and prepare the dataset.
- use preprocessing and engineering features.
- train deep learning recommendation models with TensorFlow, PyTorch, FastAI, or HugeCTR.
- deploy the models to production.

These examples are based on different datasets and provide a wide range of real-world use cases.
Each example explains
### Resources

For more information about NVIDIA Merlin and its components, see the following:
- [NVTabular GitHub](https://github.com/NVIDIA/NVTabular)
- [HugeCTR GitHub](https://github.com/NVIDIA/HugeCTR)
- [NVTabular API Documentation](https://nvidia.github.io/NVTabular/main/Introduction.html)
- [HugeCTR User Guide](https://github.com/NVIDIA/HugeCTR/blob/master/docs/hugectr_user_guide.md)
- [HugeCTR Python API](https://github.com/NVIDIA/HugeCTR/blob/master/docs/python_interface.md)
- [NVTabular Accelerated Training Documentation](https://nvidia.github.io/NVTabular/main/training/index.html)
