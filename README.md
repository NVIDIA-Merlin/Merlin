## [NVIDIA Merlin](https://github.com/NVIDIA-Merlin)

![GitHub tag (latest SemVer)](https://img.shields.io/github/v/tag/NVIDIA-Merlin/Merlin?sort=semver)
![GitHub License](https://img.shields.io/github/license/NVIDIA-Merlin/Merlin)

NVIDIA Merlin is an open source library designed to accelerate recommender systems on NVIDIA’s GPUs. It enables data scientists, machine learning engineers, and researchers to build high-performing recommenders at scale. Merlin includes tools to address common ETL, training, and inference challenges. Each stage of the Merlin pipeline is optimized to support hundreds of terabytes of data, which is all accessible through easy-to-use APIs. For more information, see [NVIDIA Merlin](https://developer.nvidia.com/nvidia-merlin).

### Benefits

NVIDIA Merlin is a scalable and GPU-accelerated solution, making it easy to build recommender systems from end to end. With NVIDIA Merlin, you can:
* transform data (ETL) for preprocessing and engineering features.
* accelerate existing training pipelines in TensorFlow, PyTorch, or FastAI by leveraging optimized, custom-built dataloaders.
* scale large deep learning recommender models by distributing large embedding tables that exceed available GPU and CPU memory.
* deploy data transformations and trained models to production with only a few lines of code.

### Components of NVIDIA Merlin

NVIDIA Merlin consists of the following open source libraries:
* NVTabular
* HugeCTR
* Models
* Systems
* Core

<p align="center">
<img src='https://developer.nvidia.com/sites/default/files/akamai/merlin/recommender-systems-dev-web-850.svg' width="65%">
</p>

**[NVTabular](https://github.com/NVIDIA-Merlin/NVTabular)** [![PyPI version shields.io](https://img.shields.io/pypi/v/merlin-core.svg)](https://pypi.python.org/pypi/merlin-core/) [![ Documentation](https://img.shields.io/badge/documentation-blue.svg)](https://nvidia-merlin.github.io/NVTabular/main/Introduction.html) <br>
NVTabular is a feature engineering and preprocessing library for tabular data. It is designed to quickly and easily manipulate terabyte-size datasets that are used to train deep learning based recommender systems. NVTabular offers a high-level API that can define complex data transformation workflows. With NVTabular you can:
- prepare datasets quickly and easily for experimentation so that more models can be trained.
- process datasets that exceed GPU and CPU memory without having to worry about scale.
- focus on what to do with the data and not how to do it by using abstraction at the operation level.

**[HugeCTR](https://github.com/NVIDIA-Merlin/HugeCTR)**<br>
HugeCTR is a GPU-accelerated training framework designed to scale large deep learning recommendation models by distributing training across multiple GPUs and nodes. HugeCTR contains optimized dataloaders with GPU-acceleration and provides strategies for scaling large embedding tables beyond available memory. With HugeCTR, you can:
- scale embedding tables over multiple GPUs or nodes.
- load a subset of an embedding table into the GPU in a coarse grained, on-demand manner during the training stage.

**[Merlin Models](https://github.com/NVIDIA-Merlin/models)** [![PyPI version shields.io](https://img.shields.io/pypi/v/merlin-models.svg)](https://pypi.python.org/pypi/merlin-models/)<br>
The Merlin Models library provides standard models for recommender systems with an aim for high quality implementations that range from classic machine learning models to highly-advanced deep learning models. It features:
 - Performant dataloaders for Tensorflow, PyTorch and HugeCTR, accelerating ranking model training by up to 10x.
 - Fast iteration of featuring engineering and model exploration by mapping datasets created by NVTabular into a model input layer automatically, allowing either to be changed without impacting the other.
 - Connectable building blocks for common RecSys architectures, allowing for the creation of new models quickly and easily.

**[Merlin Systems](https://github.com/NVIDIA-Merlin/systems)** [![PyPI version shields.io](https://img.shields.io/pypi/v/merlin-systems.svg)](https://pypi.python.org/pypi/merlin-systems/)<br>
Merlin Systems provides tools for combining recommendation models with other elements of production recommender systems like feature stores, nearest neighbor search, and exploration strategies into end-to-end recommendation pipelines that can be served with Triton Inference Server.  It provides:
 - An integrated platform for serving recommendations built on Triton Inference Server.
 - Easy to create graphs that define the end-to-end process of generating recommendations.
 - Integrations with existing tools commonly found in recommender system pipelines.

**[Merlin Core](https://github.com/NVIDIA-Merlin/core)** [![PyPI version shields.io](https://img.shields.io/pypi/v/merlin-core.svg)](https://pypi.python.org/pypi/merlin-core/)<br>
Merlin Core provides functionality used throughout the Merlin ecosystem including:
- A standard dataset abstraction for processing large datasets across multiple GPUs and nodes
- A schema that identifies key dataset features, enabling Merlin to automate common modeling and serving tasks
- A shared API for constructing graphs of data transformation operators

### Example Notebooks and Tutorials

A collection of [end-to-end examples](./examples/) is available within this repository in the form of Jupyter notebooks. The example notebooks demonstrate how to:
- download and prepare the dataset.
- use preprocessing and engineering features.
- train deep learning recommendation models with TensorFlow, PyTorch, FastAI, or HugeCTR.
- deploy the models to production.

These examples are based on different datasets and provide a wide range of real-world use cases.

### Merlin Is Built On

**[cuDF](https://github.com/rapidsai/cudf)**<br>
Merlin relies on cuDF for GPU-accelerated DataFrame operations used in feature engineering.

**[Dask](https://dask.org/)**<br>
Merlin relies on Dask to distribute and scale feature engineering and preprocessing within NVTabular and to accelerate dataloading in Merlin Models and HugeCTR.

**[Triton Inference Server](https://github.com/triton-inference-server/server)**<br>
Merlin leverages Triton Inference Server to provide GPU-accelerated serving for recommender system pipelines. 

## Feedback and Support

To report bugs or get help, please [open an issue](https://github.com/NVIDIA-Merlin/Merlin/issues/new/choose).
