## [NVIDIA Merlin](https://github.com/NVIDIA-Merlin)

![GitHub tag (latest SemVer)](https://img.shields.io/github/v/tag/NVIDIA-Merlin/Merlin?sort=semver)
![GitHub License](https://img.shields.io/github/license/NVIDIA-Merlin/Merlin)
[![Documentation](https://img.shields.io/badge/documentation-blue.svg)](https://nvidia-merlin.github.io/Merlin/main/README.html)

NVIDIA Merlin is an open source library that accelerates recommender systems on
NVIDIA GPUs. The library enables data scientists, machine learning engineers,
and researchers to build high-performing recommenders at scale. Merlin includes
tools to address common feature engineering, training, and inference challenges.
Each stage of the Merlin pipeline is optimized to support hundreds of terabytes
of data, which is all accessible through easy-to-use APIs. For more information,
see [NVIDIA Merlin](https://developer.nvidia.com/nvidia-merlin) on the NVIDIA
developer web site.

### Benefits

NVIDIA Merlin is a scalable and GPU-accelerated solution, making it easy to
build recommender systems from end to end. With NVIDIA Merlin, you can:

- Transform data (ETL) for preprocessing and engineering features.
- Accelerate your existing training pipelines in TensorFlow, PyTorch, or FastAI
  by leveraging optimized, custom-built data loaders.
- Scale large deep learning recommender models by distributing large embedding
  tables that exceed available GPU and CPU memory.
- Deploy data transformations and trained models to production with only a few
  lines of code.

### Components of NVIDIA Merlin

NVIDIA Merlin consists of the following open source libraries:

**[NVTabular](https://github.com/NVIDIA-Merlin/NVTabular)**
[![PyPI version shields.io](https://img.shields.io/pypi/v/nvtabular.svg)](https://pypi.python.org/pypi/nvtabular/)
[![ Documentation](https://img.shields.io/badge/documentation-blue.svg)](https://nvidia-merlin.github.io/NVTabular/main/Introduction.html)
<br> NVTabular is a feature engineering and preprocessing library for tabular
data. The library can quickly and easily manipulate terabyte-size datasets that
are used to train deep learning based recommender systems. The library offers a
high-level API that can define complex data transformation workflows. With
NVTabular, you can:

- Prepare datasets quickly and easily for experimentation so that you can train
  more models.
- Process datasets that exceed GPU and CPU memory without having to worry about
  scale.
- Focus on what to do with the data and not how to do it by using abstraction at
  the operation level.

**[HugeCTR](https://github.com/NVIDIA-Merlin/HugeCTR)**<br> HugeCTR is a
GPU-accelerated training framework that can scale large deep learning
recommendation models by distributing training across multiple GPUs and nodes.
HugeCTR contains optimized data loaders with GPU-acceleration and provides
strategies for scaling large embedding tables beyond available memory. With
HugeCTR, you can:

- Scale embedding tables over multiple GPUs or nodes.
- Load a subset of an embedding table into a GPU in a coarse-grained, on-demand
  manner during the training stage.

**[Merlin Models](https://github.com/NVIDIA-Merlin/models)**
[![PyPI version shields.io](https://img.shields.io/pypi/v/merlin-models.svg)](https://pypi.python.org/pypi/merlin-models/)<br>
The Merlin Models library provides standard models for recommender systems with
an aim for high-quality implementations that range from classic machine learning
models to highly-advanced deep learning models. With Merlin Models, you can:

- Accelerate your ranking model training by up to 10x by using performant data
  loaders for TensorFlow, PyTorch, and HugeCTR.
- Iterate rapidly on featuring engineering and model exploration by mapping
  datasets created with NVTabular into a model input layer automatically. The
  model input layer enables you to change either without impacting the other.
- Assemble connectable building blocks for common RecSys architectures so that
  you can create of new models quickly and easily.

**[Merlin Systems](https://github.com/NVIDIA-Merlin/systems)**
[![PyPI version shields.io](https://img.shields.io/pypi/v/merlin-systems.svg)](https://pypi.python.org/pypi/merlin-systems/)<br>
Merlin Systems provides tools for combining recommendation models with other
elements of production recommender systems like feature stores, nearest neighbor
search, and exploration strategies into end-to-end recommendation pipelines that
can be served with Triton Inference Server. With Merlin Systems, you can:

- Start with an integrated platform for serving recommendations built on Triton
  Inference Server.
- Create graphs that define the end-to-end process of generating
  recommendations.
- Benefit from existing integrations with popular tools that are commonly found
  in recommender system pipelines.

**[Merlin Core](https://github.com/NVIDIA-Merlin/core)**
[![PyPI version shields.io](https://img.shields.io/pypi/v/merlin-core.svg)](https://pypi.python.org/pypi/merlin-core/)<br>
Merlin Core provides functionality that is used throughout the Merlin ecosystem.
With Merlin Core, you can:

- Use a standard dataset abstraction for processing large datasets across
  multiple GPUs and nodes.
- Benefit from a common schema that identifies key dataset features and enables
  Merlin to automate routine modeling and serving tasks.
- Simplify your code by using a shared API for constructing graphs of data
  transformation operators.

### Example Notebooks and Tutorials

A collection of [end-to-end examples](./examples/) is available within this
repository in the form of Jupyter notebooks. The example notebooks demonstrate
how to:

- download and prepare the dataset.
- use preprocessing and engineering features.
- train deep learning recommendation models with TensorFlow, PyTorch, FastAI, or
  HugeCTR.
- deploy the models to production.

These examples are based on different datasets and provide a wide range of
real-world use cases.

### Merlin Is Built On

**[cuDF](https://github.com/rapidsai/cudf)**<br> Merlin relies on cuDF for
GPU-accelerated DataFrame operations used in feature engineering.

**[Dask](https://dask.org/)**<br> Merlin relies on Dask to distribute and scale
feature engineering and preprocessing within NVTabular and to accelerate
dataloading in Merlin Models and HugeCTR.

**[Triton Inference Server](https://github.com/triton-inference-server/server)**<br>
Merlin leverages Triton Inference Server to provide GPU-accelerated serving for
recommender system pipelines.

### Feedback and Support

To report bugs or get help, please
[open an issue](https://github.com/NVIDIA-Merlin/Merlin/issues/new/choose).
