# Scaling Large Datasets with Criteo

Criteo provides the largest publicly available dataset for recommender systems with a size of 1TB of uncompressed click logs that contain 4 billion examples.
We demonstrate how to scale NVTabular, as well as:

- Use multiple GPUs and nodes with NVTabular for feature engineering.
- Train recommender system models with the NVTabular dataloader for PyTorch.
- Train recommender system models with the NVTabular dataloader for TensorFlow
- Train recommender system models with HugeCTR using multiple GPUs.
- Inference with the Triton Inference Server and TensorFlow or HugeCTR.

There are example compose files for use with Docker in the `scaling-criteo <https://github.com/NVIDIA-Merlin/Merlin/tree/main/examples/scaling-criteo>`_ directory of the Merlin repository on GitHub.
The compose files enable you to run a pair of training and inference containers.

Explore the following notebooks:

- [Download and Convert](01-Download-Convert.ipynb)
- [Feature Engineering with NVTabular](02-ETL-with-NVTabular.ipynb)
- [Training with FastAI](03-Training-with-FastAI.ipynb)
- [Training with HugeCTR](03-Training-with-HugeCTR.ipynb)
- [Training with TensorFlow](03-Training-with-TF.ipynb)
- [Deploy the HugeCTR Model with Triton Inference Server](04-Triton-Inference-with-HugeCTR.ipynb)
- [Deploy the TensorFlow Model with Triton Inference Server](04-Triton-Inference-with-TF.ipynb)