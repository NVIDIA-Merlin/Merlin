# Getting Started with Merlin and the MovieLens Dataset

The MovieLens25M is a popular dataset for recommender systems and is used in academic publications.
Most users are familiar with the dataset and we will teach the basic concepts of Merlin:

- Learn to use NVTabular for using GPU-accelerated feature engineering and data preprocessing.
- Become familiar with the high-level API for NVTabular.
- Use single-hot/multi-hot categorical input features with NVTabular.
- Train a Merlin Model with Tensorflow.
- Use the Merlin Dataloader with PyTorch.
- Train a HugeCTR model.
- Serve recommendations from the Tensorflow model with the Triton Inference Server.
- Serve recommendations from the HugeCTR model with the Triton Inference Server.

Explore the following notebooks:

- [Download and Convert](01-Download-Convert.ipynb)
- [Feature Engineering with NVTabular](02-ETL-with-NVTabular.ipynb)
- [Training with TensorFlow](03-Training-with-TF.ipynb)
- [Training with PyTorch](03-Training-with-PyTorch.ipynb)
- [Training with HugeCTR](03-Training-with-HugeCTR.ipynb)
- [Serve Recommendations with Triton Inference Server (Tensorflow)](04-Triton-Inference-with-TF.ipynb)
- [Serve Recommendations with Triton Inference Server (HugeCTR)](04-Triton-Inference-with-HugeCTR.ipynb)
