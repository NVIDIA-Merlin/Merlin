# Scaling Large Datasets with Criteo

Criteo provides the largest publicly available dataset for recommender systems with a size of 1TB of uncompressed click logs that contain 4 billion examples.

We demonstrate how to scale NVTabular, as well as:

- Use multiple GPUs and nodes with NVTabular for feature engineering.
- Train recommender system models with the Merlin Models for TensorFlow.
- Inference with the Triton Inference Server and Merlin Models for TensorFlow.

Our recommendation is to use our latest stable [Merlin containers](https://catalog.ngc.nvidia.com/containers?filters=&orderBy=dateModifiedDESC&query=merlin) for the examples. Each notebook provides the required container.  

Explore the following notebooks:

Training and Deployment with **TensorFlow**:
- [Download and Convert](01-Download-Convert.ipynb)
- [Feature Engineering with NVTabular](02-ETL-with-NVTabular.ipynb)
- [Training with TensorFlow](03-Training-with-Merlin-Models-TensorFlow.ipynb)
- [Deploy the TensorFlow Model with Triton Inference Server](04-Triton-Inference-with-Merlin-Models-TensorFlow.ipynb)
