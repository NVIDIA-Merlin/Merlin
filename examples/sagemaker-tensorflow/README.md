# Training and Serving Merlin on AWS SageMaker

With AWS Sagemaker, you can package your own models that can then be trained
and deployed in the SageMaker environment.
The following notebook will show you how to use NVIDIA Merlin for training and
inference in the SageMaker environment.

- [Training and Serving Merlin on AWS SageMaker]

It assumes that readers are familiar wtth some basic concepts in NVIDIA Merlin,
such as:

- Using NVTabular to GPU-accelerate preprocessing and feature engineering,
- Training a ranking model using Merlin Models, and
- Inference with the Triton Inference Server and Merlin Models for Tensorflow.

To learn more about these concepts in NVIDIA Merlin, see for example
[Deploying a Multi-Stage Recommender System](../Building-and-deploying-multi-stage-RecSys/README.md)
in this repository or example notebooks in
[Merlin Models](https://github.com/NVIDIA-Merlin/models/tree/main/examples).


## Running the Example Notebook

You can run the example notebook either from your local environment or from
a Sagemaker notebook instance.
To run it locally, you need to configure basic AWS settings.
For setting up AWS credentials, we refer you to
[AWS documentations](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html).
