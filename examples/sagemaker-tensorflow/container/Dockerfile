FROM nvcr.io/nvidia/merlin/merlin-tensorflow:22.09

RUN pip3 install sagemaker-training

COPY train.py /opt/ml/code/train.py
COPY serve /opt/ml/code/serve

ENV SAGEMAKER_PROGRAM train.py
ENV SAGEMAKER_TRITON_TENSORFLOW_VERSION 2

EXPOSE 8080
