FROM nvcr.io/nvidia/merlin/merlin-tensorflow:23.08

RUN pip3 install sagemaker-training

COPY --chown=1000:1000 serve /usr/bin/serve
