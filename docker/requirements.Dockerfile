FROM python:3.8.16-slim-buster
RUN apt-get update && apt-get install -y git
RUN pip install --upgrade pip && pip install pip-tools==6.12.3
