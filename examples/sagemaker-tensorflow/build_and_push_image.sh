#!/bin/bash

set -euo pipefail

# The name of our algorithm
ALGORITHM_NAME=sagemaker-merlin-tensorflow
REGION=us-east-1

cd container

ACCOUNT=$(aws sts get-caller-identity --query Account --output text --region ${REGION})

# Get the region defined in the current configuration (default to us-west-2 if none defined)

REPOSITORY="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE_URI="${REPOSITORY}/${ALGORITHM_NAME}:latest"

# Get the login command from ECR and execute it directly
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${REPOSITORY}

# If the repository doesn't exist in ECR, create it.

aws ecr describe-repositories --repository-names "${ALGORITHM_NAME}" --region ${REGION} > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${ALGORITHM_NAME}" --region ${REGION} > /dev/null
fi

# Build the docker image locally with the image name and then push it to ECR
# with the full name.

docker build  -t ${ALGORITHM_NAME} .
docker tag ${ALGORITHM_NAME} ${IMAGE_URI}

docker push ${IMAGE_URI}
