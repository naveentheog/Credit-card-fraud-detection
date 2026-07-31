#!/usr/bin/env bash
# deploy/ecr_push.sh
# ---------------------------------------------------------------
# Builds the Docker image and pushes it to AWS ECR (Elastic Container Registry).
# Run this from the project root: bash deploy/ecr_push.sh
#
# Prerequisites (you must have these set up in your own AWS account -
# none of this can run inside this sandbox, which has no AWS access):
#   - AWS CLI v2 installed and configured (`aws configure`)
#   - An IAM user/role with ECR push permissions
#   - An ECR repository already created (or this script creates one)
# ---------------------------------------------------------------
set -euo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
REPO_NAME="fraud-detection-api"
IMAGE_TAG="${IMAGE_TAG:-latest}"

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPO_NAME}"

echo "Account: ${ACCOUNT_ID}  Region: ${AWS_REGION}  Repo: ${REPO_NAME}"

# Create the ECR repo if it doesn't already exist
aws ecr describe-repositories --repository-names "${REPO_NAME}" --region "${AWS_REGION}" \
  >/dev/null 2>&1 || aws ecr create-repository --repository-name "${REPO_NAME}" --region "${AWS_REGION}"

# Authenticate Docker to ECR
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# Build, tag, push
docker build -t "${REPO_NAME}:${IMAGE_TAG}" .
docker tag "${REPO_NAME}:${IMAGE_TAG}" "${ECR_URI}:${IMAGE_TAG}"
docker push "${ECR_URI}:${IMAGE_TAG}"

echo ""
echo "Pushed: ${ECR_URI}:${IMAGE_TAG}"
echo "Use this URI in deploy/ecs-task-definition.json's 'image' field."
