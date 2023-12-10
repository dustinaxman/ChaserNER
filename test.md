 # ChaserNER Project Deployment Guide

 This guide details the steps for deploying the ChaserNER model using Docker and AWS services.
 Please keep in mind that you should ensure that your aws account has an ec2 keypair called "main" 
 and has an ECR repo with this id: 198449958201.dkr.ecr.us-east-1.amazonaws.com.  If you do not have this, 
 please change this to the repo you would like to use (run aws ecr describe-repositories to see your repos)

 ## Setting Up Environment

 Before you begin, set the `PYTHONPATH` to the source directory of the ChaserNER project:

 ```bash
 export PYTHONPATH=~/Projects/ChaserNER/src/
 ```

 ## Training the Model

 Run the training script as follows:

 ```bash
 /opt/homebrew/bin/python3 ~/Projects/ChaserNER/bin/train.py --save_model_dir ~/test_model_save_dir
 ```

 ## Preparing Model Deployment

Set up the working directory and prepare the model directory:

```bash
WORKING_DIR=~/Downloads
expname=model_deployment_12_9_23
model_dir=${WORKING_DIR}/${expname}_model
model_dir="${model_dir%/}"
torchserve_image_name=${expname}_image
docker_container_name=${expname}_container
```

### Adding torchserve (and torchscript)

Add torchserve (creates the mar file in the dir) and optionall torchscript (jit) to speed things up:
```bash
# commented out for deberta which doesn't yet support torchscript
# when it supports, also change "insert_torchserve.sh" file to use "torchscript_model"
# /opt/homebrew/bin/python3 /Users/deaxman/Projects/ChaserNER/bin/insert_torchscript.py --config_path ${model_dir}/config.json
/Users/deaxman/Projects/ChaserNER/bin/insert_torchserve.sh ${model_dir}
```

### Building the Docker Image

Use the following command to build a Docker image for the TorchServe service:

```bash
docker build -t ${torchserve_image_name} -f ${model_dir}/Dockerfile ${model_dir}/
```

## Testing the Docker Container Locally

To test the Docker container offline, run the container and test it with a sample request:

```bash
docker run -p 8080:8080 -p 8081:8081 --name ${docker_container_name} ${torchserve_image_name}

curl -X POST http://localhost:8080/predictions/chaser_ner_model \
     -H "Content-Type: application/json" \
     -d '{"text": "dustin please finish the report on profit by 10/21"}'
```

### Monitoring the Docker Container

To monitor the Docker container, use the following commands:

```bash
docker stats ${docker_container_name}
docker inspect -f '{{.HostConfig.Memory}}' ${docker_container_name}
docker top ${docker_container_name}
```

## Pushing the Container to Amazon ECR

Authenticate and push the Docker image to ECR:

```bash
ecr_uri="198449958201.dkr.ecr.us-east-1.amazonaws.com"

docker login -u AWS -p $(aws ecr get-login-password --region us-east-1) ${ecr_uri}
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $ecr_uri

docker tag ${torchserve_image_name} ${ecr_uri}/chaser_ner:latest
docker push ${ecr_uri}/chaser_ner:latest
```

## Deploying the Stack on AWS

Deploy the cloud formation stack:

```bash
aws cloudformation create-stack --stack-name chaser-ner-host --template-body file:///Users/deaxman/Projects/ChaserNER/misc/cloudformation_template.yaml --capabilities CAPABILITY_IAM
```

### Wait for Stack Deployment

Wait for approximately 5 minutes until the stack is fully deployed.

## Testing the Deployed API

Retrieve the API key and endpoint, then test the API call:

```bash
# Retrieve API endpoint
API_ENDPOINT=$(aws cloudformation describe-stacks --stack-name chaser-ner-host --query 'Stacks[0].Outputs[?OutputKey==`ApiEndpoint`].OutputValue' --output text | sed 's/\/$//')
echo $API_ENDPOINT

# Retrieve API key
API_KEY_ID=$(aws apigateway get-api-keys --include-values --query 'items[?name==`MyRestrictedAPIKey`].id' --output text)
echo ${API_KEY_ID}

API_KEY_VALUE=$(aws apigateway get-api-key --api-key $API_KEY_ID --include-value --query 'value' --output text)
echo $API_KEY_VALUE

# Make API call
curl -X POST ${API_ENDPOINT} -H "Content-Type: application/json" -H "x-api-key: ${API_KEY}" -d '{"text": "dustin please finish the report on profit by 10/21"}'
```

## Spinning Down the Stack

To spin down the deployed stack:

```bash
aws cloudformation delete-stack --stack-name chaser-ner-host
```
