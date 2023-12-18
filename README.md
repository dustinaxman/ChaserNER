# ChaserNER Project Deployment Guide

This guide details the steps for deploying the ChaserNER model using Docker and AWS services.
Please keep in mind that you should ensure that your aws account has an ec2 keypair called "main" 
and has an ECR repo with this id: 198449958201.dkr.ecr.us-east-1.amazonaws.com. 

If you do not have this, 
please change this to the repo you would like to use (run aws ecr describe-repositories to see your repos)
Also add "model_repo" as a bucket to the AWS account if you don't have it.
Finally, make sure you have enough vcpus on your aws account to spin up one g5.xlarge and at least one c7g.large.

## Training the Model

### Spin up EC2 and SSH in

```bash
instance_info=$(aws ec2 run-instances --image-id ami-0f837acd9af5d0944 --count 1 --instance-type g5.xlarge --key-name main --security-group-ids sg-079c29fe50f0767d7)
instance_id=$(echo ${instance_info} | jq -r .Instances[0].InstanceId)
aws ec2 wait instance-running --instance-ids ${instance_id}
public_ip=$(aws ec2 describe-instances --instance-ids ${instance_id} | jq -r .Reservations[0].Instances[0].PublicIpAddress)
rsync -avz -e "ssh -i ~/Downloads/main.pem -o StrictHostKeyChecking=no" ~/Projects/ChaserNER/ ec2-user@${public_ip}:~/ChaserNER/
ssh -i "~/Downloads/main.pem" -o "StrictHostKeyChecking=no" ec2-user@${public_ip}
```


### Train model 
Run on EC2 after SSHing in above
```bash
sudo yum update -y
sudo yum install python3-pip -y
python3 -m pip install transformers pytorch-lightning datasets pytest seqeval lightning_lite torch torchvision
python3 -m pip install 'urllib3<2.0'
screen -D -R train
export PYTHONPATH=~/ChaserNER/src/
python3 ~/ChaserNER/bin/train.py --save_model_dir ~/test_model_save_dir
#vim ~/test_model_save_dir/DESCRIPTION.txt
```

### Adding torchserve (and torchscript)
Run on EC2 after SSHing in above
Add torchserve (creates the mar file in the dir) and optionall torchscript (jit) to speed things up:
```bash
# commented out for deberta which doesn't yet support torchscript
# when it supports, also change "insert_torchserve.sh" file to use "torchscript_model"
# /opt/homebrew/bin/python3 /Users/deaxman/Projects/ChaserNER/bin/insert_torchscript.py --config_path ~/test_model_save_dir/config.json
/Users/deaxman/Projects/ChaserNER/bin/insert_torchserve.sh ~/test_model_save_dir
```

## Preparing and Pushing Container
Run this on your local laptop
Set up the working directory and prepare the model directory:

```bash
WORKING_DIR=~/Downloads
expname=model_deployment_12_12_23_a6e50694ad7d184fb9c07f68d9b4584a08d40af2_v1.0.0
model_dir=${WORKING_DIR}/${expname}_model
model_dir="${model_dir%/}"
torchserve_image_name=${expname}_image
docker_container_name=${expname}_container
```

### Pulling the model down locally and delete the instance
```bash
rm -r ${model_dir}
rsync -avz -e "ssh -i ~/Downloads/main.pem -o StrictHostKeyChecking=no" ec2-user@${public_ip}:~/test_model_save_dir/ ${model_dir}/
aws ec2 terminate-instances --instance-ids ${instance_id}
aws ec2 wait instance-terminated --instance-ids ${instance_id}
```

### Push to s3
```bash
aws s3 cp -r ${model_dir}/ s3://chaser-model-repo/${expname}/
```

### Building the Docker Image

Use the following command to build a Docker image for the TorchServe service. 
To optionally test locally, see "LOCAL_TESTING_README.md"

```bash
docker build -t ${torchserve_image_name} -f ${model_dir}/Dockerfile ${model_dir}/
```

### Pushing the Container to Amazon ECR

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
API_ENDPOINT=$(aws cloudformation describe-stacks --stack-name chaser-ner-host --query 'Stacks[0].Outputs[?OutputKey==`ApiEndpoint`].OutputValue' --output text | sed 's/\/$//')
API_KEY_ID=$(aws apigateway get-api-keys --include-values --query 'items[?name==`MyRestrictedAPIKey`].id' --output text)
API_KEY_VALUE=$(aws apigateway get-api-key --api-key $API_KEY_ID --include-value --query 'value' --output text)
echo "API KEY: ${API_KEY_VALUE}"
echo "API ENDPOINT: ${API_ENDPOINT}"

curl -X POST ${API_ENDPOINT} -H "Content-Type: application/json" -H "x-api-key: ${API_KEY}" -d '{"text": "dustin please finish the report on profit by 10/21"}'
```

## Spinning Down the Stack

To spin down the deployed stack:

```bash
aws cloudformation delete-stack --stack-name chaser-ner-host
```
