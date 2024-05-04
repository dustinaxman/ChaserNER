# ChaserNER Project Deployment Guide

This guide details the steps for deploying the ChaserNER model using Docker and AWS services.
Please keep in mind that you should ensure that your aws account has an ec2 keypair called "main" and pem file main_chaser.pem
and has an ECR repo with this id: 372052397911.dkr.ecr.us-east-1.amazonaws.com. 
Create ECR_and_S3 iam role for ec2, which gives ecr and s3 full access.

If you do not have this, 
please change this to the repo you would like to use (run aws ecr describe-repositories to see your repos)
Also add "model_repo" as a bucket to the AWS account if you don't have it.
Finally, make sure you have enough vcpus on your aws account to spin up one g5.xlarge and at least one c7g.large.
```bash
default_vpcid=$(aws ec2 describe-vpcs --filters "Name=is-default,Values=true" --query "Vpcs[*].{ID:VpcId}" --output text)
aws ec2 create-security-group --group-name ssh_and_scp --description "ssh_and_scp" --vpc-id ${default_vpcid}
sg_name=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=ssh_and_scp" --query "SecurityGroups[*].{ID:GroupId}" --output text)
your_computer_ip=$(curl ifconfig.me)
aws ec2 authorize-security-group-ingress --group-id ${sg_name} --protocol tcp --port 22 --cidr ${your_computer_ip}/32
aws ec2 authorize-security-group-ingress --group-id ${sg_name} --protocol tcp --port 873 --cidr ${your_computer_ip}/32
```
export AWS_PROFILE=chaser

## Training the Model

### Spin up EC2 and SSH in

```bash
sg_name=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=ssh_and_scp" --query "SecurityGroups[*].{ID:GroupId}" --output text)
instance_info=$(aws ec2 run-instances --image-id ami-0f837acd9af5d0944 --count 1 --instance-type g5.2xlarge --key-name main --security-group-ids ${sg_name})
instance_id=$(echo ${instance_info} | jq -r .Instances[0].InstanceId)
aws ec2 wait instance-running --instance-ids ${instance_id}
public_ip=$(aws ec2 describe-instances --instance-ids ${instance_id} | jq -r .Reservations[0].Instances[0].PublicIpAddress)
rsync -avz -e "ssh -i ~/Downloads/main_chaser.pem -o StrictHostKeyChecking=no" ~/Projects/ChaserNER/ ec2-user@${public_ip}:~/ChaserNER/
ssh -i "~/Downloads/main_chaser.pem" -o "StrictHostKeyChecking=no" ec2-user@${public_ip}
```


### Train model 
Run on EC2 after SSHing in above
```bash
screen -D -R train
WORKING_DIR=~/
expname=model_deployment_01_08_24_b7815c3b5ef0832acc5e51134012087abc8a1dea_v1.0.1
model_dir=${WORKING_DIR}/${expname}_model
model_dir="${model_dir%/}"
torchserve_image_name=${expname}_image
sudo yum update -y
sudo yum install python3-pip -y
sudo yum install -y docker
sudo service docker start
sudo usermod -a -G docker ec2-user
python3 -m pip install transformers pytorch-lightning datasets pytest seqeval lightning_lite torch torchvision
python3 -m pip install 'urllib3<2.0'
python3 -m pip install torchserve torch-model-archiver
export PYTHONPATH=~/ChaserNER/src/
python3 ~/ChaserNER/bin/train.py --save_model_dir ${model_dir}
#vim ~/test_model_save_dir/DESCRIPTION.txt
aws s3 cp --recursive ${model_dir}/ s3://chaser-models/${expname}/
```

### Process trained model to hostable ECR image
Add torchserve (creates the mar file in the dir) and optional torchscript (jit) to speed things up:
backup folder to s3
Build image, authenticate and push the Docker image to ECR:
```bash
expname=model_deployment_01_08_24_b7815c3b5ef0832acc5e51134012087abc8a1dea_v1.0.1
WORKING_DIR=~/Downloads/
model_dir=${WORKING_DIR}/${expname}_model
model_dir="${model_dir%/}"
torchserve_image_name=${expname}_image
aws s3 cp --recursive s3://chaser-models/${expname}/ ${model_dir}/
# commented out for deberta which doesn't yet support torchscript
# when it supports, also change "insert_torchserve.sh" file to use "torchscript_model"
# /opt/homebrew/bin/python3 ~/Projects/ChaserNER/bin/insert_torchscript.py --config_path  ${model_dir}/config.json
~/Projects/ChaserNER/bin/insert_torchserve.sh ${model_dir}

ecr_uri="372052397911.dkr.ecr.us-east-1.amazonaws.com"
docker build -t ${torchserve_image_name} -f ${model_dir}/Dockerfile ${model_dir}/
docker login -u AWS -p $(aws ecr get-login-password --region us-east-1) ${ecr_uri}
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $ecr_uri
docker tag ${torchserve_image_name} ${ecr_uri}/chaser_ner:latest
docker push ${ecr_uri}/chaser_ner:latest
```





### Delete the instance
```bash
aws ec2 terminate-instances --instance-ids ${instance_id}
aws ec2 wait instance-terminated --instance-ids ${instance_id}
```

## Deploying the Stack on AWS

Deploy the cloud formation stack (after deleting the log group if it exists so it can be created again):

```bash
aws logs delete-log-group --log-group-name "/aws/apigateway/TorchServeAPI"
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

curl -X POST ${API_ENDPOINT} -H "Content-Type: application/json" -H "x-api-key: ${API_KEY_VALUE}" -d '{"text": "Design new logo due Tuesday"}'
```

```bash
curl -X POST ${API_ENDPOINT} -H "Content-Type: application/json" -H "x-api-key: ${API_KEY_VALUE}" -d '{"text": ["Design new logo due Tuesday", "listen derek, get the dog fed by monday"]}' &

curl -X POST ${API_ENDPOINT} -H "Content-Type: application/json" -H "x-api-key: ${API_KEY_VALUE}" -d '{"text": ["Design new logo due wednesday", "listen derek, get the dog fed by monday"]}' &
```

## Spinning Down the Stack

To spin down the deployed stack:

```bash
aws cloudformation delete-stack --stack-name chaser-ner-host
```
