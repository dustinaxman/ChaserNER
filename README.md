
export PYTHONPATH=~/Projects/ChaserNER/src/
/opt/homebrew/bin/python3 ~/Projects/ChaserNER/bin/train.py --save_model_dir ~/test_model_save_dir

WORKING_DIR=~/Downloads

expname=model_deployment_12_9_23
model_dir=${WORKING_DIR}/${expname}_model
model_dir="${model_dir%/}"
torchserve_image_name=${expname}_image
docker_container_name=${expname}_container

# commented out for deberta which doesn't yet support torchscript
# /opt/homebrew/bin/python3 /Users/deaxman/Projects/ChaserNER/bin/insert_torchscript.py --config_path ${model_dir}/config.json

/Users/deaxman/Projects/ChaserNER/bin/insert_torchserve.sh ${model_dir}

docker build -t ${torchserve_image_name} -f ${model_dir}/Dockerfile ${model_dir}/


## TEST the Docker container offline

docker run -p 8080:8080 -p 8081:8081 --name ${docker_container_name} ${torchserve_image_name}

curl -X POST http://localhost:8080/predictions/chaser_ner_model \
     -H "Content-Type: application/json" \
     -d '{"text": "dustin please finish the report on profit by 10/21"}'

docker stats ${docker_container_name}
docker inspect -f '{{.HostConfig.Memory}}' ${docker_container_name}
docker top ${docker_container_name}


## Push the container to ECR

ecr_uri="198449958201.dkr.ecr.us-east-1.amazonaws.com"

docker login -u AWS -p $(aws ecr get-login-password --region us-east-1) ${ecr_uri}
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $ecr_uri

docker tag ${torchserve_image_name} ${ecr_uri}/chaser_ner:latest
docker push ${ecr_uri}/chaser_ner:latest


## Spin up the stack

aws cloudformation create-stack --stack-name chaser-ner-host --template-body file:///Users/deaxman/Projects/ChaserNER/misc/cloudformation_template.yaml --capabilities CAPABILITY_IAM


### wait until spun up (5m)


## Get the api key and endpoint, and test API call

API_ENDPOINT=$(aws cloudformation describe-stacks --stack-name chaser-ner-host --query 'Stacks[0].Outputs[?OutputKey==`ApiEndpoint`].OutputValue' --output text | sed 's/\/$//')
echo $API_ENDPOINT

API_KEY_ID=$(aws apigateway get-api-keys --include-values --query 'items[?name==`MyRestrictedAPIKey`].id' --output text)
echo ${API_KEY_ID}

API_KEY_VALUE=$(aws apigateway get-api-key --api-key $API_KEY_ID --include-value --query 'value' --output text)
echo $API_KEY_VALUE

echo "API KEY: ${API_KEY_VALUE}"
echo "API ENDPOINT: ${API_ENDPOINT}"


curl -X POST ${API_ENDPOINT} -H "Content-Type: application/json" -H "x-api-key: ${API_KEY}" -d '{"text": "dustin please finish the report on profit by 10/21"}'

## Spin down

aws cloudformation delete-stack --stack-name chaser-ner-host

