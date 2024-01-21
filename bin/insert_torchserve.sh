#!/bin/bash

model_dir=$1
model_dir="${model_dir%/}"
#model_file_path=${model_dir}/$(cat ${model_dir}/config.json | jq -r '.["torchscript_model"]')
model_file_path=${model_dir}/$(cat ${model_dir}/config.json | jq -r '.["best_checkpoint"]')


#zip -r ${model_dir}/chaser_extras.zip /home/ec2-user/ChaserNER/src/chaserner
export PATH="/opt/homebrew/bin:$PATH"

TEMP_DIR=$(mktemp -d)
ln -s "/home/ec2-user/ChaserNER/src/chaserner" $TEMP_DIR

torch-model-archiver --model-name chaser_ner_model \
                     --version 1.0 \
                     --serialized-file ${model_file_path} \
                     --handler /home/ec2-user/ChaserNER/src/chaserner/inference/handler.py \
                     --extra-files $TEMP_DIR,${model_dir}/config.json,/home/ec2-user/ChaserNER/misc/chaser_ner_model.config \
                     --export-path ${model_dir} \
#                     --config-file /home/ec2-user/ChaserNER/misc/chaser_ner_model_config.yaml

rm -rf $TEMP_DIR

cp /home/ec2-user/ChaserNER/misc/chaser_ner_model_config.yaml ${model_dir}/chaser_ner_model_config.yaml
cp /home/ec2-user/ChaserNER/misc/Dockerfile ${model_dir}/Dockerfile
cp /home/ec2-user/ChaserNER/misc/torchserve_properties ${model_dir}/config.properties
cp /home/ec2-user/ChaserNER/misc/chaser_ner_model.config ${model_dir}/chaser_ner_model.config