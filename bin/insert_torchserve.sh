#!/bin/bash

model_dir=$1
model_dir="${model_dir%/}"
model_file_path=${model_dir}/$(cat ${model_dir}/config.json | jq -r '.["torchscript_model"]')
#model_file_path=${model_dir}/$(cat ${model_dir}/config.json | jq -r '.["best_checkpoint"]')


# Path to the config file
#config_file_path="${model_dir}/config.json"

# Get the model_file_path based on the existence of the keys
#if model_file_path=$(jq -r 'if has("torchscript_model") then .torchscript_model else empty end' "${config_file_path}"); then
#    echo "Using torchscript_model key"
#else
#    model_file_path=$(jq -r '.best_checkpoint' "${config_file_path}")
#    echo "Using best_checkpoint key"
#fi
#
## Prepend the directory path to the model_file_path
#model_file_path="${model_dir}/${model_file_path}"
#
## Print the resulting path
#echo "Model file path: ${model_file_path}"

#zip -r ${model_dir}/chaser_extras.zip /Users/deaxman/Projects/ChaserNER/src/chaserner
#export PATH="/opt/homebrew/bin:$PATH"

TEMP_DIR=$(mktemp -d)
ln -s "$(dirname "$0")/../src/chaserner" $TEMP_DIR
rm ${model_dir}/chaser_ner_model.mar
torch-model-archiver --model-name chaser_ner_model \
                     --version 1.0 \
                     --serialized-file ${model_file_path} \
                     --handler $(dirname "$0")/../src/chaserner/inference/handler.py \
                     --extra-files $TEMP_DIR,${model_dir}/config.json,$(dirname "$0")/../misc/chaser_ner_model.config \
                     --export-path ${model_dir} \
#                     --config-file /Users/deaxman/Projects/ChaserNER/misc/chaser_ner_model_config.yaml

rm -rf $TEMP_DIR
rm ${model_dir}/chaser_ner_model_config.yaml
rm ${model_dir}/Dockerfile
rm ${model_dir}/config.properties
rm ${model_dir}/chaser_ner_model.config
cp $(dirname "$0")/../misc/chaser_ner_model_config.yaml ${model_dir}/chaser_ner_model_config.yaml
cp $(dirname "$0")/../misc/Dockerfile ${model_dir}/Dockerfile
cp $(dirname "$0")/../misc/torchserve_properties ${model_dir}/config.properties
cp $(dirname "$0")/../misc/chaser_ner_model.config ${model_dir}/chaser_ner_model.config