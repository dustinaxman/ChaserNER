aws ecr get-login-password --region region | docker login --username AWS --password-stdin aws_account_id.dkr.ecr.region.amazonaws.com
docker tag e9ae3c220b23 aws_account_id.dkr.ecr.us-west-2.amazonaws.com/my-repository:tag
docker push aws_account_id.dkr.ecr.us-west-2.amazonaws.com/my-repository:tag



docker login -u AWS -p $(aws ecr get-login-password --region us-east-1) 198449958201.dkr.ecr.us-east-1.amazonaws.com

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 198449958201.dkr.ecr.us-east-1.amazonaws.com
docker tag dustinaxman/btcrecover:real 198449958201.dkr.ecr.us-east-1.amazonaws.com/btcrecover
docker push 198449958201.dkr.ecr.us-east-1.amazonaws.com/btcrecover


@Abraham Martos to determine how many issues are still pending based on current fixes already implemented
Elias please create a report on Q3 sales by next friday


aws ecr create-repository --repository-name chaser_ner

aws ecr describe-repositories --repository-names chaser_ner --query 'repositories[0].repositoryUri' --output text
ecr_uri="198449958201.dkr.ecr.us-east-1.amazonaws.com"




docker login -u AWS -p $(aws ecr get-login-password --region us-east-1) ${ecr_uri}
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $ecr_uri


docker tag good_ner_spanbert_model_image_2 ${ecr_uri}/chaser_ner:latest
docker push ${ecr_uri}/chaser_ner:latest




aws ec2 describe-instances --query "Reservations[*].Instances[*].[PublicIpAddress,Tags[?Key=='Name'].Value]" --output table

ssh -i ~/Downloads/main.pem ec2-user@ec2-3-236-211-26.compute-1.amazonaws.com


INSTANCE_IP=$(aws ec2 describe-instances --query "Reservations[0].Instances[0].PublicIpAddress" --output text)
chmod 400 ~/Downloads/main.pem
ssh -i ~/Downloads/main.pem ec2-user@${INSTANCE_IP}

pub_dns=$(aws ec2 describe-instances --query "Reservations[*].Instances[*].PublicDnsName" --output text)
echo ${pub_dns}
ssh -i ~/Downloads/main.pem ec2-user@${pub_dns}

aws cloudformation create-stack --stack-name chaser-ner-host --template-body file:///Users/deaxman/Projects/ChaserNER/misc/cloudformation_template.yaml --capabilities CAPABILITY_IAM



aws cloudformation delete-stack --stack-name chaser-ner-host


API_ENDPOINT=$(aws cloudformation describe-stacks --stack-name chaser-ner-host --query 'Stacks[0].Outputs[?OutputKey==`ApiEndpoint`].OutputValue' --output text | sed 's/\/$//')
echo $API_ENDPOINT

API_KEY_ID=$(aws apigateway get-api-keys --include-values --query 'items[?name==`MyRestrictedAPIKey`].id' --output text)
echo ${API_KEY_ID}

API_KEY_VALUE=$(aws apigateway get-api-key --api-key $API_KEY_ID --include-value --query 'value' --output text)
echo $API_KEY_VALUE


curl -X POST ${URL} -H "Content-Type: application/json" -H "x-api-key: ${API_KEY}" -d '{"text": "Lindsey is going to complete the TPS reports by 5pm today"}'

curl -X POST ${API_ENDPOINT} -H "Content-Type: application/json" -H "x-api-key: ${API_KEY_VALUE}" -d '{"text": "dustin please finish the report on profit by friday"}'


curl -X POST https://jlo4wvvgxa.execute-api.us-east-1.amazonaws.com/prod -H "Content-Type: application/json" -H "x-api-key: cCRedxFXRB4pNdzX56ekr2I0ukYDlhN0aqMI8SAM" -d '{"text": "dustin please finish the report on profit by 10/21/23"}'




YOUR_API_ID=$(aws apigateway get-rest-apis --query 'items[?name==`TorchServeAPI`].id' --output text)
echo $YOUR_API_ID

YOUR_RESOURCE_ID=$(aws apigateway get-resources --rest-api-id $YOUR_API_ID --query 'items[?path==`/predictions/chaser_ner_model`].id' --output text)
echo $YOUR_RESOURCE_ID


aws apigateway get-resources --rest-api-id ${YOUR_API_ID}











curl -X POST https://yumidv6tgl.execute-api.us-east-1.amazonaws.com/prod -H "Content-Type: application/json" -H "x-api-key: 2WFXYj1aQp28U8LXk8heZ7TwyM7hdJA6UVwGUNgf" -d '{"txt": "dustin please finish the report on profit by 10/5/2023"}'
{
  "person": "dustin",
  "task": "report on profit",
  "date": "10/5/2023"
}(base) MacBook-Air:debugmar deaxman$ 
















mkdir ~/venv_dir/
cd ~/venv_dir
venv_name=chaserner4
/opt/homebrew/bin/python3 -m venv ${venv_name}
source ${venv_name}/bin/activate


#python3 -m pip install torch torchvision -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html



/opt/homebrew/bin/python3 -m pip install transformers pytorch-lightning datasets pytest


/opt/homebrew/bin/python3 -m pip install seqeval lightning_lite


/opt/homebrew/bin/python3 -m pip install torch torchvision



mkdir ~/Downloads/test_arm_build/
cd ~/Downloads/test_arm_build/
git clone https://github.com/pytorch/serve.git
cd serve



./scripts/install_dependencies_macos.sh
./benchmarks/mac_install_dependencies.sh


cd frontend
npm install
npm run build



cd ..



/opt/homebrew/bin/python3 -m pip install .

# This will produce a dist directory with the .whl file
/opt/homebrew/bin/python3 setup.py bdist_wheel


vim Dockerfile.arm64

docker build -f Dockerfile.arm64 -t torchserve-arm64:latest .


docker image inspect torchserve-arm64:latest --format '{{.Architecture}}'

























/opt/homebrew/bin/python3 -m install torchserve torch-model-archiver






model_dir=~/Downloads/good_ner_spanbert_model
model_dir="${model_dir%/}"
torchserve_image_name=good_ner_spanbert_model_image_5
docker_container_name=good_ner_spanbert_model_container_5

/opt/homebrew/bin/python3 /Users/deaxman/Projects/ChaserNER/bin/insert_torchscript.py --config_path ${model_dir}/config.json


/Users/deaxman/Projects/ChaserNER/bin/insert_torchserve.sh ${model_dir}



docker build -t ${torchserve_image_name} -f ${model_dir}/Dockerfile ${model_dir}/
docker run -p 8080:8080 -p 8081:8081 --name ${docker_container_name} ${torchserve_image_name}



curl -X POST http://localhost:8080/predictions/my_model \
     -H "Content-Type: application/json" \
     -d '{"text": "Your sample text here"}'


docker stats good_ner_spanbert_model_container_12
docker inspect -f '{{.HostConfig.Memory}}' good_ner_spanbert_model_container_12
docker top good_ner_spanbert_model_container_12







number_of_workers=1
torchserve --start --ncs --model-store ~/Downloads/good_ner_spanbert_model/ --models chaser_ner_model.mar --ts-config ~/Downloads/config.properties




curl -X POST http://localhost:8080/predictions/chaser_ner_model -H "Content-Type: application/json" -d '{"text": "this is a team effort dustin axman his is really import, please work very hard to finish the report on the operating costs by friday the twenty seventh"}'


curl -X POST http://localhost:8080/predictions/chaser_ner_model -H "Content-Type: application/json" -d '{"text": "jason will complete the excel spreadsheet work by tend of week  wht kind of dog wt why"}'





rm ~/Downloads/good_ner_spanbert_model/chaser_ner_model.mar 
rm ~/Downloads/good_ner_spanbert_model/config.properties 
/Users/deaxman/Projects/ChaserNER/bin/insert_torchserve.sh ${model_dir}

torchserve_image_name=good_ner_spanbert_model_image_2
docker_container_name=good_ner_spanbert_model_container_2


docker build -t ${torchserve_image_name} -f ${model_dir}/Dockerfile ${model_dir}/
docker run -p 8080:8080 -p 8081:8081 --name ${docker_container_name} ${torchserve_image_name}


docker rmi ${torchserve_image_name}
docker stop ${docker_container_name}
docker rm ${docker_container_name}





curl http://localhost:8081/models/chaser_ner_model


seq 40 | xargs -I{} -P40 bash -c 'curl -X POST http://localhost:8080/predictions/chaser_ner_model -H "Content-Type: application/json" -d "{\"text\": \"jason will complete the excel spreadsheet work by tend of week  wht kind of dog wt why what are $(uuidgen)\"}"'






good_ner_spanbert_model_container



for i in {1..40}; do 
    curl -X POST http://localhost:8080/predictions/my_model \
    -H "Content-Type: application/json" \
    -d '{"text": "this is a team effort dustin axman his is really import, please work very hard to finish the report on the operating costs by friday the twenty seventh '"${i}"'"}' ; 
done



#!/bin/bash

model_dir=$1

model_file_path=${model_dir}/$(cat ${model_dir}/config.json | jq '.["torchscript_model"]')

torch-model-archiver --model-name chaser_ner_model \
                     --version 1.0 \
                     --serialized-file ${model_file_path} \
                     --handler /Users/deaxman/Projects/ChaserNER/src/chaserner/inference/handler.py \
                     --extra-files /Users/deaxman/Projects/ChaserNER/src/chaserner/inference/utils.py /Users/deaxman/Projects/ChaserNER/src/chaserner/utils/__init__.py ${model_dir}/config.json \
                     --export-path ${model_dir}


cp /Users/deaxman/Projects/ChaserNER/misc/Dockerfile ${model_dir}/Dockerfile



wat=" ".join(["p"*16]*64)

a = {wat+str(i): {"task":wat, "person": wat, "date": wat}.items() for i in range(100000)}
b = {wat+str(i): int(i) for i in range(100000)}
from pympler import asizeof


print(asizeof.asizeof(a)/float(1000000), "mb")
print(asizeof.asizeof(b)/float(1000000), "mb")


3. The Custom Handler
In your model_handler.py, you'll define the custom batching logic and how to handle incoming requests. This is also where you'll need to handle the batching logic you described (waiting for 0.5 seconds or until a batch size is met).



Within the handler, you can import necessary functions or classes from model_def.py.

4. Build and Run the Docker Container
The steps remain the same as previously described. Build the Docker image and then run the container.

5. Making Requests
Once the container is up and running, you can make inference requests to the TorchServe endpoint:

bash
Copy code
curl -X POST http://localhost:8080/predictions/my_model -T sample_input.json
By following the above steps, you can effectively serve your PyTorch NER model using TorchServe inside a Docker container with custom batching logic.














from chaserner.data.simulator import *
import random
from transformers import BertTokenizerFast

train, dev, test = simulate_train_dev_test()

all_data = train+dev+test

random.shuffle(all_data)


# def proc_dict(dict_val):
#     return " | ".join([k+":"+v for k, v in dict_val.items()])

# with open("/Users/deaxman/Downloads/simulated_data.txt", "w") as f:
#     f.write("\n".join(["\t".join([" ".join([txt+"|"+lbl for txt, lbl in zip(txt_input.split(), raw_labels)]), proc_dict(labels)]) for txt_input, labels, raw_labels in all_data]))



mkdir ~/venv_dir/
cd ~/venv_dir
venv_name=chaserner4
/opt/homebrew/bin/python3 -m venv ${venv_name}
source ${venv_name}/bin/activate


#python3 -m pip install torch torchvision -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html



/opt/homebrew/bin/python3 -m pip install transformers pytorch-lightning datasets pytest


/opt/homebrew/bin/python3 -m pip install seqeval lightning_lite



model_dir=~/Downloads/good_ner_spanbert_model
rm -r ~/Downloads/debugmar
mkdir ~/Downloads/debugmar
cd ~/Downloads/debugmar
cp ${model_dir}/chaser_ner_model.mar chaser_ner_model.zip
unzip chaser_ner_model.zip -d chaser_ner_model_contents/




ner_data_module.test_dataloader




/opt/homebrew/bin/python3


import torch

all_logits = []
batchinfo = []

with torch.no_grad(): 
    for batch in ner_data_module.test_dataloader():
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        offset_mapping = batch['offset_mapping']
        batchinfo.append(batchinfo)

        #logits = model(input_ids, attention_mask, labels)
        #all_logits.append(logits)

# Concatenate all logits if they're stored in a list


all_logits = torch.cat([output.logits for output in all_logits], dim=0)


all_predicted_classes = torch.argmax(all_logits, dim=-1)



offsets = tokenized_data['offset_mapping'][0]




def join_raw_labels(raw_labels, offset_mapping):

label_idx = 0
for token_idx, (start, end) in enumerate(offsets):
    # Check if this token corresponds to a new word
    if start == 0 and end != 0:
        token_labels[token_idx] = self.label_to_id[ner_label_strings[label_idx]]
        label_idx += 1





python3
import torch

if torch.cuda.is_available():
    print("GPU is available!")
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("No GPU detected.")


export PYTHONPATH=~/Projects/ChaserNER/src/:


deactivate


tokenizer_name='SpanBERT/spanbert-base-cased'
max_length=512
tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)


text, _, ner_label_strings = all_data[0]

# Tokenizing the text
tokenized_data = tokenizer(
            text.split(),
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
            is_split_into_words=True,
            return_offsets_mapping=True
        )


offsets = tokenized_data['offset_mapping'][0]

print(tokenized_data)


token_labels = [-100] * max_length

label_idx = 0
for token_idx, (start, end) in enumerate(offsets):
    print(token_idx, start, end)
    # Check if this token corresponds to a new word
    if start == 0 and label_idx < len(ner_label_strings) and end != 0:
        token_labels[token_idx] = ner_label_strings[label_idx]
        label_idx += 1



# Step 1: Create the mask
mask = (offsets[:,:,0] == 0) & (offsets[:,:,1] == 0)

# Step 2: Use the mask to select elements from hyp_labels for each sample
selected_values_list = [hyp_labels[i][mask[i]] for i in range(mask.size(0))]

# Convert the list of tensors to a padded 2D tensor. This assumes you want to pad shorter sequences.
max_len = max([t.size(0) for t in selected_values_list])
padded_selected_values = torch.stack([torch.cat([t, torch.zeros(max_len - t.size(0))]) for t in selected_values_list])












from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from chaserner.data.data_processors import SimulatorNERDataModule
from chaserner.model import NERModel, DummyNERModel

early_stop_callback = EarlyStopping(
    monitor='avg_val_loss',  # Monitor the average validation loss
    min_delta=0.00,
    patience=3,
    verbose=True,
    mode='min'
)

# Define model checkpoint criteria
checkpoint_callback = ModelCheckpoint(
    monitor='avg_val_loss',
    dirpath='/Users/deaxman/Downloads/checkpoints',
    filename='ner-epoch{epoch:02d}-avg_val_loss{avg_val_loss:.2f}',
    save_top_k=1,
    mode='min'
)

# Initialize trainer with callbacks
trainer = Trainer(
    accelerator="mps",
    max_epochs=5,
    callbacks=[early_stop_callback, checkpoint_callback]
)


ner_data_module = SimulatorNERDataModule(batch_size=128, tokenizer_name='SpanBERT/spanbert-base-cased', max_length=32)

ner_data_module.setup('fit')

num_labels = len(ner_data_module.label_to_id)

model = NERModel.load_from_checkpoint(checkpoint_path=Path("~/Downloads/checkpoints/ner-epochepoch=09-avg_val_lossavg_val_loss=0.11.ckpt"), num_labels=num_labels)


from datasets import load_metric
import torch
seqeval_metric = load_metric("seqeval")


all_outputs = []
for batch in ner_data_module.test_dataloader():
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        raw_labels = batch['labels']
        offset_mapping = batch['offset_mapping']
        outputs = model(input_ids, attention_mask, raw_labels)
        loss = outputs.loss
        logits = outputs.logits
        all_predicted_classes = torch.argmax(logits, dim=-1)
        offset_mapping = offset_mapping.squeeze(1)
        mask = (offset_mapping[:, :, 0] == 0) & (offset_mapping[:, :, 1] != 0)
        labels_regrouped = [raw_labels[i][mask[i]] for i in range(mask.size(0))]
        hyps_regrouped = [all_predicted_classes[i][mask[i]] for i in range(mask.size(0))]
        all_outputs.append({"test_loss": loss, "labels": labels_regrouped, "hypotheses": hyps_regrouped})




avg_loss = torch.stack([x['test_loss'] for x in all_outputs]).mean()
unrolled_lbls = [sample for batch in all_outputs for sample in batch['labels']]
unrolled_hyps = [sample for batch in all_outputs for sample in batch['hypotheses']]
seqeval_results = seqeval_metric.compute(predictions=unrolled_hyps, references=unrolled_lbls)
metrics = {
    "precision": seqeval_results["overall_precision"],
    "recall": seqeval_results["overall_recall"],
    "f1": seqeval_results["overall_f1"],
    "accuracy": seqeval_results["overall_accuracy"],
    'avg_test_loss': avg_loss
}



