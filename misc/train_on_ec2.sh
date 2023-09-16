instance_info=$(aws ec2 run-instances --image-id ami-04cb4ca688797756f --count 1 --instance-type g5.xlarge --key-name cda_key --security-group-ids sg-00cb9b43107bb15ea)
instance_id=$(echo ${instance_info} | jq -r .Instances[0].InstanceId)

aws ec2 wait instance-running --instance-ids ${instance_id}

public_ip=$(aws ec2 describe-instances --instance-ids ${instance_id} | jq -r .Reservations[0].Instances[0].PublicIpAddress)

rsync -avz -e "ssh -i ~/Downloads/main.pem" ~/Projects/ChaserNER/ ec2-user@${public_ip}:~/ChaserNER/

ssh -i "~/Downloads/main.pem" ec2-user@${public_ip}


python3 -m pip install transformers pytorch-lightning datasets pytest seqeval lightning_lite torch torchvision




python3 ~/ChaserNER/bin/train.py --save_model_dir ~/test_model_save_dir

#--tokenizer_name --hf_model_name --max_epochs 5 --batch_size 128 --max_length 64 --learning_rate 0.00002 --frozen_layers 0 --min_delta 0.0 --patience 2




