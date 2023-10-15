#ON DEMAND
instance_info=$(aws ec2 run-instances --image-id ami-0f837acd9af5d0944 --count 1 --instance-type g5.xlarge --key-name main --security-group-ids sg-079c29fe50f0767d7)
instance_id=$(echo ${instance_info} | jq -r .Instances[0].InstanceId)
aws ec2 wait instance-running --instance-ids ${instance_id}
public_ip=$(aws ec2 describe-instances --instance-ids ${instance_id} | jq -r .Reservations[0].Instances[0].PublicIpAddress)

rsync -avz -e "ssh -i ~/Downloads/main.pem -o StrictHostKeyChecking=no" ~/Projects/ChaserNER/ ec2-user@${public_ip}:~/ChaserNER/

ssh -i "~/Downloads/main.pem" -o "StrictHostKeyChecking=no" ec2-user@${public_ip}

sudo yum update -y
sudo yum install python3-pip -y

python3 -m pip install transformers pytorch-lightning datasets pytest seqeval lightning_lite torch torchvision

python3 -m pip install 'urllib3<2.0'


screen -D -R train
export PYTHONPATH=~/ChaserNER/src/
python3 ~/ChaserNER/bin/train.py --save_model_dir ~/test_model_save_dir


#--tokenizer_name --hf_model_name --max_epochs 5 --batch_size 128 --max_length 64 --learning_rate 0.00002 --frozen_layers 0 --min_delta 0.0 --patience 2

rm -r ~/Downloads/remote_trained_model_10_5/
rsync -avz -e "ssh -i ~/Downloads/main.pem -o StrictHostKeyChecking=no" ec2-user@${public_ip}:~/test_model_save_dir/ ~/Downloads/remote_trained_model/


aws ec2 terminate-instances --instance-ids ${instance_id}

aws ec2 wait instance-terminated --instance-ids ${instance_id}




#
#
##SPOT
#spot_request=$(aws ec2 request-spot-instances --spot-price "0.8" --instance-count 1 --type "one-time" --launch-specification "{ \"ImageId\":\"ami-0f837acd9af5d0944\", \"InstanceType\":\"g5.xlarge\", \"KeyName\":\"main\", \"SecurityGroupIds\":[\"sg-079c29fe50f0767d7\"] }")
#spot_request_id=$(echo $spot_request | jq -r .SpotInstanceRequests[0].SpotInstanceRequestId)
#aws ec2 wait spot-instance-request-fulfilled --spot-instance-request-ids $spot_request_id
#instance_id=$(aws ec2 describe-spot-instance-requests --spot-instance-request-ids $spot_request_id | jq -r .SpotInstanceRequests[0].InstanceId)
#aws ec2 wait instance-running --instance-ids ${instance_id}
#public_ip=$(aws ec2 describe-instances --instance-ids ${instance_id} | jq -r .Reservations[0].Instances[0].PublicIpAddress)



