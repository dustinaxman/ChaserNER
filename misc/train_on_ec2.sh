instance_info=$(aws ec2 run-instances --image-id ami-04cb4ca688797756f --count 1 --instance-type g5.xlarge --key-name cda_key --security-group-ids sg-00cb9b43107bb15ea)
instance_id=$(echo ${instance_info} | jq -r .Instances[0].InstanceId)

aws ec2 wait instance-running --instance-ids ${instance_id}

public_ip=$(aws ec2 describe-instances --instance-ids ${instance_id} | jq -r .Reservations[0].Instances[0].PublicIpAddress)

rsync -avz -e "ssh -i ~/Downloads/main.pem" ~/Projects/ChaserNER/ ec2-user@${public_ip}:~/ChaserNER/

ssh -i "~/Downloads/main.pem" ec2-user@${public_ip}











