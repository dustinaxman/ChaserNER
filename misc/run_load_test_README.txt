aws cloudformation create-stack --stack-name chaser-load-test --template-body file:///Users/deaxman/Projects/ChaserNER/misc/load_testing_t3nano_spinup_CLOUDFORMATION.yaml --capabilities CAPABILITY_IAM

public_ip=$(aws ec2 describe-instances   --filters "Name=instance-type,Values=t3.micro"   --query "Reservations[*].Instances[0].PublicIpAddress"  --output text)

echo ${public_ip}

rsync -avz -e "ssh -i ~/Downloads/main_chaser.pem -o StrictHostKeyChecking=no" ~/Projects/ChaserNER/ ec2-user@${public_ip}:~/ChaserNER/
ssh -i "~/Downloads/main_chaser.pem" -o "StrictHostKeyChecking=no" ec2-user@${public_ip}


cd ~/ChaserNER/misc/
artillery run artillery_config.yaml

aws cloudformation delete-stack --stack-name chaser-load-test