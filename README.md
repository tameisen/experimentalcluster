Cluster Automation and Energy Optimization
This project is part of a Bachelor thesis at FH Aachen, Germany. The goal is to automate cluster management and optimize energy consumption across the infrastructure using Raspberry Pi 5's.

Prerequisites
Ensure that the following tools are installed on your system:

SSH
Git
Ansible
SSH Key Setup


Copy your SSH keys to the .ssh directory:

bash
cp id_rsa ~/.ssh/
cp id_rsa.pub ~/.ssh/
cp deploy_rsa ~/.ssh/
cp deploy_rsa.pub ~/.ssh/


Set the appropriate permissions for the SSH keys:

bash
sudo chmod 600 ~/.ssh/id_rsa ~/.ssh/id_rsa.pub ~/.ssh/deploy_rsa ~/.ssh/deploy_rsa.pub


Update and upgrade your system packages:

bash
sudo apt update
sudo apt upgrade -y


Project Deployment
You can choose to deploy the project using scp or git:

Option 1: Using SCP (headnode)
Transfer the project files to your local machine:

bash
scp -r node@192.168.0.1:~/Documents/AnsibleSetup C:\Users\User\Desktop (example)


Option 2: Using Git (headnode)
Install Git:

bash
sudo apt install git -y
Clone the repository using a specific SSH key:

bash
GIT_SSH_COMMAND='ssh -i /home/node/.ssh/deploy_rsa' git clone git@github.com:tameisen/clenman.git


Alternatively, configure SSH for GitHub:

bash
nano ~/.ssh/config
Add the following configuration:

plaintext
Host github-repo
    HostName github.com
    User git
    IdentityFile /home/user/.ssh/deploy_key_repo_name
Navigate to the project directory:

bash
cd clenman/ansible/


Ansible Setup
Install Ansible:

bash
sudo apt install ansible -y


Install the necessary Ansible collections:

bash
sudo ansible-galaxy collection install community.general
(this may take some time)

Make sure you have accessed all Nodes via SSH from the headnode and accepted the fingerprint. Make sure you are using the hostname (instead of the IP)

Running the Playbook
To execute the playbook for automating cluster setup and energy optimization, run:

bash
ansible-playbook -i inventory.yaml clustersetuprework.yaml

License
This project is licensed under the MIT License.

