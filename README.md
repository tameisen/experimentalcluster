This project set's up a (Raspberry Pi 5) Cluster with Ansible. It aims to provide the easiest way to create a fully functional Cluster with several HPC Tools as a playground for research and training purposes.

It uses Monitoring Tools like Grafana and Prometheus. Reframe for Benchmarktests. Slurm as the Scheduler. Munge and MPI for parallel computing. It can measure and Store Energy Data from the internal Raspberry Pi 5 PMIC and Shelly Plug S. It also includes some Scripts for Data Analysis.

It is part of a Bachelor thesis written by Tobias Alexander Meisen at the FH Aachen University, Germany.

Prerequisites

Use the Raspian Pi Imager to flash the necessary SD cards with Raspberry Pi Os Lite (64) (Lite recommended). Make sure to Enable SSH.

Ensure that the following tools are installed on your system:

SSH
Git
Ansible
SSH Key Setup


Copy your local SSH keys to the .ssh directory of the HeadNode:

bash
cp id_rsa ~/.ssh/
cp id_rsa.pub ~/.ssh/

Now the Headnode should have access to the Computing Nodes. This is of course not neccessary, if you run the Raspian Imager on the Headnode.

Set the appropriate permissions for the SSH keys:

bash
sudo chmod 600 ~/.ssh/id_rsa ~/.ssh/id_rsa.pub

Login to youre Computing Nodes from your Headnode:

bash
ssh node@node001

Accept all fingerprints with "yes".

Update and upgrade your system packages:

bash
sudo apt update
sudo apt upgrade -y


Install Git:

bash
sudo apt install git -y
Clone the repository

bash
git clone https://github.com/tameisen/experimentalcluster.git


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
ansible-playbook -i inventory.yaml clustersetup.yaml

You can find a short documentation here:
https://docs.google.com/document/d/163e-198UjBhi-P1V7CVf_UOZJqFy_x6RHkrAmssChVc/edit?tab=t.0
