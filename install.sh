#!/bin/bash
# Ensure if all base software result installed

# For project
sudo apt update
sudo apt upgrade
sudo apt install git gh -y

# Python (via conda)
CONDA_DIR=~/miniconda3
if [ ! -d "$CONDA_DIR" ]; then
	echo "[Installing Conda]"
	mkdir -p ~/miniconda3
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
	bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
	rm ~/miniconda3/miniconda.sh	
else
	echo "[Conda just installed, SKIP]"
fi

# configure conda
source ~/miniconda3/bin/activate
conda init --all

# create environment Vision (if not just exist)
ENV=~/miniconda3/envs/Vision

if [[ ! -d $ENV  ]]; then
	conda create -n Vision python=3.13.2 # latest release
fi

# activate environment

conda activate Vision

# install required python packages for base operations
echo '[installing Base packages]'
pip3 install scipy numpy pandas pytest
pip3 install matplotlib wandb
pip3 install seaborn
pip3 install datasets
pip3 install opencv-python
# install ml tools (and torch system)
echo '[installing Torch and Torch tools]'
nvidia-smi >> /dev/null
if [[ -x "$(command -v nvidia-smi)" ]] ; then
	pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 #require install giga-byte
else
	pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi	

pip3 install torchmetrics



