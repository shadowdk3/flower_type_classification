#!/bin/bash

current_dir=$(pwd)
last_folder=$(basename "$current_dir")

if [ "$last_folder" == "script" ]; then
    cd ..
fi

echo "Create python3 venv"
python3 -m venv venv

echo "Activate the virtual environment"
source venv/bin/activate

echo "Install library"
pip install -U pip
pip install -r requirements.txt
pip install wheels/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
pip install wheels/torchvision-0.15.1-cp38-cp38-manylinux2014_aarch64.whl