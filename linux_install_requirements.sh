#!/bin/bash

echo "Creating conda environment"
conda create -n visuai python=3.8.2
conda activate visuai

echo "Installing Python dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "Do you want to install the GPU version of PyTorch (CUDA 12.4)? (y/n)"
read gpu_choice

if [ "$gpu_choice" = "y" ]; then
    echo "Installing PyTorch with CUDA support..."
    pip install torch==2.4.1+cu124 torchvision==0.19.1+cu124 torchaudio==2.4.1+cu124 --index-url https://download.pytorch.org/whl/cu124
else
    echo "Installing CPU-only PyTorch..."
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
fi

echo "Installation complete."
