#!/bin/bash

echo "Installing Python dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "Do you want to install the GPU version of PyTorch (CUDA 12.4)? (y/n)"
read gpu_choice

if [ "$gpu_choice" = "y" ]; then
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
else
    echo "Installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio
fi

echo "✅ Installation complete."
