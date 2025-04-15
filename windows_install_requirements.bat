@echo off
echo Installing Python dependencies from requirements.txt...
pip install -r requirements.txt

echo.
set /p gpu_choice=Do you want to install the GPU version of PyTorch (CUDA 12.4)? (y/n): 

if /i "%gpu_choice%"=="y" (
    echo Installing PyTorch with CUDA support...
    pip install torch==2.4.1+cu124 torchvision==0.19.1+cu124 torchaudio==2.4.1+cu124 --index-url https://download.pytorch.org/whl/cu124
) else (
    echo Installing CPU-only PyTorch...
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
)

echo.
echo Installation complete.
pause
