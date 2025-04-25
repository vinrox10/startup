#!/usr/bin/env bash
set -e

# 1) System prerequisites
apt update
apt install -y git git-lfs ffmpeg build-essential
git lfs install

# 2) Clone the Web UI
git clone https://github.com/natlamir/Wav2Lip-WebUI.git
cd Wav2Lip-WebUI

# 3) Install Miniconda (silent mode)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda3"
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# 4) Create & activate a single Conda env
conda create -n wav2lip python=3.10 -y
conda activate wav2lip

# 5) Install GPU-accelerated PyTorch
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 6) Install all Python dependencies
pip install --upgrade pip==24.0
pip install -r requirements.txt
pip install sympy==1.13.1

# 7) Download required model weights
mkdir -p face_detection/detection/sfd
wget -O face_detection/detection/sfd/s3fd.pth https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth
mkdir -p checkpoints
# TODO: manually download and place the High-accuracy Wav2Lip and Wav2Lip+GAN models into 'checkpoints/'

# 8) Upgrade Gradio for public URL & other fixes
pip install --force-reinstall "gradio==4.44.1"
pip install "numpy<1.27,>=1.22" --force-reinstall
cp ui.py ui.py.bak
sed -i 's/, *info="[^"]*"//g' ui.py
pip install pydantic==2.10.6 --force-reinstall
wget -O checkpoints/wav2lip.pth https://huggingface.co/numz/wav2lip_studio/resolve/main/Wav2lip/wav2lip.pth
wget -O checkpoints/wav2lip_gan.pth "https://huggingface.co/numz/wav2lip_studio/resolve/main/Wav2lip/wav2lip_gan.pth"
conda install -c conda-forge ffmpeg=4.3

# 9) Real-ESRGAN-ncnn-vulkan CLI (optional)
# wget https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip -O realesrgan-ncnn-vulkan-20220424-ubuntu.zip
# apt update && apt install -y unzip
# unzip realesrgan-ncnn-vulkan-20220424-ubuntu.zip -d realesrgan_bin
# mv realesrgan_bin/realesrgan-ncnn-vulkan-20220424-ubuntu/realesrgan-ncnn-vulkan "$HOME/miniconda3/envs/wav2lip/bin/real-esrgan-ncnn-vulkan"
# cp -r realesrgan_bin/realesrgan-ncnn-vulkan-20220424-ubuntu/models "$HOME/miniconda3/envs/wav2lip/bin/"
# chmod +x "$HOME/miniconda3/envs/wav2lip/bin/real-esrgan-ncnn-vulkan"

# 10) GFPGAN 1.4 (optional)
# pip install gfpgan==1.3.8

# 11) Install Vulkan loader for NVIDIA Tesla V100 (optional)
# apt update
# apt install -y libvulkan1 vulkan-tools mesa-vulkan-drivers

# 12) GpuPixel Setup (optional)
# git clone https://github.com/pixpark/gpupixel.git
# cd gpupixel
# mkdir -p build && cd build
# sudo apt update && sudo apt install -y build-essential cmake libglfw3-dev libglew-dev libglm-dev libeigen3-dev libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev
# cmake .. -DCMAKE_BUILD_TYPE=Release -DGPUPIXEL_BUILD_DESKTOP_DEMO=ON -DCMAKE_INSTALL_PREFIX=../output
# make -j"$(nproc)"
# make install
# cd ../..
# cd Wav2Lip-WebUI/gpupixel
# mkdir -p build
# ln -sf ../output/bin/app build/gpupixel_cli
# cd ../..

# 13) Launch the Gradio app
export GRADIO_SHARE=true
python3 ui.py
