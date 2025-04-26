#!/usr/bin/env bash
# disable “exit‐on‐error”
set +e

# 1) System prerequisites
apt update
apt install -y git git-lfs ffmpeg build-essential
git lfs install

# 2) Clone the Web UI
git clone https://github.com/natlamir/Wav2Lip-WebUI.git
cd Wav2Lip-WebUI || true

# from inside /Wav2Lip-WebUI
mkdir -p weights
cd weights || true

# download the face landmarker model
wget -O face_landmarker_v2_with_blendshapes.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

# 3) Install Miniconda (silent mode)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda3"
source "$HOME/miniconda3/etc/profile.d/conda.sh" || true

# 4) Create & activate a single Conda env
conda create -n wav2lip python=3.10 -y
conda activate wav2lip || true

# 5) Install GPU-accelerated PyTorch
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 6) Install all Python dependencies
pip install --upgrade pip==24.0
pip install -r requirements.txt
pip install sympy==1.13.1

# 7) Download required model weights
mkdir -p face_detection/detection/sfd
wget -O face_detection/detection/sfd/s3fd.pth \
  https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth
mkdir -p checkpoints
# TODO: manually download High-accuracy Wav2Lip / Wav2Lip+GAN into 'checkpoints/'

# 8) Upgrade Gradio for public URL & other fixes
pip install --force-reinstall "gradio==4.44.1"
pip install "numpy<1.27,>=1.22" --force-reinstall
cp ui.py ui.py.bak
sed -i 's/, *info="[^"]*"//g' ui.py
pip install pydantic==2.10.6 --force-reinstall
wget -O checkpoints/wav2lip.pth \
  https://huggingface.co/numz/wav2lip_studio/resolve/main/Wav2lip/wav2lip.pth
wget -O checkpoints/wav2lip_gan.pth \
  "https://huggingface.co/numz/wav2lip_studio/resolve/main/Wav2lip/wav2lip_gan.pth"
conda install -y -c conda-forge ffmpeg=4.3

# ─── Beholder-GAN setup ────────────────────────────────────────────

# Ensure SciPy is available
conda install -y scipy

# Clone & set up Beholder-GAN
cd "$HOME"
git clone https://github.com/beholdergan/Beholder-GAN.git
cd Beholder-GAN || true

# Beholder dependencies
pip install mediapipe==0.10.21
pip install "numpy>=1.13.3"
pip install "scipy>=1.0.0"
pip install --upgrade pip setuptools wheel
pip install "moviepy>=0.2.3.2"
pip install "Pillow>=3.1.1"
pip install "lmdb>=0.93"
pip install "opencv-python>=3.4.0.12"
pip install "cryptography>=2.1.4"
pip install "h5py>=2.7.1"
pip install "six>=1.11.0"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install tensorflow
pip install --upgrade tensorflow tensorflow-hub

# Download Beholder-GAN weights
mkdir -p models
pip install gdown
gdown --id 188K19ucknC6wg1R6jbuPEhTq9zoufOx4 -O models/

# E4E encoder setup
cd "$HOME"
git clone https://github.com/omertov/encoder4editing.git
cd encoder4editing || true
source ~/miniconda3/etc/profile.d/conda.sh || true
conda create -n e4e_env \
  python=3.10 \
  pytorch=1.11.0 \
  torchvision=0.12.0 \
  torchaudio=0.11.0 \
  cudatoolkit=11.8 \
  -c pytorch -c conda-forge -y
conda activate e4e_env || true
git clone https://github.com/omertov/encoder4editing.git

# Clean up & set up a separate TF env for Beholder-GAN
conda deactivate || true
conda create -n tf-beholder python=3.8 cudatoolkit=12.6 -c conda-forge -y
# (activate it if you need to run Beholder-GAN in this env)
pip install --upgrade tensorflow tensorflow-hub
pip install -r ~/Beholder-GAN/requirements-pip.txt
conda deactivate || true

# Final model download for encoder4editing
cd /root/encoder4editing/encoder4editing/models || true
gdown --id 1cUv_reLE6k3604or78EranS7XzuVMWeO -O ffhq_e4e_encoder.pt

# Final Step) Launch the Gradio app
export GRADIO_SHARE=true
python3 ui.py
