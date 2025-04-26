#!/usr/bin/env bash
# don’t exit on errors
set +e

# 1) System prerequisites
apt update
apt install -y git git-lfs ffmpeg build-essential
git lfs install

# 2) Clone the Web UI
git clone https://github.com/natlamir/Wav2Lip-WebUI.git
cd Wav2Lip-WebUI || true

# inside Wav2Lip-WebUI/weights
mkdir -p weights
cd weights || true
wget -O face_landmarker_v2_with_blendshapes.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task || true

# 3) Install Miniconda (silent)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh || true
bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda3" || true
source "$HOME/miniconda3/etc/profile.d/conda.sh" || true

# 4) Create & activate wav2lip env
conda create -n wav2lip python=3.10 -y || true
conda activate wav2lip || true

# 5) Install GPU PyTorch
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia || true

# 6) Python deps for Wav2Lip
pip install --upgrade pip==24.0 || true
pip install -r requirements.txt || true
pip install sympy==1.13.1 || true

# 7) Download face detector
mkdir -p face_detection/detection/sfd
wget -O face_detection/detection/sfd/s3fd.pth \
  https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth || true
mkdir -p checkpoints
# (manual: place high-accuracy Wav2Lip / GAN into checkpoints/)

# 8) Gradio fixes & models
pip install --force-reinstall "gradio==4.44.1" || true
pip install "numpy<1.27,>=1.22" --force-reinstall || true
cp ui.py ui.py.bak || true
sed -i 's/, *info="[^"]*"//g' ui.py || true
pip install pydantic==2.10.6 --force-reinstall || true
wget -O checkpoints/wav2lip.pth \
  https://huggingface.co/numz/wav2lip_studio/resolve/main/Wav2lip/wav2lip.pth || true
wget -O checkpoints/wav2lip_gan.pth \
  https://huggingface.co/numz/wav2lip_studio/resolve/main/Wav2lip/wav2lip_gan.pth || true
conda install -y -c conda-forge ffmpeg=4.3 || true

# ─── Beholder-GAN setup ─────────────────────────────────────────────────────────

# ensure SciPy
conda install -y scipy || true

cd "$HOME"
git clone https://github.com/beholdergan/Beholder-GAN.git || true
cd Beholder-GAN || true

# install gdown for weights
pip install gdown || true

# fix tensorflow-gpu requirement
sed -i 's|^tensorflow-gpu.*|tensorflow-gpu==2.12.0|' requirements-pip.txt || true

# install all
pip install -r requirements-pip.txt || true

# ensure tensorflow & hub
pip install tensorflow-gpu==2.12.0 tensorflow-hub || true

# download Beholder-GAN weights
mkdir -p models
gdown --id 188K19ucknC6wg1R6jbuPEhTq9zoufOx4 -O models/ || true

# ─── encoder4editing setup ───────────────────────────────────────────────────────

cd "$HOME"
git clone https://github.com/omertov/encoder4editing.git || true
cd encoder4editing || true

source ~/miniconda3/etc/profile.d/conda.sh || true
conda create -n e4e_env \
  python=3.10 \
  pytorch=1.11.0 \
  torchvision=0.12.0 \
  torchaudio=0.11.0 \
  cudatoolkit=11.8 \
  -c pytorch -c conda-forge -y || true
conda activate e4e_env || true

# grab models
mkdir -p encoder4editing/models
cd encoder4editing/models || true
pip install gdown || true
gdown --id 1cUv_reLE6k3604or78EranS7XzuVMWeO -O ffhq_e4e_encoder.pt || true

# ─── back to Wav2Lip & launch ────────────────────────────────────────────────────

cd "$HOME/Wav2Lip-WebUI" || true
export GRADIO_SHARE=true
python3 ui.py || true
