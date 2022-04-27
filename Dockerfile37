FROM nvidia/cuda:11.0-base-ubuntu18.04

# Environments
ARG DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
CMD nvidia-smi

# Install linux CUDA 11.0
RUN apt-get update -y && apt-get install -y cuda-nvcc-11-0

# # Install audio libraries
RUN apt-get -y update && apt-get -y install \
    python3-soundfile \
    ffmpeg
#     espeak \
#     espeak-data \
#     libespeak1 \
#     libespeak-dev

# Install Python 3.7
RUN apt-get -y update && apt-get -y install \
    python3.7 \
    python3.7-dev \
    python3-pip \
    ipython3

# Install Python libraries with pip
RUN python3.7 -m pip install --upgrade pip && pip3 \
    install --no-cache-dir numpy==1.19.4
# Install Python libraries with pip
RUN python3.7 -m pip install --no-cache-dir \
    pyyaml \
    librosa \
    mir_eval \
    matplotlib \
    tensorboardX \
    Pillow \
    tqdm \
    ffmpeg-normalize

# # Install Torch with CUDA. Tensorflow apparently was not needed
RUN python3.7 -m pip install --no-cache-dir \
    torch==1.7.0+cu110 \
    # torchvision==0.8.0+cu110 \
    torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
# RUN python3.7 -m pip install  --no-cache-dir \
#     torch==1.7.1+cu110 \
#     torchvision==0.8.2+cu110 \
#     torchaudio==0.7.2 \
#     -f https://download.pytorch.org/whl/torch_stable.html

# # Install pytorch lightning and torchtext
# RUN python3.7 -m pip install --no-cache-dir \
#     torchtext==0.8.1 \
#     pytorch-lightning==1.2.10

VOLUME /repo
WORKDIR /repo
