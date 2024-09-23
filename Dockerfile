FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

ARG WORKDIR=/root
WORKDIR ${WORKDIR}

RUN export DEBIAN_FRONTEND=noninteractive && apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    libgl1 && \
    apt-get clean autoclean && \
    apt-get autoremove --yes && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /var/lib/{apt,dpkg,cache,log}/
    
RUN pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

RUN pip3 install jinja2 \
                jupyter-client \
                pyzmq \
                notebook \
                jupyterlab \
                pyyaml \
                tqdm \
                scipy \
                opencv-python \
                scikit-image \
                tensorboard \
                matplotlib \
                timm==0.5.4 \
                open3d

COPY . /root/IGEV-plusplus
