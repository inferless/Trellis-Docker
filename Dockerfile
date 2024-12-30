#FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-devel
FROM docker.io/library/python:3.10@sha256:81b81c80d41ec59dcee2c373b8e1d73a0b6949df793db1b043a033ca6837e02d

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    ninja-build \
    libtbb-dev \
    python3-apt \
    libdbus-1-dev \
    python3-dbus \
    libgirepository1.0-dev \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables


#RUN apt-get update && apt-get install -y \
 #   git \
  #  wget \
  #  ninja-build \
   # libtbb-dev \
   # && rm -rf /var/lib/apt/lists/*
# Set environment variables
ENV CUDA_VERSION=11.8
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6+PTX"

WORKDIR /app

# Copy TRELLIS package first
COPY TRELLIS /app/TRELLIS

# Copy API code and scripts
COPY scripts /app/scripts
COPY app.py config.yaml requirements.txt ./


# Install Pytorch
#RUN pip install --no-cache-dir torch==2.4.0 torchvision==0.19.0 --extra-index-url https://download.pytorch.org/whl/cu118

# Install other Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
#RUN pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# Install CUDA-specific packages
#RUN pip install --no-cache-dir xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu118
#RUN pip install --no-cache-dir flash-attn
#RUN pip install --no-cache-dir spconv-cu118

# Install Kaolin
#RUN pip install --no-cache-dir kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu118.html
# Install NVDIFFRAST
#RUN git clone https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast && \
#    pip install /tmp/nvdiffrast && \
#    rm -rf /tmp/nvdiffrast

# Install DIFFOCTREERAST
#RUN git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/diffoctreerast && \
#    cd /tmp/diffoctreerast && \
#    FORCE_CUDA=1 pip install . && \
#    rm -rf /tmp/diffoctreerast

# Install MIP-Splatting
#RUN git clone https://github.com/autonomousvision/mip-splatting.git /tmp/mip-splatting && \
#    pip install /tmp/mip-splatting/submodules/diff-gaussian-rasterization/ && \
#    rm -rf /tmp/mip-splatting

ENV AWS_ACCESS_KEY_ID=$MY_AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$MY_AWS_SECRET_ACCESS_KEY
ENV AWS_REGION=$MY_AWS_REGION


# Make port 8000 available to the world outside this container
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]


