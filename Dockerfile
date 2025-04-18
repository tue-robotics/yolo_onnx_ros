FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

# Basic dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libopencv-dev \
    python3 \
    python3-pip \
    python3-dev \
    && apt-get clean

# (Optional) Install ONNX Runtime GPU if you use Python version
# RUN pip3 install onnxruntime-gpu==1.20.0

WORKDIR /app

CMD ["/bin/bash"]
