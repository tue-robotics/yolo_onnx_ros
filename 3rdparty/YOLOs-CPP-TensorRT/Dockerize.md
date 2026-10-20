# Docker Guide for YOLOs-TRT

This guide covers building and running YOLOs-TRT with Docker for consistent, portable deployments on NVIDIA GPUs and Jetson devices.

## Prerequisites

### Install Docker

1. **Download Docker Desktop**: [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)

2. **Verify installation**:
   ```bash
   docker --version
   docker run hello-world
   ```

3. **Install NVIDIA Container Toolkit** (required for GPU access):
   ```bash
   # Add NVIDIA repository
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
       sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
       sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```

---

## Quick Start

### Build Images

```bash
# Data-center GPU (TensorRT)
docker build -f Dockerfile.tensorrt -t yolos-trt .

# Jetson (Xavier/Orin)
docker build -f Dockerfile.tensorrt.jetson -t yolos-trt-jetson .
```

### Run Inference

```bash
# GPU inference
docker run --gpus all --rm -it \
    -v /path/to/models:/app/models \
    -v /path/to/data:/app/data \
    yolos-trt \
    "./image_inference ./models/yolo11n.trt ./data/dog.jpg ./models/coco.names"

# Jetson inference
docker run --runtime nvidia --rm -it \
    -v /path/to/models:/app/models \
    -v /path/to/data:/app/data \
    yolos-trt-jetson \
    "./image_inference ./models/yolo11n.trt ./data/dog.jpg ./models/coco.names"
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `INFERENCE_TARGET` | Executable to run | `image_inference` |
| `MODEL_PATH` | Path to TRT engine | `models/yolo11n.trt` |
| `INPUT_PATH` | Path to input file | `data/dog.jpg` |
| `LABELS_PATH` | Path to class labels | `models/coco.names` |

### Examples

```bash
# Video inference
docker run --gpus all --rm -it \
    -v ./data:/app/data \
    -v ./models:/app/models \
    yolos-trt \
    "./video_inference ./models/yolo11n.trt ./data/sample.mp4 ./data/output.mp4 ./models/coco.names"

# Camera inference (requires device access)
docker run --gpus all --rm -it \
    --device=/dev/video0 \
    -v ./models:/app/models \
    yolos-trt \
    "./camera_inference ./models/yolo11n.trt /dev/video0 ./models/coco.names"
```

---

## Model Conversion Inside Docker

You can convert models inside the container:

```bash
docker run --gpus all --rm -it \
    -v ./models:/app/models \
    yolos-trt \
    bash -c "trtexec --onnx=./models/yolo11n.onnx --saveEngine=./models/yolo11n.trt --fp16"
```

**Note:** TensorRT engines are GPU-specific. Build engines on the same GPU architecture where you'll run inference.

---

## GUI Support

### Linux

```bash
docker run --gpus all --rm -it \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ./models:/app/models \
    yolos-trt \
    "./image_inference ./models/yolo11n.trt ./data/dog.jpg ./models/coco.names"
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `GPU not found` | Ensure NVIDIA Container Toolkit is installed and Docker restarted |
| `Engine failed to deserialize` | Rebuild TRT engine inside the container (GPU-specific) |
| `Display not found` | Configure X11 forwarding (see GUI Support) |
| `permission denied` | Run with `--privileged` or fix permissions |

### Verify GPU Access

```bash
docker run --gpus all --rm nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

---

## Image Sizes

| Image | Base | Approximate Size |
|-------|------|------------------|
| `yolos-trt` | `nvcr.io/nvidia/tensorrt` | ~5 GB |
| `yolos-trt-jetson` | `nvcr.io/nvidia/l4t-tensorrt` | ~3 GB |

---

## Docker Compose

```yaml
version: '3.8'
services:
  yolos:
    build:
      context: .
      dockerfile: Dockerfile.tensorrt
    command: "./image_inference ./models/yolo11n.trt ./data/dog.jpg ./models/coco.names"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Run with:
```bash
docker compose up
```
