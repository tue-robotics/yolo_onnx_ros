# yolo_onnx_ros

C++ YOLO object detection with two backends:
- **CPU (default)** — ONNX Runtime, runs on any machine
- **GPU** — TensorRT, sub-2ms inference on NVIDIA GPUs

## Project Structure

```
yolo_onnx_ros/
├── CMakeLists.txt
├── images/                          # Test images
├── include/
│   ├── yolo_onnx_ros/               # ONNX backend headers
│   │   ├── config.hpp.in
│   │   ├── detection.hpp
│   │   └── yolo_inference.hpp
│   └── yolos/                       # TensorRT backend (header-only)
│       ├── core/                    # Engine, CUDA preprocessing, NMS, drawing
│       └── tasks/detection.hpp      # TRT detection API
└── src/
    ├── detection.cpp                # ONNX backend implementation
    ├── yolo_inference.cpp           # ONNX backend implementation
    ├── main.cpp                     # ONNX entry point
    └── main_trt.cpp                 # TensorRT entry point
```

## Dependencies

### CPU backend (always required)
| Dependency | Version | Install |
|---|---|---|
| CMake | ≥ 3.5 | `sudo apt install cmake` |
| OpenCV | ≥ 4.5 | `sudo apt install libopencv-dev` |
| console_bridge | any | `sudo apt install libconsole-bridge-dev` |
| ONNX Runtime | ≥ 1.16 | [Download from GitHub releases](https://github.com/microsoft/onnxruntime/releases) |

### GPU backend (required only when `CUDA_ENABLED=ON`)
| Dependency | Version | Install |
|---|---|---|
| CUDA Toolkit | ≥ 12.0 | `sudo apt install nvidia-cuda-toolkit` |
| TensorRT | ≥ 10.0 | `sudo apt install libnvinfer-dev libnvinfer-headers-dev` |
| NVIDIA GPU | CC ≥ 7.5 | Driver managed by Windows (WSL2) |

### Install ONNX Runtime
```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-1.20.1.tgz
tar -xzf onnxruntime-linux-x64-1.20.1.tgz
sudo cp -r onnxruntime-linux-x64-1.20.1/include /usr/local/onnxruntime/
sudo cp -r onnxruntime-linux-x64-1.20.1/lib     /usr/local/onnxruntime/
echo '/usr/local/onnxruntime/lib' | sudo tee /etc/ld.so.conf.d/onnxruntime.conf
sudo ldconfig
```

### Install TensorRT (Ubuntu 24.04)
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y libnvinfer-dev libnvinfer-headers-dev
```

## Build

### CPU-only (ONNX Runtime)
```bash
mkdir build && cd build
cmake .. -DONNXRUNTIME_ROOT=/usr/local/onnxruntime
make -j4
```

### GPU (TensorRT)
```bash
mkdir build && cd build
cmake .. -DONNXRUNTIME_ROOT=/usr/local/onnxruntime -DCUDA_ENABLED=ON
make -j4
```

If TensorRT is installed to a non-standard path:
```bash
cmake .. -DONNXRUNTIME_ROOT=/usr/local/onnxruntime -DCUDA_ENABLED=ON -DTENSORRT_ROOT=/path/to/tensorrt
```

## Run

### CPU (ONNX Runtime)
The model is downloaded automatically during cmake. Pass an image directory:
```bash
LD_LIBRARY_PATH=/usr/local/onnxruntime/lib ./test_yolo_onnx_ros resources/yolo11m/yolo11m.onnx /path/to/images
```

To see detections printed and drawn, rebuild with logging enabled:
```bash
cmake .. -DONNXRUNTIME_ROOT=/usr/local/onnxruntime -DCMAKE_CXX_FLAGS="-DLOGGING"
make -j4
```

### GPU (TensorRT)
TensorRT requires a `.trt` engine file. Convert from ONNX first:
```bash
# Install conversion tools
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt').export(format='onnx')"

# Convert to TRT engine
trtexec --onnx=yolo11n.onnx --saveEngine=yolo11n.trt --fp16
```

Then run:
```bash
./test_yolo_onnx_ros_trt yolo11n.trt coco.names /path/to/images
```
