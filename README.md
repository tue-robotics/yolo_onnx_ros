# YOLOv8 OnnxRuntime C++

<img alt="C++" src="https://img.shields.io/badge/C++-17-blue.svg?style=flat&logo=c%2B%2B"> <img alt="Onnx-runtime" src="https://img.shields.io/badge/OnnxRuntime-717272.svg?logo=Onnx&logoColor=white">



This algorithm is inspired by [Ultralitics](https://github.com/ultralytics/ultralytics/tree/main/examples/YOLOv8-ONNXRuntime-CPP) implementation to perform inference using YOLOv8 (we also supports v11) in C++ with ONNX Runtime and OpenCV's API.

## Benefits ✨

- Friendly for deployment in the industrial sector.
- Faster than OpenCV's DNN inference on both CPU and GPU.
- Supports FP32 and FP16 CUDA acceleration.

## Note ☕

1. Benefit for Ultralytics' latest release, a `Transpose` op is added to the YOLOv8 model, while make v8 and v5 has the same output shape. Therefore, you can run inference with YOLOv5/v7/v8 via this project.

## Exporting YOLOv8 Models 📦

To export YOLOv8 models, use the following Python script:

```python
from ultralytics import YOLO

# Load a YOLOv8 model
model = YOLO("yolov8n.pt")

# Export the model
model.export(format="onnx", opset=12, simplify=True, dynamic=False, imgsz=640)
```

Alternatively, you can use the following command for exporting the model in the terminal

```bash
yolo export model=yolov8n.pt opset=12 simplify=True dynamic=False format=onnx imgsz=640,640
```

## Exporting YOLOv8 FP16 Models 📦

```python
import onnx
from onnxconverter_common import float16

model = onnx.load(R"YOUR_ONNX_PATH")
model_fp16 = float16.convert_float_to_float16(model)
onnx.save(model_fp16, R"YOUR_FP16_ONNX_PATH")
```

## Download COCO.yaml file 📂

In order to run example, you also need to download coco.yaml. You can download the file manually from [here](https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/coco.yaml)

## Dependencies ⚙️

| Dependency                       | Version       |
| -------------------------------- | ------------- |
| Onnxruntime(linux,windows,macos) | >=1.14.1      |
| OpenCV                           | >=4.0.0       |
| C++ Standard                     | >=17          |
| Cmake                            | >=3.5         |
| Cuda (Optional)                  |  =12.8        |
| cuDNN (Cuda required)            | =9            |
| TensorRT (Optional backend)      | 10.x          |

Note: The dependency on C++17 is due to the usage of the C++17 filesystem feature.

Note (2): Due to ONNX Runtime, we need to use CUDA 12.8 and cuDNN 9. Keep in mind that this requirement might change in the future.

## TensorRT Backend (Optional) ⚡

The TensorRT backend is enabled by default when the CUDA toolkit is found, and requires the `YOLOs-CPP-TensorRT` submodule.

1. Initialize the submodule (only needed once per clone):

   ```bash
   git submodule update --init --recursive
   ```

2. Build the package. Without the CUDA toolkit, it falls back to an ONNX-only build. Optionally set the CUDA architectures of your GPU:

   ```bash
   colcon build --packages-select yolo_onnx_ros \
     --cmake-args -DCMAKE_CUDA_ARCHITECTURES=<your-gpu-architecture>
   ```

3. TensorRT inference requires a serialized `.trt`/`.engine` file (not the `.onnx` file) and a `coco.names` labels file placed next to it, mirroring the `coco.yaml` used for the ONNX backend. Without a labels file, class names stay empty even though detections still work.

Set `-Dyolo_onnx_ros_TENSORRT_ENABLED=OFF` to build an ONNX-only package, even when the CUDA toolkit is available.

## Build 🛠️

1. Clone the repository to your local machine.

2. Navigate to the root directory of the repository.

3. Source ROS 2 Jazzy and build the package with colcon:

   ```console
   source /opt/ros/jazzy/setup.bash
   colcon build --packages-select yolo_onnx_ros
   ```

4. Source the workspace:

   ```console
   source install/setup.bash
   ```

5. The built executable is installed in the workspace.

## Usage 🚀

The package builds a standalone `yolo_onnx_ros_node` executable for testing outside of ROS:

```bash
./yolo_onnx_ros_node <model_path> <images_dir> [onnx|tensorrt]
```

The backend argument is optional and defaults to `onnx`. Use `tensorrt` only if the package was built with the TensorRT backend.

To run the detector from your own C++ application:
```c++
#include "yolo_onnx_ros/detection.hpp"

YoloWrapper wrapper;
DlInitParam params;
std::tie(wrapper, params) = initialize("yolov8n.onnx", YOLO::Backend::ONNX);

cv::Mat img = cv::imread("image.jpg");
std::vector<DlResult> results = detector(wrapper, img);
```

Pass `YOLO::Backend::TENSORRT` instead to use the TensorRT backend (see [TensorRT Backend (Optional)](#tensorrt-backend-optional-)) with a `.trt`/`.engine` model path.

The ROS 2 migration renamed the public API to the tue-robotics naming style: `YOLO_V8` is now `YoloV8`,
`DL_INIT_PARAM`/`DL_RESULT` are now `DlInitParam`/`DlResult`, the functions are `camelBack`, and `ModelType` is a
scoped enum. `sam_onnx_ros` renamed its API the same way.
[docs/api_changes_lint_cleanup.md](docs/api_changes_lint_cleanup.md) lists every change in both packages with its
replacement, plus a `sed` script for updating dependent packages such as `ed_sensor_integration`.
