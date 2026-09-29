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

The TensorRT backend is disabled by default and requires the `YOLOs-CPP-TensorRT` submodule.

1. Initialize the submodule (only needed once per clone):

   ```bash
   git submodule update --init --recursive
   ```

2. Enable TensorRT at configure time. Both the CMake option and the matching environment variable must be set, since `package.xml` only declares `tensorrt_ros` as a dependency when the environment variable is present:

   ```bash
   export YOLO_ONNX_ROS_ENABLE_TENSORRT=true
   catkin config --cmake-args -DYOLO_ONNX_ROS_ENABLE_TENSORRT=ON -DCMAKE_CUDA_ARCHITECTURES=<your-gpu-architecture>
   catkin build yolo_onnx_ros
   ```

3. TensorRT inference requires a serialized `.trt`/`.engine` file (not the `.onnx` file) and a `coco.names` labels file placed next to it, mirroring the `coco.yaml` used for the ONNX backend. Without a labels file, class names stay empty even though detections still work.

Leaving `YOLO_ONNX_ROS_ENABLE_TENSORRT` unset (or `OFF`) builds an ONNX-only package with no TensorRT dependency.

## Build 🛠️

1. Clone the repository to your local machine.

2. Navigate to the root directory of the repository.

3. Create a build directory and navigate to it:

   ```console
   mkdir build && cd build
   ```

4. Run CMake to generate the build files:

   ```console
   cmake ..
   ```

   **Notice**:

   If you encounter an error indicating that the `ONNXRUNTIME_ROOT` variable is not set correctly, you can resolve this by building the project using the appropriate command tailored to your system.

   ```console
   # compiled in a linux system
   cmake -D LINUX=TRUE ..
   ```

5. Build the project:

   ```console
   make
   ```

6. The built executable should now be located in the `build` directory.

## Usage 🚀

The package builds a standalone `yolo_onnx_ros_node` executable for testing outside of ROS:

```bash
./yolo_onnx_ros_node <model_path> <images_dir> [onnx|tensorrt]
```

The backend argument is optional and defaults to `onnx`. Use `tensorrt` only if the package was built with `YOLO_ONNX_ROS_ENABLE_TENSORRT=ON`.

To run the detector from your own C++ application:
```c++
#include "yolo_onnx_ros/detection.hpp"

YoloWrapper wrapper;
DL_INIT_PARAM params;
std::tie(wrapper, params) = Initialize("yolov8n.onnx", YOLO::Backend::kOnnx);

cv::Mat img = cv::imread("image.jpg");
std::vector<DL_RESULT> results = Detector(wrapper, img);
```

Pass `YOLO::Backend::kTensorRT` instead to use the TensorRT backend (see [TensorRT Backend (Optional)](#tensorrt-backend-optional-)) with a `.trt`/`.engine` model path.
