# Development Guide

Architecture overview, extending YOLOs-TRT, and debugging.

## Architecture

```
include/yolos/
├── core/                         # Shared utilities
│   ├── trt_session_base.hpp     # TensorRT engine, buffers, CUDA graphs, async inference
│   ├── trt_utils.hpp            # CUDA macros, TRT logger, RAII wrappers
│   ├── cuda_preprocessing.cu    # GPU letterbox + BGR-to-RGB + normalize kernel
│   ├── cuda_preprocessing.hpp   # Host-side kernel launch API
│   ├── preprocessing.hpp        # CPU fallback preprocessing
│   ├── types.hpp                # Detection, Segmentation, Pose types
│   ├── nms.hpp                  # Non-maximum suppression
│   ├── drawing.hpp              # Visualization
│   ├── version.hpp              # YOLO version detection
│   └── utils.hpp                # Helper functions
├── tasks/                        # Task implementations
│   ├── detection.hpp            # YOLODetector (v5-v12, v26, NAS)
│   ├── segmentation.hpp         # YOLOSegDetector
│   ├── pose.hpp                 # YOLOPoseDetector
│   ├── obb.hpp                  # YOLOOBBDetector
│   └── classification.hpp       # YOLOClassifier
└── yolos.hpp                    # Master include
```

## Core Components

### TrtSessionBase (`trt_session_base.hpp`)

The central abstraction for TensorRT inference:

```cpp
class TrtSessionBase {
    // Engine loading and deserialization
    void loadEngine(const std::string& enginePath);
    
    // Primary inference pipeline (GPU preprocessing + TRT)
    void inferGpu(const cv::Mat& image);
    
    // Fallback: CPU-preprocessed blob inference
    void infer(const float* blob, size_t count);
    
    // Output access
    const float* getOutputData(size_t idx) const;
    std::vector<int64_t> getOutputShape(size_t idx) const;
    
    // Cached preprocessing info for postprocessing
    float getCachedScale() const;
    float getCachedPadX() const;
    float getCachedPadY() const;
};
```

Key features:
- **CUDA graph capture**: `captureInferenceGraph()` captures the `enqueueV3` call into a replayable graph
- **Pinned staging buffers**: `PinnedBuffer` and `DeviceBuffer` for async H2D/D2H
- **10-iteration warm-up**: Lets TRT autotuner converge
- **Single-stream execution**: All GPU ops on one `cudaStream_t`, single sync point

### CUDA Preprocessing (`cuda_preprocessing.cu`)

Single-kernel pipeline:

```
Input:  uint8_t* BGR image on device (raw pixels)
Output: float*   NCHW tensor on device (TRT input buffer)

Operations (fused in one kernel):
  1. Bilinear letterbox resize to model input size
  2. BGR -> RGB channel reorder
  3. Divide by 255.0 (normalize to [0,1])
  4. Write in NCHW layout
```

### NMS (`nms.hpp`)

```cpp
// Class-aware batched NMS
std::vector<int> indices;
yolos::nms::NMSBoxesFBatched(
    boxes, scores, classIds,
    confThreshold, iouThreshold,
    indices
);
```

### Drawing (`drawing.hpp`)

```cpp
yolos::drawing::drawBoundingBox(image, box, label, color);
yolos::drawing::drawMask(image, mask, color, alpha);
yolos::drawing::drawKeypoints(image, keypoints);
```

## Adding a New YOLO Version

### Step 1: Update Version Enum

```cpp
// include/yolos/core/version.hpp
enum class YOLOVersion {
    V5, V6, V7, V8, V9, V10, V11, V12, V26,
    VNew  // Add your version
};
```

### Step 2: Implement Postprocessing

```cpp
// include/yolos/tasks/detection.hpp
void postprocessVNew(
    const float* output, const std::vector<int64_t>& shape,
    float invScale, float padX, float padY,
    float confThreshold, float iouThreshold,
    std::vector<Detection>& results
) {
    // Parse model output tensor
    // Apply NMS
    // Descale coordinates
}
```

### Step 3: Update Factory

```cpp
switch (version) {
    case YOLOVersion::VNew:
        return postprocessVNew(...);
}
```

### Step 4: Add Auto-Detection Logic

```cpp
// In version.hpp — detect from output tensor shape
if (outputShape matches VNew pattern) {
    return YOLOVersion::VNew;
}
```

### Step 5: Add Tests

Add model to the test suite and comparison scripts.

## TensorRT Integration Details

### Engine Loading

```cpp
// Read serialized engine file
std::vector<char> engineData = readFile("model.trt");

// Create runtime and deserialize
auto runtime = nvinfer1::createInferRuntime(logger);
auto engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
auto context = engine->createExecutionContext();
```

### Buffer Management

```cpp
// Pre-allocate device buffers for all I/O tensors
for (int i = 0; i < engine->getNbIOTensors(); i++) {
    auto name = engine->getIOTensorName(i);
    auto dims = engine->getTensorShape(name);
    auto dtype = engine->getTensorDataType(name);
    
    size_t bytes = volume(dims) * sizeof(float);
    DeviceBuffer devBuf(bytes);
    PinnedBuffer hostBuf(bytes);
    
    context->setTensorAddress(name, devBuf.ptr());
}
```

### CUDA Graph Capture

```cpp
// Capture inference into a graph (once, after warm-up)
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
context->enqueueV3(stream);
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

// Replay for every subsequent frame (much faster)
cudaGraphLaunch(graphExec, stream);
```

## Memory Architecture

```
Host (CPU)                          Device (GPU)
-----------                         -------------
cv::Mat (pageable)
   |
   v
pinnedSrc_ (PinnedBuffer)  --H2D-->  deviceSrc_ (DeviceBuffer)
                                        |
                                        | letterboxNormalizeKernel
                                        v
                                      inputTensors_[0].deviceBuf  (TRT input)
                                        |
                                        | enqueueV3 / cudaGraphLaunch
                                        v
                                      outputTensors_[i].deviceBuf (TRT output)
                                        |
                                   D2H  |
                                        v
outputTensors_[i].hostBuf  <----------
(PinnedBuffer)
   |
   v
postprocess (CPU)
```

## Debugging

### Enable TensorRT Verbose Logging

```cpp
// In trt_utils.hpp — change log level
TrtLogger logger(nvinfer1::ILogger::Severity::kVERBOSE);
```

### Profile Inference

```cpp
#include <chrono>

auto start = std::chrono::high_resolution_clock::now();
auto detections = detector.detect(frame);
auto end = std::chrono::high_resolution_clock::now();

auto ms = std::chrono::duration<double, std::milli>(end - start).count();
std::cout << "Inference: " << ms << " ms" << std::endl;
```

### Validate Against Python

```bash
cd tests
./test_detection.sh
```

### CUDA Error Debugging

```bash
# Run with CUDA memcheck
compute-sanitizer ./image_inference model.trt data/dog.jpg models/coco.names

# Enable CUDA debug sync
export CUDA_LAUNCH_BLOCKING=1
./image_inference model.trt data/dog.jpg models/coco.names
```

## Code Style

- **C++17** standard
- **snake_case** for variables and functions
- **PascalCase** for classes and types
- **UPPER_CASE** for constants and macros
- Use `const` and `[[nodiscard]]` where appropriate
- RAII for all GPU resources (no raw `cudaMalloc`/`cudaFree`)

## Building Tests

```bash
cd tests
./build_test.sh 0  # Detection
./build_test.sh 1  # Classification
./build_test.sh 2  # Segmentation
./build_test.sh 3  # Pose
./build_test.sh 4  # OBB
./build_test.sh 5  # All
```

## Benchmarking

```bash
cd benchmarks
./auto_bench.sh
```

## Next Steps

- [Contributing](contributing.md) -- Submit changes
- [Model Guide](models.md) -- Model compatibility
