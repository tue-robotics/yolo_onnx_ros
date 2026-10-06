# YOLO Unified Benchmark Tool

A comprehensive benchmarking tool that combines **performance metrics** (FPS, latency, memory) and **accuracy metrics** (mAP, AP50) for YOLO models running on **TensorRT**.

## Latest Results

Measured on **NVIDIA RTX 2000 Ada Laptop GPU** with YOLOv11n (640x640, 1000 iterations, 10-iteration warm-up):

| Precision | FPS | Avg Latency | P50 Latency | P99 Latency | Peak GPU Memory |
|:---------:|:---:|:-----------:|:-----------:|:-----------:|:---------------:|
| **FP32**  | 466 | 2.144 ms    | 2.040 ms    | 3.028 ms    | 529.5 MB        |
| **FP16**  | 479 | 2.086 ms    | 1.976 ms    | 2.913 ms    | 535.7 MB        |
| **INT8**  | 530 | 1.886 ms    | 1.775 ms    | 2.701 ms    | 443.7 MB        |

### Optimizations Active

- GPU-accelerated preprocessing (single CUDA kernel: letterbox + BGR-to-RGB + normalize)
- CUDA graph capture for inference replay
- Pinned staging buffers for async H2D transfers
- 10-iteration warm-up for TRT autotuner convergence
- Zero CPU preprocessing in the hot path

## Features

### Holistic Benchmarking
- **Performance Metrics**: FPS, latency (min/max/avg/P50/P99), GPU memory
- **Accuracy Metrics**: mAP, AP50, AP50-95 (when ground truth available)
- **Unified Output**: CSV files + tabular terminal reports

### Supported Modes
- `quick`: Quick single-engine performance benchmark
- `image`: Single image performance benchmarking
- `video`: Video file performance benchmarking
- `camera`: Real-time camera performance benchmarking
- `evaluate`: Accuracy evaluation on dataset with ground truth
- `comprehensive`: Automated multi-model/config testing

### Task Support
- **Detection**: Object detection (YOLOv5-v12, YOLO26)
- **Segmentation**: Instance segmentation
- **Pose**: Human pose estimation
- **OBB**: Oriented bounding box detection
- **Classification**: Image classification

## Prerequisites

| Component | Requirement |
|-----------|-------------|
| NVIDIA GPU | Compute Capability >= 7.5 |
| CUDA Toolkit | >= 12.0 |
| TensorRT | >= 10.0 |
| OpenCV | >= 4.5 |
| CMake | >= 3.18 |
| C++ Compiler | C++17 (GCC 9+) |

### Models and Data
- **Engines**: TensorRT `.trt` files in `../models/`
- **Labels**: `../models/coco.names`
- **Test data**: `../data/` (e.g., `dog.jpg`, `dogs.mp4`)

## Building

### Quick Build (Recommended)

```bash
cd benchmarks
./auto_bench.sh
```

### Manual Build

```bash
cd benchmarks
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Usage

### Quick Performance Test

```bash
# Run from project root or benchmarks/build/
./yolo_unified_benchmark quick \
    --engine ../models/yolo11n.trt \
    --labels ../models/coco.names \
    --task detect
```

### Single Image Benchmark

```bash
./yolo_unified_benchmark image yolo11 detection \
    models/yolo11n.trt models/coco.names data/dog.jpg \
    --iterations=1000
```

### Video Benchmark

```bash
./yolo_unified_benchmark video yolo11 detection \
    models/yolo11n.trt models/coco.names data/dogs.mp4
```

### Accuracy Evaluation (Requires Ground Truth)

```bash
./yolo_unified_benchmark evaluate yolo11 detection \
    models/yolo11n.trt models/coco.names \
    ../val2017 ../labels_val2017 \
    --dataset-type=coco
```

### Comprehensive Multi-Model Test

```bash
./yolo_unified_benchmark comprehensive
```

## Command-Line Options

### General Options
- `--iterations=N`: Number of iterations for image mode (default: 100)
- `--duration=N`: Duration in seconds for video/camera (default: 30)
- `--conf-threshold=N`: Confidence threshold (default: 0.4)
- `--nms-threshold=N`: NMS threshold (default: 0.7)

### Evaluation Options
- `--eval-conf-threshold=N`: Confidence for evaluation (default: 0.001)
- `--dataset-type=coco|custom`: Dataset type (default: custom)

## Output

### CSV Output
Results saved to `results/unified_benchmark_TIMESTAMP.csv` with columns:

| Column | Description |
|--------|-------------|
| `model_type` | Model identifier |
| `task_type` | Task type (detection, segmentation, ...) |
| `environment` | Runtime (TensorRT) |
| `precision` | Engine precision (fp32, fp16, int8) |
| `fps` | Frames per second |
| `latency_avg_ms` | Average latency |
| `latency_p50_ms` | P50 latency |
| `latency_p99_ms` | P99 latency |
| `gpu_memory_mb` | Peak GPU memory |
| `AP50` | Average Precision @ IoU 0.5 |
| `mAP50-95` | Mean Average Precision |

### Terminal Output

```
================================================================================
BENCHMARK REPORT
================================================================================
Model: yolo11n (detection)
Device: GPU (TensorRT)
Engine: models/yolo11n.trt
--------------------------------------------------------------------------------

PERFORMANCE METRICS:
  FPS:              466.23
  Latency:          2.144 ms (avg), 2.040 ms (P50), 3.028 ms (P99)
  Peak GPU Memory:  529.5 MB
  Frames Processed: 1000
================================================================================
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Engine not found" | Convert ONNX to TRT: `trtexec --onnx=model.onnx --saveEngine=model.trt --fp16` |
| "CUDA error" | Check `nvidia-smi`, ensure GPU is available |
| "Low FPS on first run" | Warm-up is automatic (10 iterations). Re-run for stable results. |
| Build fails | Verify TensorRT and CUDA toolkit are installed |

## License

Same as YOLOs-TRT project (AGPL-3.0).
