# Model Quantization Guide

Optimize TensorRT engines for faster inference and smaller memory footprint.

## Precision Options

| Format | Size Reduction | Throughput Gain | Accuracy Impact |
|--------|:--------------:|:---------------:|:---------------:|
| FP32 | Baseline | 1x | Best |
| FP16 | ~50% | 1.5-2x | Negligible |
| INT8 | ~75% | 2-3x | -0.5-2% mAP |

## FP16 (Recommended Default)

FP16 provides near-identical accuracy with significant throughput gains on GPUs with Tensor Cores (Turing and newer).

```bash
# Using trtexec
trtexec --onnx=yolo11n.onnx --saveEngine=yolo11n_fp16.trt --fp16

# Using Python converter
python trt-files/scripts/convert_to_tensorrt.py --onnx yolo11n.onnx --fp16
```

## INT8 Quantization

INT8 requires calibration data for best accuracy. The calibration process measures activation ranges across representative images.

### Step 1: Prepare Calibration Data

Use 100-500 representative images from your deployment domain:

```bash
# Generate calibration images from COCO
python trt-files/scripts/generate_calibration_data.py \
    --count 500 \
    --output calibration_data/
```

Or use existing images:

```bash
# Any folder of images works
ls trt-files/scripts/calibration_data/
# calibration_0000.jpg  calibration_0001.jpg  ...
```

### Step 2: Convert with INT8 Calibration

```bash
# Using Python converter (recommended for INT8)
python trt-files/scripts/convert_to_tensorrt.py \
    --onnx yolo11n.onnx \
    --int8 \
    --calibration-data trt-files/scripts/calibration_data/ \
    --calibration-batch-size 8

# Using trtexec with calibration cache
trtexec --onnx=yolo11n.onnx --saveEngine=yolo11n_int8.trt --int8 \
    --calib=calibration.cache
```

### Step 3: Validate Accuracy

```bash
# Compare INT8 vs FP16 accuracy
python trt-files/scripts/compare_all_precisions.py \
    --onnx yolo11n.onnx \
    --test-images data/
```

## Using Quantized Engines

```cpp
// Same API for all precisions — just load the engine
yolos::det::YOLODetector detector(
    "yolo11n_int8.trt",    // INT8 engine
    "models/coco.names"
);

auto detections = detector.detect(frame);
```

## Benchmarks

Measured on NVIDIA RTX 2000 Ada Laptop GPU (YOLOv11n, 640x640):

| Precision | FPS | P50 Latency | GPU Memory | Relative Accuracy |
|-----------|:---:|:-----------:|:----------:|:------------------:|
| FP32 | 466 | 2.040 ms | 529.5 MB | Baseline |
| FP16 | 479 | 1.976 ms | 535.7 MB | ~Same |
| INT8 | 530 | 1.775 ms | 443.7 MB | -0.5-1% mAP |

## Tips

1. **FP16 first** -- Always try FP16 before INT8. The accuracy loss is typically negligible.
2. **Calibration data matters** -- Use images from your actual deployment domain, not random images.
3. **Calibration count** -- 100-500 images is sufficient. More doesn't significantly improve accuracy.
4. **Per-channel** -- TensorRT uses per-channel quantization by default, which is more accurate than per-tensor.
5. **Calibration algorithm** -- `MinMax` is default; `Entropy` can be better for some models.
6. **Cache the calibration** -- Calibration is slow. The converter caches results for re-use.
7. **Rebuild on new hardware** -- INT8 calibration is GPU-specific.
8. **TRT handles quantization** -- Unlike ONNX Runtime, TRT does quantization natively during engine build. No need for separate ONNX quantization tools.

## Advanced: DLA on Jetson

On Jetson Xavier/Orin, you can offload INT8 inference to DLA (Deep Learning Accelerator):

```cpp
// Use DLA core 0 with INT8 engine
yolos::det::YOLODetector detector(
    "yolo11n_int8.trt",
    "coco.names",
    yolos::YOLOVersion::Auto,
    /*dlaCore=*/0
);
```

Build the engine with DLA support:

```bash
python trt-files/scripts/convert_to_tensorrt.py \
    --onnx yolo11n.onnx --int8 --dla-core 0 \
    --calibration-data calibration_data/
```

## Next Steps

- [Model Guide](models.md) -- Model export and conversion
- [Benchmarks](../benchmarks/README.md) -- Performance testing
