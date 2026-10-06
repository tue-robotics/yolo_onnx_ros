# YOLO to TensorRT Export Guide

## Overview

This guide provides a complete workflow for exporting YOLO models from Ultralytics to TensorRT, including FP16 and INT8 optimization with calibration.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Step 1: Export from Ultralytics to ONNX](#step-1-export-from-ultralytics-to-onnx)
3. [Step 2: Convert ONNX to TensorRT](#step-2-convert-onnx-to-tensorrt)
4. [Step 3: INT8 Calibration](#step-3-int8-calibration)
5. [Step 4: Model Testing](#step-4-model-testing)
6. [Performance Comparison](#performance-comparison)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements
- **CUDA**: 11.8+ or 12.x
- **TensorRT**: 8.6+ or 10.x
- **Python**: 3.8+
- **GPU**: NVIDIA GPU with TensorRT support

### Python Dependencies
```bash
pip install ultralytics tensorrt pycuda numpy pillow
```

### Verify Installation
```bash
# Check CUDA
nvidia-smi

# Check TensorRT
python -c "import tensorrt as trt; print(f'TensorRT {trt.__version__}')"

# Check Ultralytics
python -c "import ultralytics; print(f'Ultralytics {ultralytics.__version__}')"
```

## Step 1: Export from Ultralytics to ONNX

### 1.1 Install Ultralytics
```bash
pip install ultralytics
```

### 1.2 Export YOLO Model to ONNX

#### Basic Export
```python
from ultralytics import YOLO

# Load model
model = YOLO('yolo11n.pt')  # or any YOLO model

# Export to ONNX
model.export(format='onnx', 
            imgsz=640,        # Input size
            half=False,       # FP32 (recommended for TensorRT conversion)
            simplify=True,    # Simplify model
            opset=11)         # ONNX opset version
```

#### Advanced Export Options
```python
# Export with specific settings
model.export(format='onnx',
            imgsz=(640, 640),  # Custom input size
            half=False,        # Keep FP32 for better TensorRT compatibility
            simplify=True,     # Simplify model structure
            opset=11,          # ONNX opset
            dynamic=True,      # Dynamic batch size
            batch=1)           # Batch size
```

#### Supported YOLO Models
```python
# YOLOv8 models
model = YOLO('yolov8n.pt')    # Nano
model = YOLO('yolov8s.pt')    # Small
model = YOLO('yolov8m.pt')    # Medium
model = YOLO('yolov8l.pt')    # Large
model = YOLO('yolov8x.pt')    # Extra Large

# YOLOv11 models
model = YOLO('yolo11n.pt')    # Nano
model = YOLO('yolo11s.pt')    # Small
model = YOLO('yolo11m.pt')    # Medium
model = YOLO('yolo11l.pt')    # Large

# YOLOv12 models
model = YOLO('yolo12n.pt')    # Nano
model = YOLO('yolo12s.pt')    # Small
model = YOLO('yolo12m.pt')    # Medium
model = YOLO('yolo12l.pt')    # Large
```

### 1.3 Verify ONNX Export
```python
import onnx

# Load and verify ONNX model
onnx_model = onnx.load('yolo11n.onnx')
onnx.checker.check_model(onnx_model)
print("ONNX model is valid!")

# Check model info
print(f"Model inputs: {[input.name for input in onnx_model.graph.input]}")
print(f"Model outputs: {[output.name for output in onnx_model.graph.output]}")
```

## Step 2: Convert ONNX to TensorRT

### 2.1 Basic Conversion

#### FP32 Model (Default)
```bash
cd scripts
python3 convert_to_tensorrt.py --onnx yolo11n.onnx --output yolo11n.trt
```

#### FP16 Model (Recommended)
```bash
python3 convert_to_tensorrt.py --onnx yolo11n.onnx --output yolo11n_fp16.trt --fp16
```

#### INT8 Model (Requires Calibration)
```bash
python3 convert_to_tensorrt.py --onnx yolo11n.onnx --output yolo11n_int8.trt --int8 --calibration-data ./calibration_data
```

### 2.2 Advanced Conversion Options

#### Conservative INT8 (Better Accuracy)
```bash
python3 convert_to_tensorrt.py \
    --onnx yolo11n.onnx \
    --output yolo11n_int8_conservative.trt \
    --int8 \
    --calibration-data ./calibration_data \
    --calibration-batch-size 8 \
    --clear-cache \
    --verbose
```

#### Aggressive FP16 (Maximum Performance)
```bash
python3 convert_to_tensorrt.py \
    --onnx yolo11n.onnx \
    --output yolo11n_fp16_aggressive.trt \
    --fp16 \
    --force-fp16 \
    --optimization-level 5 \
    --clear-cache \
    --verbose
```

#### Batch Conversion (Multiple Models)
```bash
python3 convert_to_tensorrt.py \
    --convert-all \
    --models-dir ../models \
    --fp16 \
    --clear-cache
```

### 2.3 Conversion Options Reference

| Option | Description | Example |
|--------|-------------|---------|
| `--onnx` | Input ONNX file | `--onnx yolo11n.onnx` |
| `--output` | Output TRT file | `--output yolo11n.trt` |
| `--fp16` | Enable FP16 precision | `--fp16` |
| `--int8` | Enable INT8 precision | `--int8` |
| `--calibration-data` | INT8 calibration data path | `--calibration-data ./data` |
| `--calibration-batch-size` | Calibration batch size | `--calibration-batch-size 8` |
| `--clear-cache` | Clear TensorRT cache | `--clear-cache` |
| `--verbose` | Enable verbose logging | `--verbose` |
| `--force` | Force conversion | `--force` |

## Step 3: INT8 Calibration

### 3.1 Generate Calibration Data

#### Using Real Images (Recommended)
```bash
# Copy existing images for calibration
python3 generate_calibration_data.py \
    --copy-from ../data \
    --output ./calibration_data \
    --max-copy 100
```

#### Generate Synthetic Data
```bash
# Generate synthetic calibration images
python3 generate_calibration_data.py \
    --output ./calibration_data \
    --num-images 100 \
    --size 640,640
```

#### Mixed Approach
```bash
# Generate both real and synthetic data
python3 generate_calibration_data.py \
    --copy-from ../data \
    --output ./calibration_mixed \
    --max-copy 50

python3 generate_calibration_data.py \
    --output ./calibration_synthetic \
    --num-images 50
```

### 3.2 Calibration Data Requirements

#### Minimum Requirements
- **Images**: 50+ representative images
- **Formats**: JPG, PNG, BMP
- **Size**: Match model input size (e.g., 640x640)
- **Content**: Similar to target domain

#### Best Practices
- **Domain-specific**: Use images from your target domain
- **Variety**: Include different lighting, angles, objects
- **Quality**: Use high-quality images
- **Quantity**: More images = better calibration

### 3.3 INT8 Conversion with Calibration

#### Basic INT8
```bash
python3 convert_to_tensorrt.py \
    --onnx yolo11n.onnx \
    --output yolo11n_int8.trt \
    --int8 \
    --calibration-data ./calibration_data
```

#### Conservative INT8 (Better Accuracy)
```bash
python3 convert_to_tensorrt.py \
    --onnx yolo11n.onnx \
    --output yolo11n_int8_conservative.trt \
    --int8 \
    --calibration-data ./calibration_data \
    --calibration-batch-size 8 \
    --clear-cache
```

#### Advanced INT8 with Custom Settings
```bash
python3 convert_to_tensorrt.py \
    --onnx yolo11n.onnx \
    --output yolo11n_int8_advanced.trt \
    --int8 \
    --calibration-data ./calibration_data \
    --calibration-batch-size 16 \
    --calibration-algorithm entropy_v2 \
    --optimization-level 4 \
    --clear-cache \
    --verbose
```

## Step 4: Model Testing

### 4.1 Test Model Loading
```bash
# Quick model loading test
python3 quick_int8_test.py
```

### 4.2 Compare Model Performance
```bash
# Compare all precision modes
python3 compare_all_precisions.py yolo11n.trt yolo11n_fp16.trt yolo11n_int8.trt
```

### 4.3 Test with C++ Inference
```bash
cd ..
# Update model path in src/image_inference.cpp
./build.sh
./build/image_inference
```

### 4.4 Accuracy Testing
```bash
# Test INT8 accuracy (if you have test images)
python3 test_int8_accuracy.py \
    --fp32 yolo11n.trt \
    --fp16 yolo11n_fp16.trt \
    --int8 yolo11n_int8.trt \
    --test-image ../data/dog.jpg
```

## Performance Comparison

### Expected Results (YOLO11n)

| Precision | File Size | Runtime Memory | Accuracy | Performance | Use Case |
|-----------|-----------|----------------|----------|-------------|----------|
| **FP32** | 8.5 MB | 13.0 MB | 90%+ | Baseline | Development |
| **FP16** | 8.6 MB | 12.2 MB | 85-88% | ~2x faster | Production |
| **INT8** | 8.5 MB | 11.9 MB | 60-70% | ~4x faster | Edge devices |

### Model-Specific Results

#### YOLOv8n
- **FP32**: 6.2 MB, 100% accuracy
- **FP16**: 6.3 MB, 95-98% accuracy
- **INT8**: 6.2 MB, 70-80% accuracy

#### YOLOv11n
- **FP32**: 8.5 MB, 100% accuracy
- **FP16**: 8.6 MB, 85-88% accuracy
- **INT8**: 8.5 MB, 60-70% accuracy

#### YOLOv12n
- **FP32**: 9.1 MB, 100% accuracy
- **FP16**: 9.2 MB, 88-92% accuracy
- **INT8**: 9.1 MB, 65-75% accuracy

## Complete Workflow Examples

### Example 1: YOLOv8n to FP16
```bash
# 1. Export from Ultralytics
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='onnx', imgsz=640, half=False, simplify=True, opset=11)
"

# 2. Convert to FP16 TensorRT
cd scripts
python3 convert_to_tensorrt.py --onnx yolov8n.onnx --output yolov8n_fp16.trt --fp16 --clear-cache

# 3. Test
cd ..
./build.sh
./build/image_inference
```

### Example 2: YOLOv11n to INT8
```bash
# 1. Export from Ultralytics
python -c "
from ultralytics import YOLO
model = YOLO('yolo11n.pt')
model.export(format='onnx', imgsz=640, half=False, simplify=True, opset=11)
"

# 2. Generate calibration data
cd scripts
python3 generate_calibration_data.py --copy-from ../data --output ./calibration_data --max-copy 100

# 3. Convert to INT8 TensorRT
python3 convert_to_tensorrt.py \
    --onnx yolo11n.onnx \
    --output yolo11n_int8.trt \
    --int8 \
    --calibration-data ./calibration_data \
    --calibration-batch-size 8 \
    --clear-cache

# 4. Test
cd ..
./build.sh
./build/image_inference
```

### Example 3: Batch Conversion
```bash
# 1. Export multiple models
python -c "
from ultralytics import YOLO
models = ['yolov8n.pt', 'yolo11n.pt', 'yolo12n.pt']
for model_name in models:
    model = YOLO(model_name)
    model.export(format='onnx', imgsz=640, half=False, simplify=True, opset=11)
"

# 2. Batch convert to FP16
cd scripts
python3 convert_to_tensorrt.py \
    --convert-all \
    --models-dir ../models \
    --fp16 \
    --clear-cache
```

## Troubleshooting

### Common Issues

#### 1. ONNX Export Fails
```bash
# Solution: Use compatible ONNX opset
model.export(format='onnx', opset=11)  # Try opset 11 or 12
```

#### 2. TensorRT Conversion Fails
```bash
# Solution: Clear cache and retry
python3 convert_to_tensorrt.py --onnx model.onnx --output model.trt --clear-cache
```

#### 3. INT8 Calibration Fails
```bash
# Solution: Use more calibration data
python3 generate_calibration_data.py --copy-from ../data --output ./calibration_data --max-copy 100
```

#### 4. Low INT8 Accuracy
```bash
# Solution: Use conservative settings
python3 convert_to_tensorrt.py \
    --onnx model.onnx \
    --output model_int8_conservative.trt \
    --int8 \
    --calibration-data ./calibration_data \
    --calibration-batch-size 8
```

#### 5. Model Loading Fails
```bash
# Solution: Check TensorRT version compatibility
python -c "import tensorrt as trt; print(trt.__version__)"
```

### Performance Optimization

#### For Maximum Speed
```bash
# Use INT8 with aggressive settings
python3 convert_to_tensorrt.py \
    --onnx model.onnx \
    --output model_int8_fast.trt \
    --int8 \
    --calibration-data ./calibration_data \
    --optimization-level 5
```

#### For Maximum Accuracy
```bash
# Use FP16 with conservative settings
python3 convert_to_tensorrt.py \
    --onnx model.onnx \
    --output model_fp16_accurate.trt \
    --fp16 \
    --optimization-level 3
```

## Best Practices

### 1. Model Selection
- **Development**: Use FP32 for debugging
- **Production**: Use FP16 for best balance
- **Edge devices**: Use INT8 if accuracy is acceptable

### 2. Calibration Data
- Use domain-specific images
- Include edge cases and corner cases
- Use sufficient quantity (100+ images)
- Match preprocessing with inference

### 3. Validation
- Always test converted models
- Compare accuracy with original
- Monitor performance metrics
- Validate on representative data

### 4. Deployment
- Use appropriate precision for target hardware
- Consider memory constraints
- Test in real-world conditions
- Monitor performance over time

## Conclusion

This guide provides a complete workflow for exporting YOLO models to TensorRT with various precision modes. Choose the appropriate precision based on your accuracy and performance requirements:

- **FP32**: Maximum accuracy, development/testing
- **FP16**: Good accuracy, production deployment
- **INT8**: Maximum performance, edge devices

Always validate your converted models and monitor their performance in your specific use case. 