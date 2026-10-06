# YOLO to TensorRT Quick Reference

## 🚀 Quick Start

### 1. Export YOLO to ONNX
```python
from ultralytics import YOLO
model = YOLO('yolo11n.pt')
model.export(format='onnx', imgsz=640, half=False, simplify=True, opset=11)
```

### 2. Convert to TensorRT
```bash
cd scripts
python3 convert_to_tensorrt.py --onnx yolo11n.onnx --output yolo11n_fp16.trt --fp16
```

### 3. Test Model
```bash
cd ..
./build.sh
./build/image_inference
```

## 📋 Common Commands

### Basic Conversions
```bash
# FP32 (default)
python3 convert_to_tensorrt.py --onnx model.onnx --output model.trt

# FP16 (recommended)
python3 convert_to_tensorrt.py --onnx model.onnx --output model_fp16.trt --fp16

# INT8 (requires calibration)
python3 convert_to_tensorrt.py --onnx model.onnx --output model_int8.trt --int8 --calibration-data ./data
```

### INT8 Calibration
```bash
# Generate calibration data
python3 generate_calibration_data.py --copy-from ../data --output ./calibration_data --max-copy 100

# Conservative INT8 (better accuracy)
python3 convert_to_tensorrt.py --onnx model.onnx --output model_int8_conservative.trt --int8 --calibration-data ./calibration_data --calibration-batch-size 8
```

### Batch Operations
```bash
# Convert all ONNX files in directory
python3 convert_to_tensorrt.py --convert-all --models-dir ../models --fp16 --clear-cache
```

## 🎯 Model-Specific Examples

### YOLOv8n
```bash
# Export
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', imgsz=640, half=False, simplify=True)"

# Convert to FP16
python3 convert_to_tensorrt.py --onnx yolov8n.onnx --output yolov8n_fp16.trt --fp16 --clear-cache
```

### YOLOv11n
```bash
# Export
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt').export(format='onnx', imgsz=640, half=False, simplify=True)"

# Convert to FP16
python3 convert_to_tensorrt.py --onnx yolo11n.onnx --output yolo11n_fp16.trt --fp16 --clear-cache
```

### YOLOv12n
```bash
# Export
python -c "from ultralytics import YOLO; YOLO('yolo12n.pt').export(format='onnx', imgsz=640, half=False, simplify=True)"

# Convert to FP16
python3 convert_to_tensorrt.py --onnx yolo12n.onnx --output yolo12n_fp16.trt --fp16 --clear-cache
```

## 🔧 Testing & Validation

### Model Loading Test
```bash
python3 quick_int8_test.py
```

### Performance Comparison
```bash
python3 compare_all_precisions.py fp32_model.trt fp16_model.trt int8_model.trt
```

### Accuracy Testing
```bash
python3 test_int8_accuracy.py --fp32 fp32.trt --fp16 fp16.trt --int8 int8.trt --test-image ../data/test.jpg
```

## 📊 Expected Performance

| Model | FP32 | FP16 | INT8 |
|-------|------|------|------|
| **YOLOv8n** | 6.2 MB | 6.3 MB | 6.2 MB |
| **YOLOv11n** | 8.5 MB | 8.6 MB | 8.5 MB |
| **YOLOv12n** | 9.1 MB | 9.2 MB | 9.1 MB |

| Precision | Accuracy | Performance | Use Case |
|-----------|----------|-------------|----------|
| **FP32** | 100% | Baseline | Development |
| **FP16** | 85-95% | ~2x faster | Production |
| **INT8** | 60-80% | ~4x faster | Edge devices |

## 🛠️ Troubleshooting

### Common Issues
```bash
# Clear cache
python3 convert_to_tensorrt.py --onnx model.onnx --output model.trt --clear-cache

# Force conversion
python3 convert_to_tensorrt.py --onnx model.onnx --output model.trt --force

# Verbose logging
python3 convert_to_tensorrt.py --onnx model.onnx --output model.trt --verbose
```

### INT8 Issues
```bash
# More calibration data
python3 generate_calibration_data.py --copy-from ../data --output ./calibration_data --max-copy 200

# Smaller batch size
python3 convert_to_tensorrt.py --onnx model.onnx --output model_int8.trt --int8 --calibration-data ./calibration_data --calibration-batch-size 4
```

## 📁 File Structure
```
YOLOs-TensorRT-CPP/
├── models/                 # ONNX and TRT models
├── scripts/               # Conversion tools
│   ├── convert_to_tensorrt.py
│   ├── generate_calibration_data.py
│   ├── compare_all_precisions.py
│   └── test_int8_accuracy.py
├── src/                   # C++ inference code
└── data/                  # Test images
```

## 🎯 Best Practices

### For Production
- Use **FP16** for best accuracy/performance balance
- Test thoroughly on representative data
- Monitor accuracy metrics

### For Edge Devices
- Use **INT8** if accuracy is acceptable
- Use conservative calibration settings
- Consider model size constraints

### For Development
- Use **FP32** for debugging
- Compare with original model accuracy
- Validate all precision modes

## 📞 Quick Help

### Check System
```bash
# CUDA
nvidia-smi

# TensorRT
python -c "import tensorrt as trt; print(trt.__version__)"

# Ultralytics
python -c "import ultralytics; print(ultralytics.__version__)"
```

### Model Info
```bash
# ONNX model info
python -c "import onnx; model = onnx.load('model.onnx'); print(f'Inputs: {[i.name for i in model.graph.input]}'); print(f'Outputs: {[o.name for o in model.graph.output]}')"

# TRT model info
python3 quick_int8_test.py
```

## 🚀 Complete Workflow Example

```bash
# 1. Export YOLOv11n to ONNX
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt').export(format='onnx', imgsz=640, half=False, simplify=True, opset=11)"

# 2. Generate calibration data
cd scripts
python3 generate_calibration_data.py --copy-from ../data --output ./calibration_data --max-copy 100

# 3. Convert to FP16 (recommended)
python3 convert_to_tensorrt.py --onnx yolo11n.onnx --output yolo11n_fp16.trt --fp16 --clear-cache

# 4. Convert to INT8 (alternative)
python3 convert_to_tensorrt.py --onnx yolo11n.onnx --output yolo11n_int8.trt --int8 --calibration-data ./calibration_data --calibration-batch-size 8 --clear-cache

# 5. Test models
python3 compare_all_precisions.py yolo11n.trt yolo11n_fp16.trt yolo11n_int8.trt

# 6. Use in C++
cd ..
# Update model path in src/image_inference.cpp
./build.sh
./build/image_inference
``` 