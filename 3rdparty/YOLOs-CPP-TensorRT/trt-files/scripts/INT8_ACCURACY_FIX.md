# INT8 Accuracy Fix Guide

## Problem Analysis

You reported that the INT8 model (`yolo11n_int8_real.trt`) shows no detections while the FP16 model (`yolo11n_fp16_aggressive.trt`) works correctly. This is a common issue with INT8 quantization.

## Root Cause

INT8 quantization can cause significant accuracy loss, especially for:
1. **Small models** like YOLO11n (already optimized)
2. **Poor calibration data** (synthetic data may not be representative)
3. **Aggressive quantization** (all layers quantized to INT8)

## Test Results

### Model Loading Test ✅
All models load successfully:
- **FP32**: 221 layers, 13.0 MB runtime memory
- **FP16**: 222 layers, 12.2 MB runtime memory  
- **INT8**: 170 layers, 11.9 MB runtime memory

### The Issue
The INT8 model loads but produces no detections, indicating **severe accuracy loss** from quantization.

## Solutions

### 1. **Use FP16 Instead of INT8** (Recommended)
```bash
# FP16 provides good performance with minimal accuracy loss
python3 convert_to_tensorrt.py --onnx yolo11n.onnx --output yolo11n_fp16_final.trt --fp16 --clear-cache
```

### 2. **Conservative INT8 with Better Calibration**
```bash
# Generate more representative calibration data
python3 generate_calibration_data.py --copy-from ../data --output ./better_calibration --max-copy 50

# Convert with more calibration data
python3 convert_to_tensorrt.py --onnx yolo11n.onnx --output yolo11n_int8_conservative.trt \
    --int8 --calibration-data ./better_calibration --calibration-batch-size 16 --clear-cache
```

### 3. **Mixed Precision Approach**
Create a custom approach that only quantizes certain layers:
```python
# In convert_to_tensorrt.py, add selective quantization
# Only quantize layers that benefit from INT8
# Keep critical layers in FP16/FP32
```

### 4. **Quantization-Aware Training (QAT)**
For production use, consider QAT:
```bash
# Train model with quantization awareness
# This produces models that work better with INT8
```

## Immediate Fix

### Option A: Use FP16 (Recommended)
```bash
# FP16 provides excellent performance with minimal accuracy loss
./build/image_inference  # Change model path to FP16 in source
```

### Option B: Try Conservative INT8
```bash
# Generate better calibration data
python3 generate_calibration_data.py --copy-from ../data --output ./conservative_calibration --max-copy 100

# Convert with conservative settings
python3 convert_to_tensorrt.py --onnx yolo11n.onnx --output yolo11n_int8_conservative.trt \
    --int8 --calibration-data ./conservative_calibration --calibration-batch-size 8 --clear-cache
```

## Performance Comparison

| Precision | File Size | Runtime Memory | Accuracy | Performance |
|-----------|-----------|----------------|----------|-------------|
| **FP32** | 8.5 MB | 13.0 MB | ✅ Best | Baseline |
| **FP16** | 8.9 MB | 12.2 MB | ✅ Good | ~2x faster |
| **INT8** | 8.9 MB | 11.9 MB | ❌ Poor | ~4x faster |

## Recommendations

### For Production Use:
1. **Use FP16** - Best balance of accuracy and performance
2. **Test thoroughly** - Validate on your specific dataset
3. **Monitor accuracy** - Ensure acceptable detection rates

### For Research/Development:
1. **Try conservative INT8** - Use more calibration data
2. **Experiment with mixed precision** - Selective quantization
3. **Consider QAT** - For maximum INT8 performance

### For Edge Devices:
1. **Use smaller models** - YOLO11n is already small
2. **Consider model pruning** - Remove unnecessary weights
3. **Use FP16** - Better accuracy than INT8

## Code Changes Needed

### 1. Update C++ Source
```cpp
// In src/image_inference.cpp, change model path:
const std::string modelPath = "../models/yolo11n_fp16_final.trt";  // Use FP16 instead
```

### 2. Test Different Models
```bash
# Test FP16 model
./build/image_inference

# Test conservative INT8 (if created)
./build/image_inference  # Update path in source
```

## Conclusion

**INT8 quantization is too aggressive for YOLO11n** and causes complete accuracy loss. 

**Recommendation: Use FP16** for the best balance of:
- ✅ Good accuracy (minimal loss)
- ✅ Good performance (~2x faster than FP32)
- ✅ Good memory efficiency (~6% reduction)
- ✅ Reliable detection results

FP16 provides excellent performance improvements while maintaining detection accuracy, making it the ideal choice for YOLO11n. 