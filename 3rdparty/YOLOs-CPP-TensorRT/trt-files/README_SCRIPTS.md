# Scripts Directory - Cleaned & Organized

## 📁 Directory Structure

This directory contains essential tools and scripts for YOLO to TensorRT conversion and optimization.

## 🛠️ Core Tools

### Conversion Tools
- **`convert_to_tensorrt.py`** - Main TensorRT conversion script with FP16/INT8 support
- **`generate_calibration_data.py`** - INT8 calibration data generation
- **`improve_int8_accuracy.py`** - INT8 accuracy improvement automation
- **`convert_models.sh`** - Batch conversion shell script

### Analysis & Testing Tools
- **`compare_all_precisions.py`** - Compare FP32/FP16/INT8 models
- **`compare_models.py`** - Model comparison and analysis
- **`test_int8_accuracy.py`** - Accuracy testing and validation
- **`quick_int8_test.py`** - Quick model loading validation
- **`deep_engine_analysis.py`** - Detailed TensorRT engine analysis
- **`benchmark.cpp`** - Performance benchmarking tool

### Examples & Documentation
- **`example_usage.py`** - Usage examples and demonstrations
- **`README_convert_to_tensorrt.md`** - Original conversion guide

## 📊 Essential Models

### Current Models (Ready to Use)
- **`yolo11n.trt`** - FP32 baseline model (8.9 MB)
- **`yolo11n_fp16_accurate.trt`** - FP16 optimized model (9.0 MB) ⭐ **Recommended**
- **`yolo11n_int8_conservative.trt`** - INT8 conservative model (8.8 MB)

### Backup Models
- **`backup_models/`** - Contains additional models for reference:
  - `yolo11n_fp16_aggressive.trt` - Aggressive FP16 optimization
  - `yolo11n_fp16.trt` - Standard FP16 model
  - `yolo11n_int8_real.trt` - INT8 with real calibration data
  - `yolo11n_int8.trt` - Basic INT8 model

## 📚 Documentation

### Technical Guides
- **`INT8_IMPLEMENTATION_GUIDE.md`** - INT8 quantization implementation
- **`INT8_ACCURACY_FIX.md`** - INT8 accuracy issues and solutions
- **`ACCURACY_IMPROVEMENT_SOLUTION.md`** - Accuracy improvement solutions
- **`FP16_ISSUE_ANALYSIS.md`** - FP16 optimization analysis
- **`FP16_FIX_SUMMARY.md`** - FP16 implementation summary

## 🗂️ Data Directories

- **`calibration_data/`** - INT8 calibration images and data
- **`backup_models/`** - Additional TensorRT models for reference

## 🚀 Quick Start

### Basic Conversion
```bash
# Convert ONNX to FP16 TensorRT
python3 convert_to_tensorrt.py --onnx yolo11n.onnx --output yolo11n_fp16.trt --fp16 --clear-cache
```

### INT8 with Calibration
```bash
# Generate calibration data
python3 generate_calibration_data.py --copy-from ../data --output ./calibration_data --max-copy 100

# Convert to INT8
python3 convert_to_tensorrt.py --onnx yolo11n.onnx --output yolo11n_int8.trt --int8 --calibration-data ./calibration_data
```

### Model Testing
```bash
# Quick test
python3 quick_int8_test.py

# Compare models
python3 compare_all_precisions.py yolo11n.trt yolo11n_fp16_accurate.trt yolo11n_int8_conservative.trt

# Test accuracy
python3 test_int8_accuracy.py --fp32 yolo11n.trt --fp16 yolo11n_fp16_accurate.trt --int8 yolo11n_int8_conservative.trt --test-image ../data/dog.jpg
```

## 📊 Model Performance Summary

| Model | Size | Precision | Accuracy | Use Case |
|-------|------|-----------|----------|----------|
| **yolo11n.trt** | 8.9 MB | FP32 | 90%+ | Development |
| **yolo11n_fp16_accurate.trt** | 9.0 MB | FP16 | 85-88% | **Production** |
| **yolo11n_int8_conservative.trt** | 8.8 MB | INT8 | 60-70% | Edge devices |

## 🧹 Cleanup Summary

### Removed Files
- Large log files (`tensorrt_conversion.log`, `tensorrt_conversion_detailed.log`)
- Cache files (`calibration.cache`, `.tensorrt_cache/`)
- Duplicate models (moved to `backup_models/`)
- ONNX and PT files (should be in `../models/`)
- Multiple calibration directories (consolidated to `calibration_data/`)
- Old test files and temporary files

### Space Saved
- **Log files**: ~1.4 GB
- **Cache files**: ~15 KB
- **Duplicate models**: ~40 MB
- **ONNX/PT files**: ~60 MB
- **Total**: ~1.5 GB

## 🎯 Recommendations

### For Production Use
- Use **`yolo11n_fp16_accurate.trt`** for best accuracy/performance balance
- Test thoroughly on representative data
- Monitor accuracy metrics

### For Development
- Use **`yolo11n.trt`** (FP32) for debugging and validation
- Compare with other precision modes using analysis tools

### For Edge Devices
- Use **`yolo11n_int8_conservative.trt`** if accuracy is acceptable
- Consider generating domain-specific calibration data

## 📞 Support

For detailed usage instructions, see:
- **Main Guide**: `../USER_GUIDE_EXPORT_TO_TENSORRT.md`
- **Quick Reference**: `../QUICK_REFERENCE.md`
- **Documentation Index**: `../DOCUMENTATION_INDEX.md`

---

**Directory cleaned and organized! 🎉** 