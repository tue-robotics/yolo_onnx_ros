# Enhanced TensorRT Conversion Script

## Overview

The enhanced `convert_to_tensorrt.py` script provides a modern, feature-rich solution for converting ONNX models to optimized TensorRT engines. It leverages the latest TensorRT 10.x API and includes numerous improvements over the original version.

## 🚀 Key Improvements

### Modern TensorRT API (10.x+)
- **Tensor-based API**: Replaced deprecated binding-based methods with modern tensor management
- **enqueueV3**: Uses the latest execution API for optimal performance
- **Memory Pool Management**: Uses `set_memory_pool_limit()` instead of deprecated workspace settings
- **Enhanced Error Handling**: Comprehensive error recovery and validation

### Advanced Features
- **Dynamic Shapes**: Full support for variable input dimensions with optimization profiles
- **INT8 Calibration**: Comprehensive calibration with multiple algorithms and dataset support
- **Weight Streaming**: Support for large models with memory optimization
- **Engine Caching**: Intelligent caching system to avoid repeated conversions
- **Timing Cache**: Persistent timing optimization for faster subsequent builds

### Enhanced Precision Support
- **FP16**: Automatic capability detection and optimization
- **INT8**: Multiple calibration algorithms (entropy_v2, legacy, percentile)
- **TF32**: Configurable TF32 support for optimal performance
- **Mixed Precision**: Intelligent precision selection based on model type

### Developer Experience
- **Progress Tracking**: Visual progress bars for batch operations
- **Detailed Logging**: Multi-level logging with file output
- **Benchmarking**: Built-in performance testing and validation
- **Auto-optimization**: Automatic settings optimization based on model type detection

## 📋 Requirements

### Essential Dependencies
```bash
pip install tensorrt>=10.0.0
pip install pycuda>=2024.1
pip install numpy>=1.21.0
```

### Optional Dependencies (Enhanced Features)
```bash
pip install tqdm              # Progress bars
pip install Pillow           # INT8 calibration image preprocessing
pip install onnx             # Model type detection
```

### System Requirements
- NVIDIA GPU with Compute Capability 6.0+
- CUDA Toolkit 12.x+
- TensorRT 10.x+ (recommended)
- Python 3.8+

## 🔧 Usage Examples

### Basic Conversion
```bash
# Simple ONNX to TensorRT conversion
python convert_to_tensorrt.py --onnx model.onnx
```

### FP16 Optimization
```bash
# Enable FP16 for better performance
python convert_to_tensorrt.py --onnx model.onnx --fp16
```

### Dynamic Shapes
```bash
# Support variable input sizes
python convert_to_tensorrt.py --onnx model.onnx --dynamic-shapes \
  --min-shapes "input:1,3,320,320" \
  --opt-shapes "input:1,3,640,640" \
  --max-shapes "input:1,3,1280,1280"
```

### INT8 Quantization with Calibration
```bash
# INT8 optimization with calibration dataset
python convert_to_tensorrt.py --onnx model.onnx --int8 \
  --calibration-data ./calibration_images/ \
  --calibration-batch-size 32
```

### Batch Conversion
```bash
# Convert all ONNX models in a directory
python convert_to_tensorrt.py --convert-all --models-dir ./models/ \
  --fp16 --recursive
```

### Advanced Optimization
```bash
# Full optimization with benchmarking
python convert_to_tensorrt.py --onnx model.onnx \
  --fp16 \
  --workspace-size 8 \
  --optimization-level 5 \
  --timing-cache \
  --validate-engine \
  --benchmark-runs 100 \
  --auto-optimize
```

## 📊 Command Line Options

### Input/Output
- `--onnx`: Path to input ONNX model
- `--output, -o`: Path to output TensorRT engine
- `--convert-all`: Convert all ONNX models in directory
- `--models-dir`: Directory containing ONNX models (default: ../models)
- `--recursive`: Search for ONNX files recursively

### Precision & Optimization
- `--fp16`: Enable FP16 precision
- `--int8`: Enable INT8 precision
- `--disable-tf32`: Disable TF32 precision
- `--optimization-level`: Builder optimization level (0-5, default: 3)

### Dynamic Shapes
- `--dynamic-shapes`: Enable dynamic input shapes
- `--min-shapes`: Minimum shapes (format: input1:1,3,224,224;input2:1,512)
- `--opt-shapes`: Optimal shapes
- `--max-shapes`: Maximum shapes

### Performance Settings
- `--batch-size`: Maximum batch size (default: 1)
- `--workspace-size`: Workspace size in GB (default: 4)
- `--timing-iterations`: Number of timing iterations (default: 8)
- `--timing-cache`: Enable timing cache (default: enabled)
- `--weight-streaming`: Enable weight streaming

### INT8 Calibration
- `--calibration-data`: Path to calibration dataset directory
- `--calibration-batch-size`: Calibration batch size (default: 32)
- `--calibration-algorithm`: Algorithm (entropy_v2, legacy, percentile)

### Advanced Features
- `--dla-core`: DLA core ID (if available)
- `--gpu-fallback`: Enable GPU fallback for DLA
- `--refittable`: Enable engine refitting
- `--version-compatible`: Enable version compatibility
- `--device-id`: CUDA device ID (default: 0)

### Validation & Testing
- `--validate-engine`: Validate generated engine
- `--benchmark-runs`: Number of benchmark runs (0 to disable)
- `--warmup-runs`: Number of warmup runs for benchmarking

### Other Options
- `--force`: Force conversion even if TRT file exists
- `--verbose, -v`: Enable verbose logging
- `--auto-optimize`: Automatically optimize settings based on model type

## 🎯 Model Type Detection & Auto-optimization

The script can automatically detect model types and apply optimal settings:

### YOLO Models
- Automatically enables FP16 if not specified
- Increases workspace size to 2GB minimum
- Optimizes for detection workloads

### Classification Models
- Detects ResNet, EfficientNet, etc.
- Applies appropriate precision settings

### Segmentation Models
- Increases workspace size to 4GB minimum
- Suggests dynamic shapes for variable input sizes

## 📈 Performance Features

### Benchmarking
The script includes comprehensive benchmarking capabilities:
- Warmup runs for stable measurements
- Statistical analysis (mean, std, min, max, median)
- FPS calculations
- Memory usage monitoring

### Caching System
- **Engine Caching**: Avoids rebuilding identical engines
- **Timing Cache**: Persistent optimization data
- **Configuration Hashing**: Intelligent cache invalidation

### Progress Tracking
- Visual progress bars for batch operations
- Detailed timing information
- Success/failure statistics

## 🔍 Validation & Testing

### Engine Validation
- Automatic validation of generated engines
- Input/output shape verification
- Basic functionality testing

### Error Handling
- Comprehensive error recovery
- Detailed error messages
- Graceful degradation for unsupported features

## 📝 Output & Logging

### Enhanced Logging
- Multi-level logging (INFO, DEBUG, ERROR)
- File-based logging for detailed analysis
- Structured output with timing information

### Conversion Metrics
- Build times for each phase
- Engine size information
- Performance statistics
- Memory usage details

## 🚨 Migration from Original Script

The enhanced script maintains backward compatibility while adding new features:

### Breaking Changes
- Some deprecated argument names have been updated
- Enhanced error checking may catch previously ignored issues

### Recommended Migration
1. Update TensorRT to 10.x+ if possible
2. Install optional dependencies for full features
3. Review and update any automation scripts
4. Test with `--validate-engine` flag

## 🐛 Troubleshooting

### Common Issues

#### TensorRT Version Compatibility
```bash
# Check TensorRT version
python -c "import tensorrt as trt; print(trt.__version__)"
```

#### CUDA Memory Issues
- Reduce `--workspace-size`
- Use `--weight-streaming` for large models
- Check GPU memory with `nvidia-smi`

#### INT8 Calibration Errors
- Ensure calibration dataset is properly formatted
- Check image preprocessing requirements
- Verify dataset path and permissions

#### Dynamic Shapes Issues
- Ensure all shape specifications are complete
- Check input names match ONNX model
- Verify shape ranges are reasonable

### Debug Mode
```bash
# Enable verbose logging for debugging
python convert_to_tensorrt.py --onnx model.onnx --verbose
```

## 📚 Additional Resources

- [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/)
- [TensorRT Python API](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/)
- [ONNX Model Zoo](https://github.com/onnx/models)
- [YOLOs-CPP Repository](https://github.com/Geekgineer/YOLOs-CPP-TensorRT)

## 🤝 Contributing

Contributions are welcome! Please ensure:
- Code follows the existing style
- New features include appropriate tests
- Documentation is updated
- Backward compatibility is maintained where possible

## 📄 License

This enhanced script maintains the same license as the parent YOLOs-CPP project. 