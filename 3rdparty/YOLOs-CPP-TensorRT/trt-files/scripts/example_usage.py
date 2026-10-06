#!/usr/bin/env python3
"""
Example usage of the Enhanced TensorRT Conversion Script

This script demonstrates how to use the enhanced convert_to_tensorrt.py
both as a command-line tool and programmatically.

Author: YOLOs-CPP Enhanced TensorRT Integration
Date: January 2025
"""

import os
import sys
from pathlib import Path

# Add the scripts directory to path so we can import our converter
sys.path.append(str(Path(__file__).parent))

try:
    from convert_to_tensorrt import EnhancedTensorRTConverter, ConversionConfig
    print("✓ Enhanced TensorRT converter imported successfully")
except ImportError as e:
    print(f"❌ Error importing converter: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install tensorrt pycuda numpy tqdm Pillow onnx")
    sys.exit(1)

def example_basic_conversion():
    """Basic ONNX to TensorRT conversion example"""
    print("\n" + "="*50)
    print("Example 1: Basic Conversion")
    print("="*50)
    
    # Define paths (adjust as needed)
    onnx_path = "../models/yolo11n.onnx"  # Adjust path as needed
    trt_path = "../models/yolo11n_basic.trt"
    
    if not os.path.exists(onnx_path):
        print(f"⚠️  ONNX file not found: {onnx_path}")
        print("Please adjust the path or download a model")
        return False
    
    # Basic configuration
    config = ConversionConfig(
        batch_size=1,
        workspace_size=2,  # 2GB
        fp16=True,  # Enable FP16 for better performance
        optimization_level=3
    )
    
    # Create converter and convert
    converter = EnhancedTensorRTConverter(config)
    success = converter.convert_onnx_to_trt(onnx_path, trt_path)
    
    if success:
        print(f"✓ Basic conversion successful: {trt_path}")
        file_size = os.path.getsize(trt_path) / 1024**2
        print(f"  Engine size: {file_size:.1f} MB")
    else:
        print("❌ Basic conversion failed")
    
    return success

def example_advanced_conversion():
    """Advanced conversion with dynamic shapes and validation"""
    print("\n" + "="*50)
    print("Example 2: Advanced Conversion")
    print("="*50)
    
    onnx_path = "../models/yolo11n.onnx"
    trt_path = "../models/yolo11n_advanced.trt"
    
    if not os.path.exists(onnx_path):
        print(f"⚠️  ONNX file not found: {onnx_path}")
        return False
    
    # Advanced configuration with dynamic shapes
    config = ConversionConfig(
        batch_size=1,
        workspace_size=4,  # 4GB for advanced optimization
        fp16=True,
        dynamic_shapes=True,
        min_shapes={"images": [1, 3, 320, 320]},
        opt_shapes={"images": [1, 3, 640, 640]},
        max_shapes={"images": [1, 3, 1280, 1280]},
        optimization_level=5,  # Maximum optimization
        timing_cache=True,
        validate_engine=True,
        benchmark_runs=50,
        warmup_runs=10
    )
    
    converter = EnhancedTensorRTConverter(config)
    success = converter.convert_onnx_to_trt(onnx_path, trt_path)
    
    if success:
        print(f"✓ Advanced conversion successful: {trt_path}")
        file_size = os.path.getsize(trt_path) / 1024**2
        print(f"  Engine size: {file_size:.1f} MB")
        
        # Print performance metrics if available
        if converter.conversion_metrics:
            print("  Performance metrics:")
            for metric, value in converter.conversion_metrics.items():
                if 'ms' in metric:
                    print(f"    {metric}: {value:.2f}")
                elif 'fps' in metric:
                    print(f"    {metric}: {value:.1f}")
                else:
                    print(f"    {metric}: {value:.2f}s")
    else:
        print("❌ Advanced conversion failed")
    
    return success

def example_int8_conversion():
    """INT8 conversion with calibration (if data available)"""
    print("\n" + "="*50)
    print("Example 3: INT8 Conversion")
    print("="*50)
    
    onnx_path = "../models/yolo11n.onnx"
    trt_path = "../models/yolo11n_int8.trt"
    calibration_path = "../data"  # Directory with calibration images
    
    if not os.path.exists(onnx_path):
        print(f"⚠️  ONNX file not found: {onnx_path}")
        return False
    
    # Check if calibration data exists
    if not os.path.exists(calibration_path):
        print(f"⚠️  Calibration data not found: {calibration_path}")
        print("INT8 conversion requires calibration images")
        return False
    
    # INT8 configuration
    config = ConversionConfig(
        batch_size=1,
        workspace_size=4,
        int8=True,
        calibration_data_path=calibration_path,
        calibration_batch_size=16,
        optimization_level=4,
        validate_engine=True,
        benchmark_runs=100
    )
    
    converter = EnhancedTensorRTConverter(config)
    success = converter.convert_onnx_to_trt(onnx_path, trt_path)
    
    if success:
        print(f"✓ INT8 conversion successful: {trt_path}")
        file_size = os.path.getsize(trt_path) / 1024**2
        print(f"  Engine size: {file_size:.1f} MB")
    else:
        print("❌ INT8 conversion failed")
    
    return success

def example_batch_conversion():
    """Batch conversion of multiple models"""
    print("\n" + "="*50)
    print("Example 4: Batch Conversion")
    print("="*50)
    
    models_dir = Path("../models")
    
    if not models_dir.exists():
        print(f"⚠️  Models directory not found: {models_dir}")
        return False
    
    # Find all ONNX files
    onnx_files = list(models_dir.glob("*.onnx"))
    if not onnx_files:
        print(f"⚠️  No ONNX files found in {models_dir}")
        return False
    
    print(f"Found {len(onnx_files)} ONNX files to convert")
    
    # Batch conversion configuration
    config = ConversionConfig(
        batch_size=1,
        workspace_size=2,
        fp16=True,
        optimization_level=3,
        timing_cache=True
    )
    
    converter = EnhancedTensorRTConverter(config)
    success_count = 0
    
    for onnx_file in onnx_files:
        trt_file = onnx_file.with_suffix('.trt')
        print(f"Converting {onnx_file.name}...")
        
        if converter.convert_onnx_to_trt(str(onnx_file), str(trt_file)):
            success_count += 1
            print(f"  ✓ Success")
        else:
            print(f"  ❌ Failed")
    
    print(f"Batch conversion completed: {success_count}/{len(onnx_files)} successful")
    return success_count == len(onnx_files)

def example_command_line_usage():
    """Show command line usage examples"""
    print("\n" + "="*50)
    print("Example 5: Command Line Usage")
    print("="*50)
    
    examples = [
        {
            "description": "Basic conversion",
            "command": "python convert_to_tensorrt.py --onnx ../models/yolo11n.onnx --fp16"
        },
        {
            "description": "Dynamic shapes with optimization",
            "command": """python convert_to_tensorrt.py --onnx ../models/yolo11n.onnx \\
  --dynamic-shapes \\
  --min-shapes "images:1,3,320,320" \\
  --opt-shapes "images:1,3,640,640" \\
  --max-shapes "images:1,3,1280,1280" \\
  --fp16 --optimization-level 5"""
        },
        {
            "description": "INT8 with calibration",
            "command": """python convert_to_tensorrt.py --onnx ../models/yolo11n.onnx \\
  --int8 --calibration-data ../data \\
  --calibration-batch-size 32"""
        },
        {
            "description": "Batch conversion with validation",
            "command": """python convert_to_tensorrt.py --convert-all \\
  --models-dir ../models --fp16 \\
  --validate-engine --benchmark-runs 50"""
        },
        {
            "description": "Auto-optimization with benchmarking",
            "command": """python convert_to_tensorrt.py --onnx ../models/yolo11n.onnx \\
  --auto-optimize --validate-engine \\
  --benchmark-runs 100 --verbose"""
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['description']}:")
        print(f"   {example['command']}")
    
    print("\nFor full help and all options:")
    print("   python convert_to_tensorrt.py --help")

def main():
    """Run all examples"""
    print("🚀 Enhanced TensorRT Conversion Examples")
    print("This script demonstrates various usage patterns")
    
    # Check system requirements
    try:
        import tensorrt as trt
        print(f"✓ TensorRT version: {trt.__version__}")
    except ImportError:
        print("❌ TensorRT not available")
        return
    
    try:
        import pycuda.driver as cuda
        cuda.init()
        print(f"✓ CUDA devices available: {cuda.Device.count()}")
    except ImportError:
        print("❌ PyCUDA not available")
        return
    
    # Run examples (commented out by default to avoid unwanted conversions)
    print("\n📚 Available Examples:")
    print("1. Basic conversion")
    print("2. Advanced conversion with dynamic shapes")
    print("3. INT8 conversion with calibration")
    print("4. Batch conversion")
    print("5. Command line usage examples")
    
    # Uncomment the examples you want to run:
    
    # example_basic_conversion()
    # example_advanced_conversion()
    # example_int8_conversion()
    # example_batch_conversion()
    example_command_line_usage()
    
    print("\n" + "="*60)
    print("🎉 Examples completed!")
    print("Uncomment the example functions in main() to run them.")
    print("Make sure to adjust file paths according to your setup.")
    print("="*60)

if __name__ == "__main__":
    main() 