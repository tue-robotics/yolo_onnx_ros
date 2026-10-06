#!/usr/bin/env python3
"""
Compare all TensorRT precision modes: FP32, FP16, and INT8
Shows the differences in size, memory usage, and performance characteristics
"""

import os
import sys
import tensorrt as trt
import numpy as np
from pathlib import Path

def analyze_engine(engine_path: str) -> dict:
    """Analyze a TensorRT engine and return detailed information"""
    if not os.path.exists(engine_path):
        print(f"Error: Engine file not found: {engine_path}")
        return None
    
    try:
        # Load engine
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = runtime.deserialize_cuda_engine(engine_data)
        
        # Get file size
        file_size = os.path.getsize(engine_path)
        
        # Get engine properties
        device_memory_size = engine.device_memory_size_v2
        num_layers = engine.num_layers
        num_io_tensors = engine.num_io_tensors
        
        # Analyze IO tensors
        inputs = []
        outputs = []
        for i in range(num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = engine.get_tensor_shape(name)
            dtype = engine.get_tensor_dtype(name)
            mode = engine.get_tensor_mode(name)
            
            tensor_info = {
                'name': name,
                'shape': tuple(shape),
                'dtype': str(dtype),
                'mode': 'INPUT' if mode == trt.TensorIOMode.INPUT else 'OUTPUT'
            }
            
            if mode == trt.TensorIOMode.INPUT:
                inputs.append(tensor_info)
            else:
                outputs.append(tensor_info)
        
        return {
            'file_size': file_size,
            'device_memory_size': device_memory_size,
            'num_layers': num_layers,
            'num_io_tensors': num_io_tensors,
            'inputs': inputs,
            'outputs': outputs,
            'engine': engine
        }
        
    except Exception as e:
        print(f"Error analyzing engine {engine_path}: {e}")
        return None

def compare_all_precisions(fp32_path: str, fp16_path: str, int8_path: str):
    """Compare FP32, FP16, and INT8 models"""
    print("=" * 100)
    print("TensorRT Precision Comparison: FP32 vs FP16 vs INT8")
    print("=" * 100)
    
    # Analyze all models
    fp32_info = analyze_engine(fp32_path)
    fp16_info = analyze_engine(fp16_path)
    int8_info = analyze_engine(int8_path)
    
    if not fp32_info or not fp16_info or not int8_info:
        print("Failed to analyze one or more models")
        return
    
    print(f"\n📁 File Sizes:")
    print(f"  FP32: {fp32_info['file_size']:,} bytes ({fp32_info['file_size'] / 1024**2:.1f} MB)")
    print(f"  FP16: {fp16_info['file_size']:,} bytes ({fp16_info['file_size'] / 1024**2:.1f} MB)")
    print(f"  INT8: {int8_info['file_size']:,} bytes ({int8_info['file_size'] / 1024**2:.1f} MB)")
    
    fp16_size_diff = fp16_info['file_size'] - fp32_info['file_size']
    fp16_size_pct = (fp16_size_diff / fp32_info['file_size']) * 100
    int8_size_diff = int8_info['file_size'] - fp32_info['file_size']
    int8_size_pct = (int8_size_diff / fp32_info['file_size']) * 100
    
    print(f"  FP16 vs FP32: {fp16_size_diff:+,} bytes ({fp16_size_pct:+.1f}%)")
    print(f"  INT8 vs FP32: {int8_size_diff:+,} bytes ({int8_size_pct:+.1f}%)")
    
    print(f"\n💾 Runtime Memory Usage:")
    print(f"  FP32: {fp32_info['device_memory_size']:,} bytes ({fp32_info['device_memory_size'] / 1024**2:.1f} MB)")
    print(f"  FP16: {fp16_info['device_memory_size']:,} bytes ({fp16_info['device_memory_size'] / 1024**2:.1f} MB)")
    print(f"  INT8: {int8_info['device_memory_size']:,} bytes ({int8_info['device_memory_size'] / 1024**2:.1f} MB)")
    
    fp16_mem_diff = fp16_info['device_memory_size'] - fp32_info['device_memory_size']
    fp16_mem_pct = (fp16_mem_diff / fp32_info['device_memory_size']) * 100
    int8_mem_diff = int8_info['device_memory_size'] - fp32_info['device_memory_size']
    int8_mem_pct = (int8_mem_diff / fp32_info['device_memory_size']) * 100
    
    print(f"  FP16 vs FP32: {fp16_mem_diff:+,} bytes ({fp16_mem_pct:+.1f}%)")
    print(f"  INT8 vs FP32: {int8_mem_diff:+,} bytes ({int8_mem_pct:+.1f}%)")
    
    print(f"\n🔧 Engine Structure:")
    print(f"  FP32: {fp32_info['num_layers']} layers")
    print(f"  FP16: {fp16_info['num_layers']} layers")
    print(f"  INT8: {int8_info['num_layers']} layers")
    
    fp16_layer_diff = fp16_info['num_layers'] - fp32_info['num_layers']
    int8_layer_diff = int8_info['num_layers'] - fp32_info['num_layers']
    
    print(f"  FP16 vs FP32: {fp16_layer_diff:+d} layers")
    print(f"  INT8 vs FP32: {int8_layer_diff:+d} layers")
    
    print(f"\n📈 Compression Analysis:")
    fp16_compression = fp16_info['device_memory_size'] / fp32_info['device_memory_size']
    int8_compression = int8_info['device_memory_size'] / fp32_info['device_memory_size']
    
    print(f"  FP16 Compression: {fp16_compression:.3f} ({fp16_compression*100:.1f}%)")
    print(f"  INT8 Compression: {int8_compression:.3f} ({int8_compression*100:.1f}%)")
    print(f"  Expected FP16: 0.5 (50%)")
    print(f"  Expected INT8: 0.25 (25%)")
    
    print(f"\n🎯 Performance Characteristics:")
    print(f"  FP32: Baseline - Full precision, highest accuracy")
    print(f"  FP16: ~2x faster operations, ~8% memory reduction")
    print(f"  INT8: ~4x faster operations, ~8% memory reduction")
    
    print(f"\n⚖️  Accuracy vs Performance Trade-offs:")
    print(f"  FP32: Highest accuracy, slowest performance")
    print(f"  FP16: Minimal accuracy loss, good performance boost")
    print(f"  INT8: Some accuracy loss, best performance")
    
    print(f"\n💡 Recommendations:")
    if int8_compression < 0.5:
        print(f"  ✅ INT8 provides significant compression ({int8_compression*100:.1f}% of FP32)")
    else:
        print(f"  ⚠️  INT8 compression limited ({int8_compression*100:.1f}% of FP32)")
    
    if fp16_compression < 0.8:
        print(f"  ✅ FP16 provides good compression ({fp16_compression*100:.1f}% of FP32)")
    else:
        print(f"  ⚠️  FP16 compression limited ({fp16_compression*100:.1f}% of FP32)")
    
    print(f"\n🚀 Best Use Cases:")
    print(f"  FP32: High accuracy requirements, development/testing")
    print(f"  FP16: Production inference, good accuracy needed")
    print(f"  INT8: Edge devices, real-time applications, size-constrained environments")
    
    print("=" * 100)

def main():
    if len(sys.argv) != 4:
        print("Usage: python compare_all_precisions.py <fp32_model.trt> <fp16_model.trt> <int8_model.trt>")
        print("Example: python compare_all_precisions.py yolo11n.trt yolo11n_fp16_final.trt yolo11n_int8.trt")
        sys.exit(1)
    
    fp32_path = sys.argv[1]
    fp16_path = sys.argv[2]
    int8_path = sys.argv[3]
    
    compare_all_precisions(fp32_path, fp16_path, int8_path)

if __name__ == "__main__":
    main() 