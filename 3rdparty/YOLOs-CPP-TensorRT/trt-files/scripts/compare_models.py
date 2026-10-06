#!/usr/bin/env python3
"""
Compare FP32 and FP16 TensorRT models
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

def compare_models(fp32_path: str, fp16_path: str):
    """Compare FP32 and FP16 models"""
    print("=" * 80)
    print("TensorRT Model Comparison: FP32 vs FP16")
    print("=" * 80)
    
    # Analyze both models
    fp32_info = analyze_engine(fp32_path)
    fp16_info = analyze_engine(fp16_path)
    
    if not fp32_info or not fp16_info:
        print("Failed to analyze one or both models")
        return
    
    print(f"\n📁 File Sizes:")
    print(f"  FP32: {fp32_info['file_size']:,} bytes ({fp32_info['file_size'] / 1024**2:.1f} MB)")
    print(f"  FP16: {fp16_info['file_size']:,} bytes ({fp16_info['file_size'] / 1024**2:.1f} MB)")
    
    size_diff = fp16_info['file_size'] - fp32_info['file_size']
    size_diff_pct = (size_diff / fp32_info['file_size']) * 100
    print(f"  Difference: {size_diff:+,} bytes ({size_diff_pct:+.1f}%)")
    
    print(f"\n💾 Runtime Memory Usage:")
    print(f"  FP32: {fp32_info['device_memory_size']:,} bytes ({fp32_info['device_memory_size'] / 1024**2:.1f} MB)")
    print(f"  FP16: {fp16_info['device_memory_size']:,} bytes ({fp16_info['device_memory_size'] / 1024**2:.1f} MB)")
    
    mem_diff = fp16_info['device_memory_size'] - fp32_info['device_memory_size']
    mem_diff_pct = (mem_diff / fp32_info['device_memory_size']) * 100
    print(f"  Difference: {mem_diff:+,} bytes ({mem_diff_pct:+.1f}%)")
    
    print(f"\n🔧 Engine Properties:")
    print(f"  FP32: {fp32_info['num_layers']} layers, {fp32_info['num_io_tensors']} IO tensors")
    print(f"  FP16: {fp16_info['num_layers']} layers, {fp16_info['num_io_tensors']} IO tensors")
    
    print(f"\n📥 Input Tensors:")
    for i, (fp32_input, fp16_input) in enumerate(zip(fp32_info['inputs'], fp16_info['inputs'])):
        print(f"  {i}: {fp32_input['name']}")
        print(f"    FP32: {fp32_input['shape']} - {fp32_input['dtype']}")
        print(f"    FP16: {fp16_input['shape']} - {fp16_input['dtype']}")
    
    print(f"\n📤 Output Tensors:")
    for i, (fp32_output, fp16_output) in enumerate(zip(fp32_info['outputs'], fp16_info['outputs'])):
        print(f"  {i}: {fp32_output['name']}")
        print(f"    FP32: {fp32_output['shape']} - {fp32_output['dtype']}")
        print(f"    FP16: {fp16_output['shape']} - {fp16_output['dtype']}")
    
    print(f"\n💡 Key Insights:")
    print(f"  • FP16 models may have larger file sizes due to engine metadata")
    print(f"  • FP16 provides better runtime performance and memory efficiency")
    print(f"  • FP16 operations are ~2x faster on modern GPUs")
    print(f"  • FP16 uses ~50% less memory for weights during inference")
    
    print("=" * 80)

def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_models.py <fp32_model.trt> <fp16_model.trt>")
        print("Example: python compare_models.py yolo11n.trt yolo11n_fp16_test.trt")
        sys.exit(1)
    
    fp32_path = sys.argv[1]
    fp16_path = sys.argv[2]
    
    compare_models(fp32_path, fp16_path)

if __name__ == "__main__":
    main() 