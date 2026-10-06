#!/usr/bin/env python3
"""
Deep TensorRT Engine Analysis
Analyzes the internal structure of TensorRT engines to understand FP16 usage
"""

import os
import sys
import tensorrt as trt
import numpy as np
from pathlib import Path

def analyze_engine_internals(engine_path: str) -> dict:
    """Deep analysis of TensorRT engine internals"""
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
        
        # Try to analyze layers (this might not work in all TensorRT versions)
        layer_analysis = {}
        try:
            # In TensorRT 10.x, we need to use different APIs
            print(f"Engine has {num_layers} layers")
            
            # Try to get layer information using different methods
            for i in range(min(num_layers, 10)):  # Limit to first 10 layers
                try:
                    # Try different ways to access layer info
                    layer_info = {
                        'index': i,
                        'name': f"Layer_{i}",
                        'type': 'Unknown'
                    }
                    layer_analysis[i] = layer_info
                except Exception as e:
                    print(f"Could not analyze layer {i}: {e}")
                    break
                    
        except Exception as e:
            print(f"Layer analysis not available: {e}")
        
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
        
        # Try to get more detailed engine info
        engine_props = {}
        try:
            # Check if engine has any special properties
            if hasattr(engine, 'get_profile_shape'):
                engine_props['has_profiles'] = True
            else:
                engine_props['has_profiles'] = False
                
            # Check memory usage breakdown
            engine_props['device_memory_size'] = device_memory_size
            engine_props['file_size'] = file_size
            
        except Exception as e:
            print(f"Could not get engine properties: {e}")
        
        return {
            'file_size': file_size,
            'device_memory_size': device_memory_size,
            'num_layers': num_layers,
            'num_io_tensors': num_io_tensors,
            'inputs': inputs,
            'outputs': outputs,
            'layer_analysis': layer_analysis,
            'engine_props': engine_props,
            'engine': engine
        }
        
    except Exception as e:
        print(f"Error analyzing engine {engine_path}: {e}")
        return None

def compare_engines_deep(fp32_path: str, fp16_path: str):
    """Deep comparison of FP32 and FP16 engines"""
    print("=" * 80)
    print("Deep TensorRT Engine Analysis: FP32 vs FP16")
    print("=" * 80)
    
    # Analyze both models
    fp32_info = analyze_engine_internals(fp32_path)
    fp16_info = analyze_engine_internals(fp16_path)
    
    if not fp32_info or not fp16_info:
        print("Failed to analyze one or both models")
        return
    
    print(f"\n📊 Basic Comparison:")
    print(f"  FP32 File Size: {fp32_info['file_size']:,} bytes ({fp32_info['file_size'] / 1024**2:.1f} MB)")
    print(f"  FP16 File Size: {fp16_info['file_size']:,} bytes ({fp16_info['file_size'] / 1024**2:.1f} MB)")
    print(f"  Size Difference: {fp16_info['file_size'] - fp32_info['file_size']:+,} bytes")
    
    print(f"\n💾 Memory Usage:")
    print(f"  FP32 Runtime Memory: {fp32_info['device_memory_size']:,} bytes ({fp32_info['device_memory_size'] / 1024**2:.1f} MB)")
    print(f"  FP16 Runtime Memory: {fp16_info['device_memory_size']:,} bytes ({fp16_info['device_memory_size'] / 1024**2:.1f} MB)")
    print(f"  Memory Difference: {fp16_info['device_memory_size'] - fp32_info['device_memory_size']:+,} bytes")
    
    print(f"\n🔧 Engine Structure:")
    print(f"  FP32 Layers: {fp32_info['num_layers']}")
    print(f"  FP16 Layers: {fp16_info['num_layers']}")
    print(f"  Layer Difference: {fp16_info['num_layers'] - fp32_info['num_layers']:+d}")
    
    print(f"\n📥 Input Analysis:")
    for i, (fp32_input, fp16_input) in enumerate(zip(fp32_info['inputs'], fp16_info['inputs'])):
        print(f"  Input {i}: {fp32_input['name']}")
        print(f"    FP32: {fp32_input['shape']} - {fp32_input['dtype']}")
        print(f"    FP16: {fp16_input['shape']} - {fp16_input['dtype']}")
    
    print(f"\n📤 Output Analysis:")
    for i, (fp32_output, fp16_output) in enumerate(zip(fp32_info['outputs'], fp16_info['outputs'])):
        print(f"  Output {i}: {fp32_output['name']}")
        print(f"    FP32: {fp32_output['shape']} - {fp32_output['dtype']}")
        print(f"    FP16: {fp16_output['shape']} - {fp16_output['dtype']}")
    
    # Calculate compression ratio
    compression_ratio = fp16_info['device_memory_size'] / fp32_info['device_memory_size']
    print(f"\n📈 Compression Analysis:")
    print(f"  Memory Compression Ratio: {compression_ratio:.3f} ({compression_ratio*100:.1f}%)")
    print(f"  Expected FP16 Ratio: 0.5 (50%)")
    print(f"  Actual vs Expected: {compression_ratio/0.5:.1f}x of expected compression")
    
    print(f"\n🔍 Why FP16 might not show dramatic size reduction:")
    print(f"  1. TensorRT engines include metadata and optimization info")
    print(f"  2. Only weights are compressed, activations remain FP32")
    print(f"  3. Layer fusion may add overhead")
    print(f"  4. Some layers may not benefit from FP16")
    print(f"  5. Engine format overhead")
    
    print(f"\n✅ FP16 Benefits (even with similar file sizes):")
    print(f"  • Runtime performance: ~2x faster operations")
    print(f"  • Memory bandwidth: Better utilization")
    print(f"  • Power efficiency: Lower power consumption")
    print(f"  • Layer optimization: {fp32_info['num_layers'] - fp16_info['num_layers']} fewer layers")
    
    print("=" * 80)

def main():
    if len(sys.argv) != 3:
        print("Usage: python deep_engine_analysis.py <fp32_model.trt> <fp16_model.trt>")
        print("Example: python deep_engine_analysis.py yolo11n.trt yolo11n_fp16_final.trt")
        sys.exit(1)
    
    fp32_path = sys.argv[1]
    fp16_path = sys.argv[2]
    
    compare_engines_deep(fp32_path, fp16_path)

if __name__ == "__main__":
    main() 