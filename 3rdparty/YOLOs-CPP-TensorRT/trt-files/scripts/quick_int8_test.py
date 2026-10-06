#!/usr/bin/env python3
"""
Quick test to verify INT8 model loading and basic inference
"""

import os
import sys
import numpy as np

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("Error: TensorRT not available")

def test_model_loading(model_path: str):
    """Test if model can be loaded"""
    print(f"Testing model: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    try:
        # Load engine
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(model_path, 'rb') as f:
            engine_data = f.read()
        engine = runtime.deserialize_cuda_engine(engine_data)
        
        print(f"✅ Model loaded successfully")
        print(f"   File size: {os.path.getsize(model_path):,} bytes")
        print(f"   Device memory: {engine.device_memory_size_v2:,} bytes")
        print(f"   Number of layers: {engine.num_layers}")
        print(f"   Number of IO tensors: {engine.num_io_tensors}")
        
        # Check IO tensors
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = engine.get_tensor_shape(name)
            dtype = engine.get_tensor_dtype(name)
            mode = engine.get_tensor_mode(name)
            
            io_type = "INPUT" if mode == trt.TensorIOMode.INPUT else "OUTPUT"
            print(f"   {io_type}: {name} - {shape} - {dtype}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

def main():
    # Test different models
    models = [
        "yolo11n.trt",
        "yolo11n_fp16_aggressive.trt", 
        "yolo11n_int8_real.trt"
    ]
    
    print("=" * 60)
    print("INT8 Model Loading Test")
    print("=" * 60)
    
    results = {}
    for model in models:
        print(f"\n🔍 Testing {model}...")
        results[model] = test_model_loading(model)
    
    print(f"\n📊 Summary:")
    for model, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {model}: {status}")
    
    if results.get("yolo11n_int8_real.trt", False):
        print(f"\n✅ INT8 model loads successfully!")
        print(f"   The issue might be in the inference pipeline, not model loading.")
    else:
        print(f"\n❌ INT8 model fails to load!")
        print(f"   This indicates a problem with the INT8 conversion.")

if __name__ == "__main__":
    main() 