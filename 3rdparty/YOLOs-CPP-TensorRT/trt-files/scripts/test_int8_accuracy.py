#!/usr/bin/env python3
"""
Test INT8 model accuracy and provide recommendations
Compares INT8 model performance with FP32/FP16 models
"""

import os
import sys
import numpy as np
from pathlib import Path
import argparse

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("Warning: TensorRT not available. Install with: pip install tensorrt pycuda")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available. Install with: pip install Pillow")

def load_engine(engine_path: str):
    """Load TensorRT engine"""
    if not TRT_AVAILABLE:
        print("Error: TensorRT not available")
        return None
    
    try:
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = runtime.deserialize_cuda_engine(engine_data)
        return engine
    except Exception as e:
        print(f"Error loading engine {engine_path}: {e}")
        return None

def preprocess_image(image_path: str, input_shape: tuple = (3, 640, 640)):
    """Preprocess image for YOLO inference"""
    if not PIL_AVAILABLE:
        print("Error: PIL required for image preprocessing")
        return None
    
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(input_shape[1:3][::-1])  # H,W -> W,H for PIL
        img_array = np.array(img, dtype=np.float32)
        
        # YOLO preprocessing: normalize to [0,1] and convert to CHW format
        img_array = img_array.transpose(2, 0, 1) / 255.0
        
        # Ensure proper data range
        img_array = np.clip(img_array, 0.0, 1.0)
        img_array = np.nan_to_num(img_array, nan=0.0, posinf=1.0, neginf=0.0)
        
        return img_array
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None

def run_inference(engine, input_data: np.ndarray):
    """Run inference on TensorRT engine"""
    if not TRT_AVAILABLE:
        return None
    
    try:
        context = engine.create_execution_context()
        
        # Allocate device memory
        input_size = input_data.nbytes
        output_size = engine.get_tensor_size("output0")  # Use tensor name instead of binding index
        
        d_input = cuda.mem_alloc(input_size)
        d_output = cuda.mem_alloc(output_size)
        
        # Create CUDA stream
        stream = cuda.Stream()
        
        # Copy input data to device
        cuda.memcpy_htod_async(d_input, input_data, stream)
        
        # Run inference
        context.execute_async_v2(
            bindings=[int(d_input), int(d_output)],
            stream_handle=stream.handle
        )
        
        # Copy output data from device
        output_data = np.empty(output_size // 4, dtype=np.float32)  # Assuming float32 output
        cuda.memcpy_dtoh_async(output_data, d_output, stream)
        
        # Synchronize
        stream.synchronize()
        
        return output_data
        
    except Exception as e:
        print(f"Error running inference: {e}")
        return None

def analyze_output(output_data: np.ndarray, confidence_threshold: float = 0.25):
    """Analyze YOLO output to count detections"""
    try:
        # Reshape output to (1, 84, 8400) for YOLO11n
        output = output_data.reshape(1, 84, 8400)
        
        # Extract confidence scores (last 80 values for each detection)
        confidence_scores = output[0, 4:84, :]  # Shape: (80, 8400)
        
        # Find detections above threshold
        max_confidences = np.max(confidence_scores, axis=0)  # Shape: (8400,)
        detections = max_confidences > confidence_threshold
        
        num_detections = np.sum(detections)
        
        return {
            'num_detections': int(num_detections),
            'max_confidence': float(np.max(max_confidences)),
            'mean_confidence': float(np.mean(max_confidences)),
            'detection_rate': float(num_detections / 8400)
        }
        
    except Exception as e:
        print(f"Error analyzing output: {e}")
        return None

def test_model_accuracy(engine_path: str, test_image_path: str, model_name: str = "Unknown"):
    """Test model accuracy on a single image"""
    print(f"\n🔍 Testing {model_name} ({engine_path})")
    print("=" * 60)
    
    # Load engine
    engine = load_engine(engine_path)
    if engine is None:
        return None
    
    # Preprocess image
    input_data = preprocess_image(test_image_path)
    if input_data is None:
        return None
    
    # Add batch dimension
    input_data = np.expand_dims(input_data, axis=0)
    
    # Run inference
    output_data = run_inference(engine, input_data)
    if output_data is None:
        return None
    
    # Analyze results
    results = analyze_output(output_data)
    if results is None:
        return None
    
    print(f"📊 Detection Results:")
    print(f"  Number of detections: {results['num_detections']}")
    print(f"  Max confidence: {results['max_confidence']:.4f}")
    print(f"  Mean confidence: {results['mean_confidence']:.4f}")
    print(f"  Detection rate: {results['detection_rate']:.4f}")
    
    return results

def compare_models(fp32_path: str, fp16_path: str, int8_path: str, test_image: str):
    """Compare all three models"""
    print("=" * 80)
    print("INT8 Accuracy Test - Model Comparison")
    print("=" * 80)
    
    # Test each model
    fp32_results = test_model_accuracy(fp32_path, test_image, "FP32 Model")
    fp16_results = test_model_accuracy(fp16_path, test_image, "FP16 Model")
    int8_results = test_model_accuracy(int8_path, test_image, "INT8 Model")
    
    if fp32_results and fp16_results and int8_results:
        print(f"\n📈 Comparison Summary:")
        print(f"  FP32 detections: {fp32_results['num_detections']}")
        print(f"  FP16 detections: {fp16_results['num_detections']}")
        print(f"  INT8 detections: {int8_results['num_detections']}")
        
        # Calculate accuracy loss
        fp16_loss = abs(fp16_results['num_detections'] - fp32_results['num_detections'])
        int8_loss = abs(int8_results['num_detections'] - fp32_results['num_detections'])
        
        print(f"\n⚠️  Accuracy Analysis:")
        print(f"  FP16 vs FP32: {fp16_loss} detection difference")
        print(f"  INT8 vs FP32: {int8_loss} detection difference")
        
        if int8_results['num_detections'] == 0:
            print(f"\n❌ CRITICAL ISSUE: INT8 model detected nothing!")
            print(f"   This indicates severe accuracy loss from quantization.")
            print(f"   Recommendations:")
            print(f"   1. Use FP16 instead of INT8 for this model")
            print(f"   2. Try different calibration data")
            print(f"   3. Consider quantization-aware training")
            print(f"   4. Use conservative INT8 settings")
        elif int8_loss > 5:
            print(f"\n⚠️  SIGNIFICANT ACCURACY LOSS: INT8 model lost {int8_loss} detections")
            print(f"   Consider using FP16 for better accuracy/performance balance")
        else:
            print(f"\n✅ ACCEPTABLE ACCURACY: INT8 model performs reasonably well")
            print(f"   INT8 quantization is working correctly")
    
    return fp32_results, fp16_results, int8_results

def main():
    parser = argparse.ArgumentParser(description="Test INT8 model accuracy")
    parser.add_argument('--fp32', type=str, required=True, help='FP32 model path')
    parser.add_argument('--fp16', type=str, required=True, help='FP16 model path')
    parser.add_argument('--int8', type=str, required=True, help='INT8 model path')
    parser.add_argument('--test-image', type=str, required=True, help='Test image path')
    
    args = parser.parse_args()
    
    # Check if files exist
    for model_path in [args.fp32, args.fp16, args.int8]:
        if not os.path.exists(model_path):
            print(f"Error: Model file not found: {model_path}")
            return 1
    
    if not os.path.exists(args.test_image):
        print(f"Error: Test image not found: {args.test_image}")
        return 1
    
    # Run comparison
    results = compare_models(args.fp32, args.fp16, args.int8, args.test_image)
    
    if results[2] and results[2]['num_detections'] == 0:
        return 1  # INT8 failed completely
    return 0

if __name__ == "__main__":
    sys.exit(main()) 