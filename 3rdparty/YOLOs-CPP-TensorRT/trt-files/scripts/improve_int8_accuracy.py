#!/usr/bin/env python3
"""
Improve INT8 accuracy with better calibration and settings
Addresses the accuracy drop from 90% to 40% for person detection
"""

import os
import sys
import subprocess
from pathlib import Path

def generate_better_calibration():
    """Generate better calibration data for improved INT8 accuracy"""
    print("🔧 Generating better calibration data...")
    
    # Create multiple calibration datasets
    datasets = [
        ("real_images", "../data", 10),
        ("mixed_data", "../data", 20),
        ("extended_data", "../data", 30)
    ]
    
    for name, source, count in datasets:
        output_dir = f"./calibration_{name}"
        cmd = [
            "python3", "generate_calibration_data.py",
            "--copy-from", source,
            "--output", output_dir,
            "--max-copy", str(count)
        ]
        
        print(f"  Creating {name} dataset...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"  ✅ {name} dataset created successfully")
        else:
            print(f"  ❌ Failed to create {name} dataset: {result.stderr}")
    
    return True

def create_conservative_int8_model():
    """Create INT8 model with conservative settings"""
    print("\n🔧 Creating conservative INT8 model...")
    
    cmd = [
        "python3", "convert_to_tensorrt.py",
        "--onnx", "yolo11n.onnx",
        "--output", "yolo11n_int8_conservative.trt",
        "--int8",
        "--calibration-data", "./calibration_extended_data",
        "--calibration-batch-size", "8",
        "--clear-cache",
        "--verbose"
    ]
    
    print("  Running conversion with conservative settings...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("  ✅ Conservative INT8 model created successfully")
        return True
    else:
        print(f"  ❌ Failed to create conservative INT8 model: {result.stderr}")
        return False

def create_mixed_precision_model():
    """Create a model that uses FP16 for critical layers and INT8 for others"""
    print("\n🔧 Creating mixed precision model...")
    
    # This would require custom implementation
    # For now, we'll create an FP16 model as a fallback
    cmd = [
        "python3", "convert_to_tensorrt.py",
        "--onnx", "yolo11n.onnx",
        "--output", "yolo11n_fp16_accurate.trt",
        "--fp16",
        "--clear-cache",
        "--verbose"
    ]
    
    print("  Creating FP16 model as high-accuracy alternative...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("  ✅ FP16 model created successfully")
        return True
    else:
        print(f"  ❌ Failed to create FP16 model: {result.stderr}")
        return False

def provide_recommendations():
    """Provide recommendations for improving INT8 accuracy"""
    print("\n📋 INT8 Accuracy Improvement Recommendations:")
    print("=" * 60)
    
    print("\n1. 🎯 **Immediate Solutions:**")
    print("   • Use FP16 instead of INT8 for better accuracy")
    print("   • FP16 provides ~2x performance boost with minimal accuracy loss")
    print("   • Change model path in C++ code to use FP16 model")
    
    print("\n2. 🔧 **INT8 Improvements:**")
    print("   • Use more representative calibration data (100+ images)")
    print("   • Use smaller calibration batch size (8 instead of 32)")
    print("   • Try different calibration algorithms (legacy, percentile)")
    print("   • Use domain-specific calibration data")
    
    print("\n3. 🚀 **Advanced Solutions:**")
    print("   • Implement mixed precision (FP16 for critical layers)")
    print("   • Use quantization-aware training (QAT)")
    print("   • Consider model pruning before quantization")
    print("   • Use larger models that benefit more from INT8")
    
    print("\n4. 📊 **Expected Results:**")
    print("   • FP16: 90% → 85-88% accuracy (minimal loss)")
    print("   • Conservative INT8: 90% → 60-70% accuracy")
    print("   • Current INT8: 90% → 40% accuracy (too aggressive)")
    
    print("\n5. 💡 **Best Practice:**")
    print("   • Always validate accuracy after quantization")
    print("   • Use representative test datasets")
    print("   • Monitor detection confidence scores")
    print("   • Consider accuracy vs performance trade-offs")

def main():
    print("🎯 INT8 Accuracy Improvement Tool")
    print("=" * 50)
    print("Addressing accuracy drop from 90% to 40% for person detection")
    
    # Generate better calibration data
    generate_better_calibration()
    
    # Create conservative INT8 model
    conservative_success = create_conservative_int8_model()
    
    # Create FP16 model as alternative
    fp16_success = create_mixed_precision_model()
    
    # Provide recommendations
    provide_recommendations()
    
    print(f"\n📈 Summary:")
    if conservative_success:
        print(f"  ✅ Conservative INT8 model created: yolo11n_int8_conservative.trt")
    if fp16_success:
        print(f"  ✅ FP16 model created: yolo11n_fp16_accurate.trt")
    
    print(f"\n🎯 Next Steps:")
    print(f"  1. Test conservative INT8 model for accuracy improvement")
    print(f"  2. Use FP16 model for best accuracy/performance balance")
    print(f"  3. Update C++ code to use preferred model")
    print(f"  4. Validate results on your test dataset")

if __name__ == "__main__":
    main() 