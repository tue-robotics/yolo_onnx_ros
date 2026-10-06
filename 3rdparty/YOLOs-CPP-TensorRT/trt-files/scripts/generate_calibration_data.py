#!/usr/bin/env python3
"""
Generate synthetic calibration data for INT8 quantization
Creates a dataset of random images that can be used for TensorRT INT8 calibration
"""

import os
import sys
import numpy as np
from pathlib import Path
import argparse

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available. Install with: pip install Pillow")

def generate_synthetic_images(output_dir: str, num_images: int = 100, size: tuple = (640, 640)):
    """Generate synthetic calibration images"""
    if not PIL_AVAILABLE:
        print("Error: PIL required for image generation")
        return False
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_images} synthetic calibration images...")
    
    for i in range(num_images):
        # Generate random image with realistic distribution
        # Use different patterns to simulate real-world data
        
        if i % 4 == 0:
            # Random noise
            img_array = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
        elif i % 4 == 1:
            # Gradient pattern
            x, y = np.meshgrid(np.linspace(0, 255, size[0]), np.linspace(0, 255, size[1]))
            img_array = np.stack([x, y, (x + y) % 256], axis=2).astype(np.uint8)
        elif i % 4 == 2:
            # Checkerboard pattern
            checker = np.indices((size[1] // 32, size[0] // 32)).sum(axis=0) % 2
            checker = np.repeat(np.repeat(checker, 32, axis=0), 32, axis=1)
            img_array = np.stack([checker * 255] * 3, axis=2).astype(np.uint8)
        else:
            # Solid color with noise
            base_color = np.random.randint(0, 256, 3)
            img_array = np.full((size[1], size[0], 3), base_color, dtype=np.uint8)
            noise = np.random.randint(-50, 50, img_array.shape, dtype=np.int16)
            img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Create PIL image and save
        img = Image.fromarray(img_array)
        img_path = output_path / f"calibration_{i:04d}.jpg"
        img.save(img_path, quality=95)
        
        if (i + 1) % 20 == 0:
            print(f"Generated {i + 1}/{num_images} images")
    
    print(f"✓ Generated {num_images} calibration images in {output_path}")
    return True

def copy_existing_images(source_dir: str, output_dir: str, max_images: int = 100):
    """Copy existing images for calibration"""
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    if not source_path.exists():
        print(f"Error: Source directory not found: {source_dir}")
        return False
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find image files
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in extensions:
        image_files.extend(source_path.glob(f"*{ext}"))
        image_files.extend(source_path.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"No image files found in {source_dir}")
        return False
    
    # Limit number of images
    image_files = sorted(image_files)[:max_images]
    
    print(f"Copying {len(image_files)} images for calibration...")
    
    for i, img_path in enumerate(image_files):
        # Copy with new name
        new_name = f"calibration_{i:04d}{img_path.suffix}"
        new_path = output_path / new_name
        
        try:
            # Open and save to ensure compatibility
            img = Image.open(img_path)
            img = img.convert('RGB')  # Ensure RGB format
            img.save(new_path, quality=95)
        except Exception as e:
            print(f"Warning: Could not process {img_path}: {e}")
            continue
        
        if (i + 1) % 20 == 0:
            print(f"Copied {i + 1}/{len(image_files)} images")
    
    print(f"✓ Copied {len(image_files)} calibration images to {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Generate calibration data for INT8 quantization")
    parser.add_argument('--output', '-o', type=str, default='./calibration_data',
                       help='Output directory for calibration images')
    parser.add_argument('--num-images', '-n', type=int, default=100,
                       help='Number of synthetic images to generate')
    parser.add_argument('--size', type=str, default='640,640',
                       help='Image size (width,height)')
    parser.add_argument('--copy-from', type=str,
                       help='Copy existing images from directory instead of generating synthetic ones')
    parser.add_argument('--max-copy', type=int, default=100,
                       help='Maximum number of images to copy')
    
    args = parser.parse_args()
    
    # Parse size
    try:
        width, height = map(int, args.size.split(','))
        size = (width, height)
    except ValueError:
        print("Error: Invalid size format. Use 'width,height' (e.g., '640,640')")
        return 1
    
    if args.copy_from:
        success = copy_existing_images(args.copy_from, args.output, args.max_copy)
    else:
        success = generate_synthetic_images(args.output, args.num_images, size)
    
    if success:
        print(f"\n🎉 Calibration data ready!")
        print(f"📁 Location: {args.output}")
        print(f"💡 Use with: --calibration-data {args.output}")
        return 0
    else:
        print("❌ Failed to generate calibration data")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 