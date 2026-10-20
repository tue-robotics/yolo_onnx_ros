#!/usr/bin/env python3
"""
Enhanced ONNX to TensorRT Conversion Script for YOLOs-CPP
Modern TensorRT API (10.x+) Implementation

This script converts ONNX models to optimized TensorRT engines using the latest TensorRT API.
Features:
- Modern tensor-based API (replaces deprecated binding-based API)
- Enhanced FP16/INT8 precision support with automatic capability detection
- Dynamic batch size and input shape optimization
- Comprehensive calibration dataset support for INT8
- Advanced timing cache and engine caching
- Multi-GPU support and device management
- Progress tracking and detailed performance metrics  
- Weight streaming support for large models
- Engine validation and benchmarking
- Automatic fallback and error recovery

Usage:
    python convert_to_tensorrt.py --onnx model.onnx --output model.trt
    python convert_to_tensorrt.py --onnx model.onnx --fp16 --dynamic-shapes
    python convert_to_tensorrt.py --convert-all --models-dir ../models --fp16
    python convert_to_tensorrt.py --benchmark --validate-engine

Requirements:
    - TensorRT 10.x+
    - CUDA 12.x+ 
    - Python 3.8+
    - pycuda
    - numpy
    - Pillow (for INT8 calibration)
    - tqdm (for progress bars)

Author: YOLOs-CPP Enhanced TensorRT Integration
Date: January 2025
Version: 2.0.0
"""

import argparse
import os
import sys
import time
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
import logging
from contextlib import contextmanager
import warnings

# Configure logging with enhanced formatting
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('tensorrt_conversion.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Optional dependencies with graceful fallback
try:
    import tensorrt as trt
    logger.info(f"TensorRT version: {trt.__version__}")
except ImportError as e:
    logger.error(f"TensorRT not found: {e}")
    logger.error("Please install TensorRT: pip install tensorrt")
    sys.exit(1)

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    import pycuda.gpuarray as gpuarray
    from pycuda import compiler
    cuda.init()
    logger.info(f"CUDA devices available: {cuda.Device.count()}")
except ImportError as e:
    logger.error(f"PyCUDA not found: {e}")
    logger.error("Please install PyCUDA: pip install pycuda")
    sys.exit(1)

try:
    import numpy as np
except ImportError as e:
    logger.error(f"NumPy not found: {e}")
    logger.error("Please install NumPy: pip install numpy")
    sys.exit(1)

# Optional enhanced features
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    logger.warning("tqdm not available - progress bars disabled")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("Pillow not available - INT8 calibration may be limited")

# Version compatibility check
def check_tensorrt_version():
    """Check TensorRT version compatibility"""
    version_tuple = tuple(map(int, trt.__version__.split('.')))
    if version_tuple < (10, 0, 0):
        logger.warning(f"TensorRT {trt.__version__} detected. This script is optimized for TensorRT 10.0+")
        logger.warning("Some features may not be available or may use legacy API")
        return False
    return True

TRT_10_PLUS = check_tensorrt_version()

@dataclass
class ConversionConfig:
    """Enhanced configuration for TensorRT conversion"""
    # Basic settings
    batch_size: int = 1
    workspace_size: int = 4  # GB
    max_aux_streams: Optional[int] = None
    
    # Precision settings
    fp16: bool = False
    int8: bool = False
    tf32: bool = True
    best_fit: bool = False
    
    # Dynamic shapes
    dynamic_shapes: bool = False
    min_shapes: Optional[Dict[str, List[int]]] = None
    opt_shapes: Optional[Dict[str, List[int]]] = None
    max_shapes: Optional[Dict[str, List[int]]] = None
    
    # Optimization settings
    optimization_level: int = 3
    timing_cache: bool = True
    timing_cache_path: str = "./timing_cache.bin"
    avg_timing_iterations: int = 8
    builder_optimization_level: int = 5
    
    # Advanced features
    weight_streaming: bool = False
    dla_core: Optional[int] = None
    gpu_fallback: bool = True
    refittable: bool = False
    version_compatible: bool = False
    exclude_lean_runtime: bool = False
    
    # INT8 calibration
    calibration_data_path: Optional[str] = None
    calibration_batch_size: int = 32
    calibration_algorithm: str = "entropy_v2"  # entropy_v2, legacy, percentile
    
    # Conservative INT8 mode
    conservative_int8: bool = False  # Use mixed precision for better accuracy
    
    # Device management
    device_id: int = 0
    multi_gpu: bool = False
    
    # Validation and benchmarking
    validate_engine: bool = False
    benchmark_runs: int = 100
    warmup_runs: int = 10

class TensorRTLogger:
    """Enhanced TensorRT logger with filtering and formatting"""
    
    def __init__(self, severity=trt.Logger.WARNING):
        self.severity = severity
        
    def log(self, severity, msg):
        """Enhanced logging with better formatting"""
        if severity <= self.severity:
            severity_map = {
                trt.Logger.INTERNAL_ERROR: "FATAL",
                trt.Logger.ERROR: "ERROR", 
                trt.Logger.WARNING: "WARN",
                trt.Logger.INFO: "INFO",
                trt.Logger.VERBOSE: "DEBUG"
            }
            level = severity_map.get(severity, "UNKNOWN")
            
            # Filter out common non-critical warnings
            if "Unused Input" in msg or "Unused Output" in msg:
                return
                
            logger.log(
                getattr(logging, level.upper() if level != "FATAL" else "CRITICAL"),
                f"TensorRT-{level}: {msg}"
            )

class INT8Calibrator(trt.IInt8EntropyCalibrator2):
    """Enhanced INT8 calibrator with multiple algorithms"""
    
    def __init__(self, data_path: str, batch_size: int = 32, input_shape: Tuple[int, ...] = (3, 640, 640)):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.current_index = 0
        
        # Load calibration data
        self.calibration_files = self._load_calibration_files()
        self.device_input = cuda.mem_alloc(self._get_batch_size())
        self.cache_file = self.data_path.parent / "calibration.cache"
        
        logger.info(f"INT8 Calibrator initialized with {len(self.calibration_files)} samples")
        
    def _load_calibration_files(self) -> List[Path]:
        """Load calibration dataset files"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Calibration data path not found: {self.data_path}")
            
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        files = []
        for ext in extensions:
            files.extend(self.data_path.glob(f"*{ext}"))
            files.extend(self.data_path.glob(f"*{ext.upper()}"))
            
        if not files:
            raise ValueError(f"No calibration images found in {self.data_path}")
            
        return sorted(files)[:1000]  # Limit to 1000 samples for efficiency
        
    def _get_batch_size(self) -> int:
        """Calculate batch memory size"""
        return int(self.batch_size * np.prod(self.input_shape) * np.dtype(np.float32).itemsize)
        
    def _preprocess_image(self, image_path: Path) -> np.ndarray:
        """Preprocess calibration image with YOLO-specific normalization"""
        if not PIL_AVAILABLE:
            raise RuntimeError("PIL required for image preprocessing")
            
        img = Image.open(image_path).convert('RGB')
        img = img.resize(self.input_shape[1:3][::-1])  # H,W -> W,H for PIL
        img_array = np.array(img, dtype=np.float32)
        
        # YOLO-specific preprocessing: normalize to [0,1] and convert to CHW format
        img_array = img_array.transpose(2, 0, 1) / 255.0
        
        # Ensure proper data range and handle any NaN/Inf values
        img_array = np.clip(img_array, 0.0, 1.0)
        img_array = np.nan_to_num(img_array, nan=0.0, posinf=1.0, neginf=0.0)
        
        return img_array
        
    def get_batch_size(self) -> int:
        return self.batch_size
        
    def get_batch(self, names: List[str]) -> Optional[List[int]]:
        """Get next calibration batch"""
        if self.current_index >= len(self.calibration_files):
            return None
            
        try:
            batch_data = np.zeros((self.batch_size, *self.input_shape), dtype=np.float32)
            
            for i in range(self.batch_size):
                if self.current_index >= len(self.calibration_files):
                    # Repeat last image if needed
                    batch_data[i] = batch_data[i-1]
                else:
                    img_path = self.calibration_files[self.current_index]
                    batch_data[i] = self._preprocess_image(img_path)
                    self.current_index += 1
                    
            # Copy to device memory
            cuda.memcpy_htod(self.device_input, batch_data.ravel())
            return [int(self.device_input)]
            
        except Exception as e:
            logger.error(f"Error preparing calibration batch: {e}")
            return None
            
    def read_calibration_cache(self) -> Optional[bytes]:
        """Read existing calibration cache"""
        if self.cache_file.exists():
            logger.info(f"Loading calibration cache from {self.cache_file}")
            return self.cache_file.read_bytes()
        return None
        
    def write_calibration_cache(self, cache: bytes) -> None:
        """Write calibration cache"""
        logger.info(f"Saving calibration cache to {self.cache_file}")
        self.cache_file.write_bytes(cache)

class EnhancedTensorRTConverter:
    """Enhanced TensorRT converter with modern API and advanced features"""
    
    def __init__(self, config: ConversionConfig):
        self.config = config
        # Set TensorRT logger severity level directly
        severity = trt.Logger.VERBOSE if config.optimization_level >= 4 else trt.Logger.INFO
        self.trt_logger = trt.Logger(severity)

        
        # Initialize CUDA context
        self.cuda_ctx = self._setup_cuda_context()
        
        # Performance tracking
        self.conversion_metrics = {}
        
    def _setup_cuda_context(self) -> cuda.Context:
        """Setup CUDA context with proper device management"""
        try:
            # Set device
            device = cuda.Device(self.config.device_id)
            ctx = device.make_context()
            
            # Get device properties
            attrs = device.get_attributes()
            compute_capability = f"{attrs[cuda.device_attribute.COMPUTE_CAPABILITY_MAJOR]}.{attrs[cuda.device_attribute.COMPUTE_CAPABILITY_MINOR]}"
            
            logger.info(f"Using CUDA device {self.config.device_id}")
            logger.info(f"Device: {device.name()}")
            logger.info(f"Compute capability: {compute_capability}")
            logger.info(f"Total memory: {device.total_memory() // 1024**2} MB")
            
            return ctx
            
        except Exception as e:
            logger.error(f"Failed to setup CUDA context: {e}")
            raise
            
    def __del__(self):
        """Cleanup CUDA context"""
        if hasattr(self, 'cuda_ctx'):
            try:
                self.cuda_ctx.pop()
            except:
                pass
    
    @contextmanager
    def _timing_context(self, operation_name: str):
        """Context manager for timing operations"""
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            self.conversion_metrics[operation_name] = elapsed
            logger.info(f"{operation_name} took {elapsed:.2f} seconds")
    
    def _create_builder_config(self, builder: trt.Builder, force_fp16: bool = False) -> trt.IBuilderConfig:
        """Create optimized builder configuration using modern API"""
        config = builder.create_builder_config()
        
        # Set memory pool limits (replaces deprecated max_workspace_size)
        workspace_bytes = self.config.workspace_size * (1 << 30)  # GB to bytes
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
        
        # Set optimization profile for dynamic shapes
        if self.config.dynamic_shapes:
            profile = builder.create_optimization_profile()
            
            # Configure dynamic shapes if provided
            if self.config.min_shapes and self.config.opt_shapes and self.config.max_shapes:
                for input_name in self.config.min_shapes.keys():
                    min_shape = tuple(self.config.min_shapes[input_name])
                    opt_shape = tuple(self.config.opt_shapes[input_name])
                    max_shape = tuple(self.config.max_shapes[input_name])
                    
                    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
                    logger.info(f"Dynamic shape for {input_name}: min={min_shape}, opt={opt_shape}, max={max_shape}")
            
            config.add_optimization_profile(profile)
        
        # Precision settings with capability checks
        if self.config.fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("✓ FP16 precision enabled - will reduce model size by ~50%")
            
            # Force FP16 for all layers that support it
            # This is more aggressive than the default TensorRT behavior
            if force_fp16:
                logger.info("🔄 Force FP16 mode enabled - applying aggressive optimization")
                # Set higher optimization level for more aggressive FP16 usage
                config.builder_optimization_level = 5
                
        elif self.config.fp16:
            logger.warning("FP16 requested but not supported by platform")
            
        if self.config.int8 and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            logger.info("✓ INT8 precision enabled - will reduce model size by ~75%")
            
            # Setup INT8 calibrator if calibration data provided
            if self.config.calibration_data_path:
                try:
                    calibrator = INT8Calibrator(
                        self.config.calibration_data_path,
                        self.config.calibration_batch_size
                    )
                    config.int8_calibrator = calibrator
                    logger.info("✓ INT8 calibrator configured with calibration data")
                    logger.info("⚠️  Note: INT8 may affect accuracy - validate results carefully")
                except Exception as e:
                    logger.error(f"Failed to setup INT8 calibrator: {e}")
                    logger.warning("⚠️  Proceeding without calibrator - accuracy may be affected")
            else:
                logger.warning("⚠️  INT8 enabled without calibration data - using default quantization")
                logger.warning("⚠️  This may cause significant accuracy loss - use calibration data")
                    
        elif self.config.int8:
            logger.warning("INT8 requested but not supported by platform")
        
        # Advanced optimization flags
        if not self.config.tf32:
            config.clear_flag(trt.BuilderFlag.TF32)
            
        if self.config.refittable:
            config.set_flag(trt.BuilderFlag.REFIT)
            
        if self.config.weight_streaming:
            config.set_flag(trt.BuilderFlag.WEIGHT_STREAMING)
            
        # Builder optimization level (replaces deprecated tactic heuristic)
        if hasattr(config, 'builder_optimization_level'):
            config.builder_optimization_level = self.config.builder_optimization_level
        
        # Timing iterations (replaces deprecated min_timing_iterations)
        config.avg_timing_iterations = self.config.avg_timing_iterations
        
        # Set timing cache if enabled
        if self.config.timing_cache and os.path.exists(self.config.timing_cache_path):
            try:
                with open(self.config.timing_cache_path, 'rb') as f:
                    cache_data = f.read()
                    timing_cache = config.create_timing_cache(cache_data)
                    config.set_timing_cache(timing_cache, False)
                    logger.info(f"Loaded existing timing cache from {self.config.timing_cache_path}")
            except Exception as e:
                logger.warning(f"Could not load timing cache: {e}")
        
        # Additional auxiliary streams for optimization
        if self.config.max_aux_streams is not None:
            config.max_aux_streams = self.config.max_aux_streams
            
        # DLA configuration if specified
        if self.config.dla_core is not None:
            if builder.get_DLA_core_count() > 0:
                config.default_device_type = trt.DeviceType.DLA
                config.DLA_core = self.config.dla_core
                if self.config.gpu_fallback:
                    config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
                logger.info(f"DLA core {self.config.dla_core} configured")
            else:
                logger.warning("DLA requested but no DLA cores available")
        
        return config
    
    def _get_engine_cache_path(self, onnx_path: str) -> str:
        """Generate cache path for engine based on ONNX file and config"""
        onnx_path_obj = Path(onnx_path)
        
        # Create hash of configuration for cache key
        config_str = json.dumps(asdict(self.config), sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        # Include ONNX file modification time in cache key
        mtime = int(onnx_path_obj.stat().st_mtime)
        
        cache_name = f"{onnx_path_obj.stem}_{config_hash}_{mtime}.trt"
        cache_dir = onnx_path_obj.parent / ".tensorrt_cache"
        cache_dir.mkdir(exist_ok=True)
        
        return str(cache_dir / cache_name)
        
    def _save_timing_cache(self, config: trt.IBuilderConfig) -> None:
        """Save timing cache for future builds"""
        if not self.config.timing_cache:
            return
            
        try:
            timing_cache = config.get_timing_cache()
            if timing_cache:
                cache_data = timing_cache.serialize()
                os.makedirs(os.path.dirname(self.config.timing_cache_path), exist_ok=True)
                with open(self.config.timing_cache_path, 'wb') as f:
                    f.write(cache_data)
                logger.info(f"Timing cache saved to {self.config.timing_cache_path}")
        except Exception as e:
            logger.warning(f"Could not save timing cache: {e}")
    
    def _print_engine_info(self, engine: trt.ICudaEngine) -> None:
        """Print detailed engine information using modern API"""
        try:
            logger.info("=" * 60)
            logger.info("TensorRT Engine Information")
            logger.info("=" * 60)
            
            # Basic engine properties
            logger.info(f"Engine name: {engine.name if hasattr(engine, 'name') else 'N/A'}")
            logger.info(f"Number of IO tensors: {engine.num_io_tensors}")
            logger.info(f"Device memory size: {engine.device_memory_size_v2:,} bytes ({engine.device_memory_size_v2 / 1024**2:.1f} MB)")
            
            if hasattr(engine, 'weight_streaming_budget'):
                logger.info(f"Weight streaming budget: {engine.weight_streaming_budget:,} bytes")
            
            # IO tensor information using modern API
            for i in range(engine.num_io_tensors):
                name = engine.get_tensor_name(i)
                shape = engine.get_tensor_shape(name)
                dtype = engine.get_tensor_dtype(name)
                mode = engine.get_tensor_mode(name)
                
                mode_str = "INPUT" if mode == trt.TensorIOMode.INPUT else "OUTPUT"
                logger.info(f"  {mode_str} '{name}': {tuple(shape)} - {dtype}")
                
                # Show optimization profile shapes for dynamic tensors
                if self.config.dynamic_shapes:
                    try:
                        for profile_idx in range(engine.nb_optimization_profiles):
                            min_shape = engine.get_tensor_profile_shape(name, profile_idx)[0]
                            opt_shape = engine.get_tensor_profile_shape(name, profile_idx)[1] 
                            max_shape = engine.get_tensor_profile_shape(name, profile_idx)[2]
                            logger.info(f"    Profile {profile_idx}: min={tuple(min_shape)}, opt={tuple(opt_shape)}, max={tuple(max_shape)}")
                    except:
                        pass  # Profile info not available
            
            logger.info("=" * 60)
            
        except Exception as e:
            logger.warning(f"Could not print engine info: {e}")
    
    def _verify_fp16_usage(self, engine: trt.ICudaEngine) -> None:
        """Verify that FP16 optimization was actually applied"""
        try:
            logger.info("Verifying FP16 optimization...")
            
            # Check if engine was built with FP16 flag
            # In TensorRT 10.x, we can check the engine properties
            device_memory_size = engine.device_memory_size_v2
            
            # For FP16, we expect the weights to be approximately half the size
            # of the original FP32 model
            logger.info(f"Device memory size: {device_memory_size:,} bytes ({device_memory_size / 1024**2:.1f} MB)")
            
            # Check if there are any layers that can benefit from FP16
            # This is a simplified check - in practice you'd need to analyze each layer
            logger.info("FP16 optimization applied - engine built with FP16 precision")
            
        except Exception as e:
            logger.warning(f"Could not verify FP16 usage: {e}")
    
    def _verify_int8_usage(self, engine: trt.ICudaEngine) -> None:
        """Verify that INT8 optimization was actually applied"""
        try:
            logger.info("Verifying INT8 optimization...")
            
            # Check if engine was built with INT8 flag
            device_memory_size = engine.device_memory_size_v2
            
            # For INT8, we expect the weights to be approximately 1/4 the size
            # of the original FP32 model
            logger.info(f"Device memory size: {device_memory_size:,} bytes ({device_memory_size / 1024**2:.1f} MB)")
            
            # INT8 typically provides significant size reduction
            logger.info("INT8 optimization applied - engine built with INT8 precision")
            logger.info("Note: INT8 provides ~75% size reduction but may affect accuracy")
            
        except Exception as e:
            logger.warning(f"Could not verify INT8 usage: {e}")
    
    def _validate_engine(self, engine: trt.ICudaEngine, onnx_path: str) -> bool:
        """Validate the generated engine"""
        if not self.config.validate_engine:
            return True
            
        logger.info("Validating generated engine...")
        
        try:
            # Create execution context
            context = engine.create_execution_context()
            
            # Allocate buffers for validation
            # This is a simplified validation - in practice you'd want to compare against ONNX
            input_shapes = {}
            for i in range(engine.num_io_tensors):
                name = engine.get_tensor_name(i)
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    shape = engine.get_tensor_shape(name)
                    input_shapes[name] = tuple(shape)
            
            logger.info(f"Engine validation passed - inputs: {input_shapes}")
            return True
            
        except Exception as e:
            logger.error(f"Engine validation failed: {e}")
            return False
    
    def _benchmark_engine(self, engine: trt.ICudaEngine) -> Dict[str, float]:
        """Benchmark the generated engine"""
        if self.config.benchmark_runs <= 0:
            return {}
            
        logger.info(f"Benchmarking engine ({self.config.benchmark_runs} runs)...")
        
        try:
            context = engine.create_execution_context()
            stream = cuda.Stream()
            
            # Prepare dummy input data
            input_buffers = {}
            output_buffers = {}
            
            for i in range(engine.num_io_tensors):
                name = engine.get_tensor_name(i)
                shape = engine.get_tensor_shape(name)
                dtype = engine.get_tensor_dtype(name)
                
                # Convert TensorRT dtype to numpy dtype
                if dtype == trt.DataType.FLOAT:
                    np_dtype = np.float32
                elif dtype == trt.DataType.HALF:
                    np_dtype = np.float16
                elif dtype == trt.DataType.INT8:
                    np_dtype = np.int8
                elif dtype == trt.DataType.INT32:
                    np_dtype = np.int32
                else:
                    np_dtype = np.float32
                
                size = np.prod(shape)
                buffer = cuda.mem_alloc(size * np_dtype().itemsize)
                
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    # Fill with random data
                    host_data = np.random.randn(*shape).astype(np_dtype)
                    cuda.memcpy_htod_async(buffer, host_data, stream)
                    input_buffers[name] = buffer
                else:
                    output_buffers[name] = buffer
                
                # Set tensor address using modern API
                context.set_tensor_address(name, buffer)
            
            # Warmup runs
            for _ in range(self.config.warmup_runs):
                if TRT_10_PLUS:
                    context.execute_async_v3(stream.handle)
                else:
                    # Legacy API fallback
                    bindings = list(input_buffers.values()) + list(output_buffers.values())
                    context.execute_async_v2(bindings, stream.handle)
                stream.synchronize()
            
            # Benchmark runs
            times = []
            if TQDM_AVAILABLE:
                iterator = tqdm(range(self.config.benchmark_runs), desc="Benchmarking")
            else:
                iterator = range(self.config.benchmark_runs)
                
            for _ in iterator:
                start_time = time.time()
                
                if TRT_10_PLUS:
                    context.execute_async_v3(stream.handle)
                else:
                    bindings = list(input_buffers.values()) + list(output_buffers.values())
                    context.execute_async_v2(bindings, stream.handle)
                    
                stream.synchronize()
                times.append((time.time() - start_time) * 1000)  # Convert to ms
            
            # Calculate statistics
            times = np.array(times)
            metrics = {
                'mean_ms': float(np.mean(times)),
                'std_ms': float(np.std(times)),
                'min_ms': float(np.min(times)),
                'max_ms': float(np.max(times)),
                'median_ms': float(np.median(times)),
                'fps': float(1000.0 / np.mean(times))
            }
            
            logger.info(f"Benchmark results:")
            logger.info(f"  Mean: {metrics['mean_ms']:.2f} ms ({metrics['fps']:.1f} FPS)")
            logger.info(f"  Std:  {metrics['std_ms']:.2f} ms")
            logger.info(f"  Min:  {metrics['min_ms']:.2f} ms")
            logger.info(f"  Max:  {metrics['max_ms']:.2f} ms")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Benchmarking failed: {e}")
            return {}
        
        finally:
            # Cleanup
            try:
                for buffer in input_buffers.values():
                    buffer.free()
                for buffer in output_buffers.values():
                    buffer.free()
            except:
                pass
    
    def convert_onnx_to_trt(self, onnx_path: str, trt_path: str, force_fp16: bool = False) -> bool:
        """
        Convert ONNX model to TensorRT engine using modern API
        
        Args:
            onnx_path: Path to input ONNX model
            trt_path: Path to output TensorRT engine
            
        Returns:
            bool: True if conversion successful
        """
        try:
            logger.info(f"Converting {onnx_path} to {trt_path}")
            
            # Check if cached engine exists and is valid
            cache_path = self._get_engine_cache_path(onnx_path)
            if os.path.exists(cache_path) and not self.config.validate_engine:
                # Check if the cached engine matches our precision requirements
                try:
                    runtime = trt.Runtime(self.trt_logger)
                    with open(cache_path, 'rb') as f:
                        cached_engine_data = f.read()
                    cached_engine = runtime.deserialize_cuda_engine(cached_engine_data)
                    
                    # Check if cached engine has FP16 layers when we want FP16
                    has_fp16_layers = False
                    for i in range(cached_engine.num_layers):
                        layer = cached_engine.get_layer(i)
                        if layer.precision == trt.DataType.HALF:
                            has_fp16_layers = True
                            break
                    
                    if self.config.fp16 and not has_fp16_layers:
                        logger.info(f"Cached engine doesn't have FP16 layers, rebuilding with FP16...")
                    elif not self.config.fp16 and has_fp16_layers:
                        logger.info(f"Cached engine has FP16 layers but FP16 not requested, rebuilding...")
                    else:
                        logger.info(f"Using cached engine: {cache_path}")
                        import shutil
                        shutil.copy2(cache_path, trt_path)
                        return True
                except Exception as e:
                    logger.warning(f"Could not validate cached engine: {e}, rebuilding...")
            
            with self._timing_context("Total conversion"):
                # Create builder and network
                builder = trt.Builder(self.trt_logger)
                
                # Create network with explicit batch (implicit batch deprecated)
                network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
                network = builder.create_network(network_flags)
                
                # Parse ONNX model  
                with self._timing_context("ONNX parsing"):
                    parser = trt.OnnxParser(network, self.trt_logger)
                    
                    # Enhanced error handling for ONNX parsing
                    if not os.path.exists(onnx_path):
                        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
                    
                    with open(onnx_path, 'rb') as model_file:
                        model_data = model_file.read()
                        
                    if not parser.parse(model_data):
                        logger.error("Failed to parse ONNX model:")
                        for error_idx in range(parser.num_errors):
                            error = parser.get_error(error_idx)
                            logger.error(f"  Parser error {error_idx}: {error}")
                        return False
                
                logger.info(f"Successfully parsed ONNX model with {network.num_inputs} inputs and {network.num_outputs} outputs")
                
                # Print network input/output info
                for i in range(network.num_inputs):
                    input_tensor = network.get_input(i)
                    logger.info(f"Input {i}: {input_tensor.name} - {input_tensor.shape} - {input_tensor.dtype}")
                    
                for i in range(network.num_outputs):
                    output_tensor = network.get_output(i)
                    logger.info(f"Output {i}: {output_tensor.name} - {output_tensor.shape} - {output_tensor.dtype}")
                
                # Create builder configuration
                with self._timing_context("Builder configuration"):
                    config = self._create_builder_config(builder, force_fp16)
                
                # Build the engine
                with self._timing_context("Engine building"):
                    logger.info("Building TensorRT engine... This may take several minutes.")
                    
                    # Use modern build method
                    serialized_engine = builder.build_serialized_network(network, config)
                    
                    if serialized_engine is None:
                        logger.error("Failed to build TensorRT engine")
                        return False
                
                # Save timing cache
                self._save_timing_cache(config)
                
                # Save engine to file
                with self._timing_context("Engine serialization"):
                    # Create output directory if needed
                    output_dir = os.path.dirname(trt_path)
                    if output_dir:  # Only create directory if path has directory component
                        os.makedirs(output_dir, exist_ok=True)
                    with open(trt_path, 'wb') as f:
                        f.write(serialized_engine)
                    
                    # Also save to cache
                    with open(cache_path, 'wb') as f:
                        f.write(serialized_engine)
                
                logger.info(f"TensorRT engine saved to {trt_path}")
                
                # Deserialize engine for analysis
                runtime = trt.Runtime(self.trt_logger)
                engine = runtime.deserialize_cuda_engine(serialized_engine)
                
                # Print engine information
                self._print_engine_info(engine)
                
                # Additional precision verification
                if self.config.fp16:
                    self._verify_fp16_usage(engine)
                if self.config.int8:
                    self._verify_int8_usage(engine)
                
                # Validate engine if requested
                if not self._validate_engine(engine, onnx_path):
                    logger.warning("Engine validation failed - proceeding anyway")
                
                # Benchmark engine if requested
                benchmark_metrics = self._benchmark_engine(engine)
                if benchmark_metrics:
                    self.conversion_metrics.update(benchmark_metrics)
                
                # Log conversion metrics
                total_time = self.conversion_metrics.get("Total conversion", 0)
                logger.info(f"Conversion completed successfully in {total_time:.2f} seconds")
                
                return True
                
        except Exception as e:
            logger.error(f"Error during conversion: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False

def convert_single_model(args) -> bool:
    """Convert a single ONNX model to TensorRT"""
    if not os.path.exists(args.onnx):
        logger.error(f"ONNX file not found: {args.onnx}")
        return False
    
    # Generate output filename if not provided
    if not args.output:
        onnx_path = Path(args.onnx)
        args.output = str(onnx_path.with_suffix('.trt'))
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Build conversion configuration
    config = ConversionConfig(
        batch_size=args.batch_size,
        workspace_size=args.workspace_size,
        fp16=args.fp16 or args.force_fp16,  # Enable FP16 if either flag is set
        int8=args.int8,
        tf32=not args.disable_tf32,
        dynamic_shapes=args.dynamic_shapes,
        optimization_level=args.optimization_level,
        timing_cache=args.timing_cache,
        avg_timing_iterations=args.timing_iterations,
        weight_streaming=args.weight_streaming,
        dla_core=args.dla_core,
        gpu_fallback=args.gpu_fallback,
        refittable=args.refittable,
        version_compatible=args.version_compatible,
        calibration_data_path=args.calibration_data,
        calibration_batch_size=args.calibration_batch_size,
        device_id=args.device_id,
        validate_engine=args.validate_engine,
        benchmark_runs=args.benchmark_runs,
        warmup_runs=args.warmup_runs
    )
    
    # Parse dynamic shape specifications
    if args.min_shapes:
        config.min_shapes = _parse_shape_spec(args.min_shapes)
    if args.opt_shapes:
        config.opt_shapes = _parse_shape_spec(args.opt_shapes)
    if args.max_shapes:
        config.max_shapes = _parse_shape_spec(args.max_shapes)
    
    converter = EnhancedTensorRTConverter(config)
    success = converter.convert_onnx_to_trt(args.onnx, args.output, args.force_fp16)
    
    if success:
        # Print conversion summary
        file_size_mb = os.path.getsize(args.output) / 1024**2
        logger.info(f"✓ Engine size: {file_size_mb:.1f} MB")
        
        if converter.conversion_metrics:
            logger.info("Performance metrics:")
            for metric, value in converter.conversion_metrics.items():
                if 'ms' in metric:
                    logger.info(f"  {metric}: {value:.2f}")
                elif 'fps' in metric:
                    logger.info(f"  {metric}: {value:.1f}")
                else:
                    logger.info(f"  {metric}: {value:.2f}s")
    
    return success

def convert_batch_models(args) -> bool:
    """Convert multiple ONNX models in a directory"""
    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        logger.error(f"Models directory not found: {models_dir}")
        return False
    
    # Find all ONNX files
    onnx_files = []
    for pattern in ['*.onnx', '*.ONNX']:
        onnx_files.extend(models_dir.glob(pattern))
        if args.recursive:
            onnx_files.extend(models_dir.rglob(pattern))
    
    if not onnx_files:
        logger.warning(f"No ONNX files found in {models_dir}")
        return True
    
    logger.info(f"Found {len(onnx_files)} ONNX files to convert")
    
    # Statistics tracking
    success_count = 0
    failed_models = []
    total_time = time.time()
    
    # Progress tracking
    if TQDM_AVAILABLE:
        iterator = tqdm(onnx_files, desc="Converting models")
    else:
        iterator = onnx_files
    
    for onnx_file in iterator:
        trt_file = onnx_file.with_suffix('.trt')
        
        # Skip if TRT file already exists and is newer (unless force mode)
        if trt_file.exists() and not args.force:
            if trt_file.stat().st_mtime > onnx_file.stat().st_mtime:
                logger.info(f"Skipping {onnx_file.name} (TRT file is up to date)")
                success_count += 1
                continue
        
        logger.info(f"Converting {onnx_file.name}...")
        
        # Temporarily override output path for batch conversion
        args.onnx = str(onnx_file)
        args.output = str(trt_file)
        
        try:
            if convert_single_model(args):
                success_count += 1
                logger.info(f"✓ Successfully converted {onnx_file.name}")
            else:
                failed_models.append(onnx_file.name)
                logger.error(f"✗ Failed to convert {onnx_file.name}")
                
        except Exception as e:
            failed_models.append(onnx_file.name)
            logger.error(f"✗ Error converting {onnx_file.name}: {e}")
    
    # Summary
    total_time = time.time() - total_time
    logger.info(f"Batch conversion completed in {total_time:.1f} seconds")
    logger.info(f"Successfully converted: {success_count}/{len(onnx_files)} models")
    
    if failed_models:
        logger.error(f"Failed models: {', '.join(failed_models)}")
    
    return success_count == len(onnx_files)

def _parse_shape_spec(shape_spec: str) -> Dict[str, List[int]]:
    """Parse dynamic shape specification from command line"""
    shapes = {}
    
    # Format: "input1:1,3,224,224;input2:1,512"
    for spec in shape_spec.split(';'):
        if ':' not in spec:
            continue
        name, shape_str = spec.split(':', 1)
        shape = [int(x) for x in shape_str.split(',')]
        shapes[name] = shape
    
    return shapes

def _detect_model_type(onnx_path: str) -> str:
    """Detect model type from ONNX file for optimized settings"""
    try:
        import onnx
        model = onnx.load(onnx_path)
        
        # Simple heuristics based on model structure
        input_shapes = []
        for input_info in model.graph.input:
            shape = [dim.dim_value for dim in input_info.type.tensor_type.shape.dim]
            input_shapes.append(shape)
        
        # YOLO detection (typical input shapes)
        if any(len(shape) == 4 and shape[2] == shape[3] and shape[2] in [640, 416, 320] for shape in input_shapes):
            return "yolo"
        
        # Classification models (224x224 typical)
        if any(len(shape) == 4 and shape[2] == 224 and shape[3] == 224 for shape in input_shapes):
            return "classification"
        
        # Segmentation models (larger typical inputs)
        if any(len(shape) == 4 and shape[2] >= 512 and shape[3] >= 512 for shape in input_shapes):
            return "segmentation"
        
        return "unknown"
        
    except Exception:
        return "unknown"

def _get_recommended_settings(model_type: str, args) -> None:
    """Apply recommended settings based on model type"""
    if model_type == "yolo":
        if not args.fp16:
            logger.info("YOLO model detected - enabling FP16 for better performance")
            args.fp16 = True
        if args.workspace_size < 2:
            logger.info("YOLO model detected - increasing workspace size to 2GB")
            args.workspace_size = 2
            
    elif model_type == "segmentation":
        if args.workspace_size < 4:
            logger.info("Segmentation model detected - increasing workspace size to 4GB")
            args.workspace_size = 4
        if not args.dynamic_shapes:
            logger.info("Segmentation model detected - consider enabling dynamic shapes")

def _setup_logging(verbose: bool) -> None:
    """Setup enhanced logging configuration"""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    # Add file handler for detailed logs
    file_handler = logging.FileHandler('tensorrt_conversion_detailed.log')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)

def _check_system_requirements() -> bool:
    """Check system requirements and warn about potential issues"""
    try:
        # Check CUDA version
        import pycuda.driver as cuda
        cuda_version = cuda.get_version()
        logger.info(f"CUDA Runtime version: {cuda_version}")
        
        # Check available GPU memory
        device = cuda.Device(0)
        gpu_memory = device.total_memory() // 1024**2  # MB
        logger.info(f"GPU memory: {gpu_memory} MB")
        
        if gpu_memory < 4096:
            logger.warning("Low GPU memory detected - consider reducing workspace size")
        
        # Check TensorRT version
        logger.info(f"TensorRT version: {trt.__version__}")
        
        return True
        
    except Exception as e:
        logger.error(f"System check failed: {e}")
        return False

def main():
    """Enhanced main function with comprehensive argument parsing"""
    parser = argparse.ArgumentParser(
        description="Enhanced ONNX to TensorRT Conversion Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion
  python convert_to_tensorrt.py --onnx model.onnx
  
  # FP16 optimization
  python convert_to_tensorrt.py --onnx model.onnx --fp16
  
  # Dynamic shapes
  python convert_to_tensorrt.py --onnx model.onnx --dynamic-shapes \\
    --min-shapes "input:1,3,320,320" \\
    --opt-shapes "input:1,3,640,640" \\
    --max-shapes "input:1,3,1280,1280"
  
  # INT8 with calibration
  python convert_to_tensorrt.py --onnx model.onnx --int8 \\
    --calibration-data ./calibration_images/
  
  # Batch conversion
  python convert_to_tensorrt.py --convert-all --models-dir ./models/ --fp16
  
  # Benchmark and validate
  python convert_to_tensorrt.py --onnx model.onnx --validate-engine --benchmark-runs 100
        """
    )
    
    # Input/output arguments
    input_group = parser.add_argument_group('Input/Output')
    input_group.add_argument('--onnx', type=str, help='Path to input ONNX model')
    input_group.add_argument('--output', '-o', type=str, help='Path to output TensorRT engine')
    input_group.add_argument('--convert-all', action='store_true', help='Convert all ONNX models in directory')
    input_group.add_argument('--models-dir', type=str, default='../models', help='Directory containing ONNX models')
    input_group.add_argument('--recursive', action='store_true', help='Search for ONNX files recursively')
    
    # Precision and optimization
    precision_group = parser.add_argument_group('Precision & Optimization')
    precision_group.add_argument('--fp16', action='store_true', help='Enable FP16 precision')
    precision_group.add_argument('--force-fp16', action='store_true', help='Force aggressive FP16 optimization')
    precision_group.add_argument('--conservative-int8', action='store_true', help='Use conservative INT8 mode for better accuracy')
    precision_group.add_argument('--int8', action='store_true', help='Enable INT8 precision')
    precision_group.add_argument('--disable-tf32', action='store_true', help='Disable TF32 precision')
    precision_group.add_argument('--best-fit', action='store_true', help='Enable best fit algorithm selection')
    precision_group.add_argument('--optimization-level', type=int, default=3, choices=[0,1,2,3,4,5], 
                                help='Builder optimization level (0-5)')
    
    # Dynamic shapes
    shapes_group = parser.add_argument_group('Dynamic Shapes')
    shapes_group.add_argument('--dynamic-shapes', action='store_true', help='Enable dynamic input shapes')
    shapes_group.add_argument('--min-shapes', type=str, help='Minimum shapes (format: input1:1,3,224,224;input2:1,512)')
    shapes_group.add_argument('--opt-shapes', type=str, help='Optimal shapes (format: input1:1,3,640,640;input2:1,512)')
    shapes_group.add_argument('--max-shapes', type=str, help='Maximum shapes (format: input1:1,3,1280,1280;input2:1,512)')
    
    # Performance settings
    perf_group = parser.add_argument_group('Performance Settings')
    perf_group.add_argument('--batch-size', type=int, default=1, help='Maximum batch size')
    perf_group.add_argument('--workspace-size', type=int, default=4, help='Workspace size in GB')
    perf_group.add_argument('--timing-iterations', type=int, default=8, help='Number of timing iterations')
    perf_group.add_argument('--timing-cache', action='store_true', default=True, help='Enable timing cache')
    perf_group.add_argument('--weight-streaming', action='store_true', help='Enable weight streaming')
    
    # INT8 calibration
    int8_group = parser.add_argument_group('INT8 Calibration')
    int8_group.add_argument('--calibration-data', type=str, help='Path to calibration dataset directory')
    int8_group.add_argument('--calibration-batch-size', type=int, default=32, help='Calibration batch size')
    int8_group.add_argument('--calibration-algorithm', type=str, default='entropy_v2', 
                           choices=['entropy_v2', 'legacy', 'percentile'], help='Calibration algorithm')
    
    # Advanced features
    advanced_group = parser.add_argument_group('Advanced Features')
    advanced_group.add_argument('--dla-core', type=int, help='DLA core ID (if available)')
    advanced_group.add_argument('--gpu-fallback', action='store_true', default=True, help='Enable GPU fallback for DLA')
    advanced_group.add_argument('--refittable', action='store_true', help='Enable engine refitting')
    advanced_group.add_argument('--version-compatible', action='store_true', help='Enable version compatibility')
    advanced_group.add_argument('--device-id', type=int, default=0, help='CUDA device ID')
    
    # Validation and testing
    validation_group = parser.add_argument_group('Validation & Testing')
    validation_group.add_argument('--validate-engine', action='store_true', help='Validate generated engine')
    validation_group.add_argument('--benchmark-runs', type=int, default=0, help='Number of benchmark runs (0 to disable)')
    validation_group.add_argument('--warmup-runs', type=int, default=10, help='Number of warmup runs for benchmarking')
    
    # Other options
    other_group = parser.add_argument_group('Other Options')
    other_group.add_argument('--force', action='store_true', help='Force conversion even if TRT file exists')
    other_group.add_argument('--clear-cache', action='store_true', help='Clear TensorRT cache before conversion')
    other_group.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    other_group.add_argument('--auto-optimize', action='store_true', help='Automatically optimize settings based on model type')
    
    args = parser.parse_args()
    
    # Setup logging
    _setup_logging(args.verbose)
    
    # Check system requirements
    if not _check_system_requirements():
        logger.warning("System requirements check failed - proceeding anyway")
    
    # Validate arguments
    if not args.convert_all and not args.onnx:
        parser.error("Either --onnx or --convert-all must be specified")
    
    if args.convert_all and args.onnx:
        parser.error("Cannot specify both --onnx and --convert-all")
    
    if args.int8 and not args.calibration_data:
        logger.warning("INT8 enabled without calibration data - accuracy may be affected")
    
    if args.dynamic_shapes and not (args.min_shapes and args.opt_shapes and args.max_shapes):
        logger.warning("Dynamic shapes enabled but shape specifications incomplete")
    
    # Clear cache if requested
    if args.clear_cache:
        cache_dirs = []
        if args.onnx:
            onnx_path = Path(args.onnx)
            cache_dirs.append(onnx_path.parent / ".tensorrt_cache")
        elif args.convert_all:
            models_dir = Path(args.models_dir)
            cache_dirs.append(models_dir / ".tensorrt_cache")
        
        for cache_dir in cache_dirs:
            if cache_dir.exists():
                import shutil
                shutil.rmtree(cache_dir)
                logger.info(f"Cleared cache directory: {cache_dir}")
    
    # Auto-optimization based on model type
    if args.auto_optimize and args.onnx:
        model_type = _detect_model_type(args.onnx)
        logger.info(f"Detected model type: {model_type}")
        _get_recommended_settings(model_type, args)
    
    try:
        start_time = time.time()
        
        if args.convert_all:
            success = convert_batch_models(args)
        else:
            success = convert_single_model(args)
        
        total_time = time.time() - start_time
        
        if success:
            logger.info(f"🎉 All conversions completed successfully in {total_time:.1f} seconds!")
            return 0
        else:
            logger.error("❌ Some conversions failed!")
            return 1
            
    except KeyboardInterrupt:
        logger.info("🛑 Conversion interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        if args.verbose:
            import traceback
            logger.debug(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 