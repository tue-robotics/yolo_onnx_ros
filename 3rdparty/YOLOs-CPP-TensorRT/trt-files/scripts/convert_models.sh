#!/bin/bash

# ONNX to TensorRT Conversion Script for YOLOs-CPP
# This script converts ONNX models to optimized TensorRT engines using trtexec

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
MODELS_DIR="${SCRIPT_DIR}/../models"

# Default parameters
BATCH_SIZE=1
WORKSPACE_SIZE=2048  # MB
FP16_MODE=false
VERBOSE=false
FORCE=false

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS] [ONNX_FILE]"
    echo
    echo "Convert ONNX models to TensorRT engines using trtexec"
    echo
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -a, --all               Convert all ONNX files in models directory"
    echo "  -b, --batch-size SIZE   Maximum batch size (default: 1)"
    echo "  -w, --workspace SIZE    Workspace size in MB (default: 2048)"
    echo "  -f, --fp16              Enable FP16 precision mode"
    echo "  -v, --verbose           Enable verbose output"
    echo "  --force                 Force conversion even if .trt file exists"
    echo "  --models-dir DIR        Directory containing ONNX models (default: ../models)"
    echo
    echo "Examples:"
    echo "  $0 model.onnx                    # Convert single model"
    echo "  $0 --all                         # Convert all models in default directory"
    echo "  $0 --all --fp16 --batch-size 4   # Convert all with FP16 and batch size 4"
    echo "  $0 --models-dir /path/to/models --all"
    echo
    exit 1
}

# Function to check if trtexec is available
check_trtexec() {
    if ! command -v trtexec >/dev/null 2>&1; then
        echo "Error: trtexec not found in PATH"
        echo "Please ensure TensorRT is properly installed and trtexec is in your PATH"
        echo
        echo "Typical installation paths:"
        echo "  /usr/local/TensorRT/bin/trtexec"
        echo "  /opt/tensorrt/bin/trtexec"
        echo
        echo "You can add it to PATH by running:"
        echo "  export PATH=/usr/local/TensorRT/bin:\$PATH"
        exit 1
    fi
    
    echo "Found trtexec: $(which trtexec)"
}

# Function to convert a single ONNX file to TensorRT
convert_onnx_to_trt() {
    local onnx_file="$1"
    local trt_file="${onnx_file%.onnx}.trt"
    
    # Check if input file exists
    if [[ ! -f "$onnx_file" ]]; then
        echo "Error: ONNX file not found: $onnx_file"
        return 1
    fi
    
    # Check if output file exists and is newer (unless force is enabled)
    if [[ -f "$trt_file" && "$FORCE" == false ]]; then
        if [[ "$trt_file" -nt "$onnx_file" ]]; then
            echo "Skipping $onnx_file (TensorRT engine is up to date)"
            return 0
        fi
    fi
    
    echo "Converting $onnx_file to $trt_file"
    
    # Build trtexec command
    local cmd="trtexec --onnx=\"$onnx_file\" --saveEngine=\"$trt_file\""
    cmd="$cmd --memPoolSize=workspace:${WORKSPACE_SIZE}M"
    
    if [[ "$FP16_MODE" == true ]]; then
        cmd="$cmd --fp16"
        echo "  Using FP16 precision mode"
    fi
    
    if [[ "$VERBOSE" == true ]]; then
        cmd="$cmd --verbose"
    fi
    
    # Add batch size optimization (only for dynamic models)
    # For static models, shapes are determined by the model itself
    if [[ "$BATCH_SIZE" -gt 1 ]]; then
        cmd="$cmd --minShapes=images:1x3x640x640"
        cmd="$cmd --optShapes=images:${BATCH_SIZE}x3x640x640"
        cmd="$cmd --maxShapes=images:${BATCH_SIZE}x3x640x640"
    fi
    
    echo "  Command: $cmd"
    echo "  Building TensorRT engine... This may take a while."
    
    # Execute conversion
    local start_time=$(date +%s)
    
    if eval "$cmd"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo "  ✓ Successfully converted in ${duration}s"
        
        # Print file sizes
        local onnx_size=$(stat -f%z "$onnx_file" 2>/dev/null || stat -c%s "$onnx_file")
        local trt_size=$(stat -f%z "$trt_file" 2>/dev/null || stat -c%s "$trt_file")
        
        echo "  File sizes: ONNX: $(numfmt --to=iec $onnx_size), TRT: $(numfmt --to=iec $trt_size)"
        return 0
    else
        echo "  ✗ Failed to convert $onnx_file"
        return 1
    fi
}

# Function to convert all ONNX files in directory
convert_all_models() {
    local models_dir="$1"
    
    if [[ ! -d "$models_dir" ]]; then
        echo "Error: Models directory not found: $models_dir"
        return 1
    fi
    
    # Find all ONNX files
    local onnx_files=()
    while IFS= read -r -d '' file; do
        onnx_files+=("$file")
    done < <(find "$models_dir" -name "*.onnx" -type f -print0)
    
    if [[ ${#onnx_files[@]} -eq 0 ]]; then
        echo "No ONNX files found in $models_dir"
        return 0
    fi
    
    echo "Found ${#onnx_files[@]} ONNX files in $models_dir"
    echo
    
    local success_count=0
    local total_count=${#onnx_files[@]}
    
    for onnx_file in "${onnx_files[@]}"; do
        echo "[$((success_count + 1))/$total_count] Processing $(basename "$onnx_file")"
        
        if convert_onnx_to_trt "$onnx_file"; then
            ((success_count++))
        fi
        echo
    done
    
    echo "Conversion summary: $success_count/$total_count models converted successfully"
    
    if [[ $success_count -eq $total_count ]]; then
        return 0
    else
        return 1
    fi
}

# Function to validate GPU and display info
check_gpu_info() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "GPU Information:"
        nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits | while read line; do
            echo "  $line"
        done
        echo
    else
        echo "Warning: nvidia-smi not found. Cannot display GPU information."
    fi
}

# Parse command line arguments
CONVERT_ALL=false
ONNX_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            ;;
        -a|--all)
            CONVERT_ALL=true
            shift
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -w|--workspace)
            WORKSPACE_SIZE="$2"
            shift 2
            ;;
        -f|--fp16)
            FP16_MODE=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --models-dir)
            MODELS_DIR="$2"
            shift 2
            ;;
        -*)
            echo "Unknown option: $1"
            usage
            ;;
        *)
            if [[ -n "$ONNX_FILE" ]]; then
                echo "Error: Multiple ONNX files specified"
                usage
            fi
            ONNX_FILE="$1"
            shift
            ;;
    esac
done

# Validate arguments
if [[ "$CONVERT_ALL" == true && -n "$ONNX_FILE" ]]; then
    echo "Error: Cannot specify both --all and a specific ONNX file"
    usage
fi

if [[ "$CONVERT_ALL" == false && -z "$ONNX_FILE" ]]; then
    echo "Error: Must specify either --all or an ONNX file"
    usage
fi

# Main execution
main() {
    echo "======================================"
    echo "YOLOs-CPP ONNX to TensorRT Converter"
    echo "======================================"
    echo
    
    # Configuration summary
    echo "Configuration:"
    echo "  Batch size: $BATCH_SIZE"
    echo "  Workspace size: ${WORKSPACE_SIZE}MB"
    echo "  FP16 mode: $FP16_MODE"
    echo "  Verbose: $VERBOSE"
    echo "  Force conversion: $FORCE"
    echo
    
    # Check prerequisites
    check_trtexec
    check_gpu_info
    
    # Perform conversion
    if [[ "$CONVERT_ALL" == true ]]; then
        echo "Converting all ONNX models in: $MODELS_DIR"
        convert_all_models "$MODELS_DIR"
    else
        echo "Converting single model: $ONNX_FILE"
        convert_onnx_to_trt "$ONNX_FILE"
    fi
    
    local exit_code=$?
    
    echo
    echo "======================================"
    if [[ $exit_code -eq 0 ]]; then
        echo "Conversion completed successfully!"
        echo
        echo "You can now use the .trt files with YOLOs-CPP:"
        echo "  ./build/image_inference"
        echo "  ./build/camera_inference"
        echo "  ./build/video_inference"
    else
        echo "Conversion completed with errors!"
        echo "Check the output above for details."
    fi
    echo "======================================"
    
    return $exit_code
}

# Run main function
main 