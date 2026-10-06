#!/bin/bash
# ============================================================================
# YOLOs-TRT Unified Benchmark — Auto Build & Run
# ============================================================================
# Builds the benchmark tool against system TensorRT/CUDA and runs benchmarks.
#
# Usage:
#   ./auto_bench.sh                              # Quick benchmark
#   ./auto_bench.sh --models yolo11n,yolov8n     # Specific models
#   ./auto_bench.sh --eval-dataset /path/to/val2017  # With accuracy eval
# ============================================================================
set -euo pipefail

CURRENT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$CURRENT_DIR/.." && pwd)

# Defaults
MODELS="${MODELS:-yolo11n,yolov8n,yolo26n}"
EVAL_DATASET=""
BUILD_TYPE="Release"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ============================================================================
# Parse Arguments
# ============================================================================
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --models LIST        Comma-separated model names (default: yolo11n,yolov8n,yolo26n)"
    echo "  --eval-dataset DIR   Path to evaluation dataset for accuracy testing"
    echo "  --debug              Debug build"
    echo "  -h, --help           Show this help"
    echo ""
    echo "Prerequisites:"
    echo "  - NVIDIA GPU with CUDA and TensorRT installed"
    echo "  - TensorRT engine files (.trt) in models/ directory"
    echo "  - OpenCV, CMake, C++17 compiler"
    echo ""
    echo "Examples:"
    echo "  $0                                              # Build and run quick benchmark"
    echo "  $0 --models yolo11n                             # Test single model"
    echo "  $0 --models yolo11n,yolov8n --eval-dataset ../val2017  # With accuracy eval"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)
            MODELS="$2"
            shift 2
            ;;
        --eval-dataset)
            EVAL_DATASET="$2"
            shift 2
            ;;
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            ;;
    esac
done

# ============================================================================
# Check Prerequisites
# ============================================================================
check_prerequisites() {
    echo -e "${BLUE}Checking prerequisites...${NC}"

    if ! command -v cmake &>/dev/null; then
        echo -e "${RED}ERROR: cmake not found${NC}"
        exit 1
    fi

    if ! command -v nvcc &>/dev/null; then
        echo -e "${RED}ERROR: nvcc not found. Install CUDA Toolkit.${NC}"
        exit 1
    fi

    # Check for TensorRT headers
    local trt_found=0
    for path in /usr/include/x86_64-linux-gnu/NvInfer.h /usr/include/NvInfer.h /usr/include/aarch64-linux-gnu/NvInfer.h; do
        if [[ -f "$path" ]]; then
            trt_found=1
            break
        fi
    done
    if [[ $trt_found -eq 0 ]]; then
        echo -e "${RED}ERROR: TensorRT headers not found. Install: sudo apt install tensorrt${NC}"
        exit 1
    fi

    echo -e "${GREEN}Prerequisites OK${NC}"
}

# ============================================================================
# Build
# ============================================================================
build_project() {
    local build_dir="${CURRENT_DIR}/build"
    mkdir -p "$build_dir"
    cd "$build_dir"

    echo -e "${BLUE}Configuring CMake ($BUILD_TYPE)...${NC}"
    cmake .. \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native"

    echo -e "${BLUE}Building benchmark tool...${NC}"
    cmake --build . -- -j$(nproc 2>/dev/null || echo 4)

    echo -e "${GREEN}Build completed${NC}"
}

# ============================================================================
# Run Benchmarks
# ============================================================================
run_benchmarks() {
    local benchmark_exe="${CURRENT_DIR}/build/yolo_unified_benchmark"

    if [[ ! -f "$benchmark_exe" ]]; then
        echo -e "${RED}ERROR: Benchmark executable not found${NC}"
        exit 1
    fi

    cd "$PROJECT_ROOT"

    echo ""
    echo -e "${BLUE}===========================================${NC}"
    echo -e "${BLUE}  Running Comprehensive Benchmarks${NC}"
    echo -e "${BLUE}===========================================${NC}"
    echo ""

    # Run comprehensive benchmark
    "$benchmark_exe" comprehensive

    # If evaluation dataset is provided, run accuracy evaluation
    if [[ -n "$EVAL_DATASET" && -d "$EVAL_DATASET" ]]; then
        echo ""
        echo -e "${BLUE}===========================================${NC}"
        echo -e "${BLUE}  Running Accuracy Evaluation${NC}"
        echo -e "${BLUE}===========================================${NC}"
        echo ""

        GT_LABELS_DIR="${EVAL_DATASET}/../labels_val2017"
        if [[ ! -d "$GT_LABELS_DIR" ]]; then
            GT_LABELS_DIR="${EVAL_DATASET}/labels"
        fi

        if [[ -d "$GT_LABELS_DIR" ]]; then
            IFS=',' read -ra MODEL_ARRAY <<< "$MODELS"
            for model in "${MODEL_ARRAY[@]}"; do
                model_path="models/${model}.trt"
                if [[ ! -f "$model_path" ]]; then
                    echo -e "${YELLOW}Warning: Engine not found: $model_path, skipping...${NC}"
                    continue
                fi

                echo -e "${BLUE}Evaluating model: $model${NC}"
                "$benchmark_exe" evaluate yolo11 detection "$model_path" models/coco.names "$EVAL_DATASET" "$GT_LABELS_DIR"
                echo ""
            done
        else
            echo -e "${YELLOW}Ground truth labels not found. Skipping accuracy evaluation.${NC}"
        fi
    fi

    echo ""
    echo -e "${GREEN}===========================================${NC}"
    echo -e "${GREEN}  Benchmarking Complete!${NC}"
    echo -e "${GREEN}===========================================${NC}"
    echo "Results saved in: benchmarks/results/"
}

# ============================================================================
# Main
# ============================================================================
echo "============================================"
echo "  YOLOs-TRT Benchmark — Auto Build & Run"
echo "============================================"
echo "  Backend: TensorRT + CUDA"
echo "  Models:  $MODELS"
echo "============================================"
echo ""

check_prerequisites
build_project
run_benchmarks

echo ""
echo -e "${GREEN}All done!${NC}"
