#!/bin/bash
# ============================================================================
# YOLOs-TRT Test Runner
# ============================================================================

# Default values
TEST_TASK="${1:-0}" # 0: detection, 1: classification, 2: segmentation, 3: pose, 4: obb, 5: all

usage() {
    echo "Usage: $0 [TEST_TASK]"
    echo
    echo "This script runs the specified test task for YOLOs-TRT."
    echo "Requires NVIDIA GPU with CUDA and TensorRT."
    echo
    echo "Arguments:"
    echo "  TEST_TASK   0=detection, 1=classification, 2=segmentation,"
    echo "              3=pose, 4=obb, 5=all (default: 0)"
    echo
    echo "Examples:"
    echo "  $0 0    # Run detection tests"
    echo "  $0 5    # Run all tests"
    exit 1
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
fi

cd tests

case "$TEST_TASK" in
    0) echo "Running detection tests..."     && ./test_detection.sh ;;
    1) echo "Running classification tests..." && ./test_classification.sh ;;
    2) echo "Running segmentation tests..."   && ./test_segmentation.sh ;;
    3) echo "Running pose tests..."           && ./test_pose.sh ;;
    4) echo "Running obb tests..."            && ./test_obb.sh ;;
    5) echo "Running all tests..."            && ./test_all.sh ;;
    *) echo "Invalid TEST_TASK: $TEST_TASK"   && usage ;;
esac
