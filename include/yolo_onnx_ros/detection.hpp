#pragma once

#include "yolo_onnx_ros/yolo_inference.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

// Uncomment the following line to enable additional logging output for debugging purposes.
// #define LOGGING

#if defined(YOLO_ONNX_ROS_CUDA_ENABLED) && YOLO_ONNX_ROS_CUDA_ENABLED
#include "yolos/tasks/detection.hpp"
#endif

namespace YOLO
{
    enum class Backend
    {
        kOnnx,
        kTensorRT,
    };
}

class YoloWrapper
{
public:
    YOLO::Backend backend;
    /// Class names — may be empty if the model ships without a labels file (e.g. YOLO26).
    std::vector<std::string> classes;

    std::unique_ptr<YOLO_V8> onnxDetector;  ///< Active when backend == kOnnx

#if defined(YOLO_ONNX_ROS_CUDA_ENABLED) && YOLO_ONNX_ROS_CUDA_ENABLED
    std::unique_ptr<yolos::det::YOLODetector> trtDetector;  ///< Active when backend == kTensorRT
#endif

    YoloWrapper() : backend(YOLO::Backend::kOnnx) {}
};

/// @brief Initialize a YOLO detector for the requested backend.
/// Class names are loaded from coco.yaml (ONNX) or coco.names (TRT) when present
/// in the model directory; if absent the wrapper's classes vector stays empty.
std::tuple<YoloWrapper, DL_INIT_PARAM> Initialize(const std::filesystem::path& model_filename,
                                                   YOLO::Backend backend = YOLO::Backend::kOnnx);

/// @brief Run YOLO detection. Returns DL_RESULT regardless of active backend.
std::vector<DL_RESULT> Detector(YoloWrapper& wrapper, const cv::Mat& img);

int ReadYaml(const std::filesystem::path& filename, std::unique_ptr<YOLO_V8>& p);
