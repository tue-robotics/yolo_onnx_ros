#pragma once

#include "yolo_onnx_ros/config.hpp"
#include "yolo_onnx_ros/yolo_inference.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

// Uncomment the following line to enable additional logging output for debugging purposes.
// #define LOGGING

#if defined(YOLO_ONNX_ROS_TENSORRT_ENABLED) && YOLO_ONNX_ROS_TENSORRT_ENABLED
#include <yolos/tasks/detection.hpp>
#endif

namespace YOLO
{
enum class Backend : std::uint8_t
{
    ONNX,
    TENSORRT,
};
}

class YoloWrapper
{
public:
    YOLO::Backend backend{YOLO::Backend::ONNX};
    /// Class names — may be empty if the model ships without a labels file (e.g. YOLO26).
    std::vector<std::string> classes;

    std::unique_ptr<YoloV8> onnxDetector; ///< Active when backend == ONNX

#if defined(YOLO_ONNX_ROS_TENSORRT_ENABLED) && YOLO_ONNX_ROS_TENSORRT_ENABLED
    std::unique_ptr<yolos::det::YOLODetector> trtDetector; ///< Active when backend == TENSORRT
#endif
};

/// @brief Initialize a YOLO detector for the requested backend.
/// Class names are loaded from coco.yaml (ONNX) or coco.names (TRT) when present
/// in the model directory; if absent the wrapper's classes vector stays empty.
std::tuple<YoloWrapper, DlInitParam> initialize(const std::filesystem::path& model_filename,
                                                YOLO::Backend backend = YOLO::Backend::ONNX);

/// @brief Run YOLO detection. Returns DlResult regardless of active backend.
std::vector<DlResult> detector(YoloWrapper& wrapper, const cv::Mat& img);

int readYaml(const std::filesystem::path& filename, std::unique_ptr<YoloV8>& p);
