#include "yolo_onnx_ros/yolo_inference.hpp"

#include <filesystem>

// #define LOGGING
std::tuple<std::unique_ptr<YOLO_V8>, DL_INIT_PARAM> Initialize(const std::filesystem::path& model_filename);

std::vector<DL_RESULT> Detector(std::unique_ptr<YOLO_V8>& p, const cv::Mat& img);

int ReadCocoYaml(const std::filesystem::path& filename, std::unique_ptr<YOLO_V8>& p);
