#include "yolo_onnx_ros/detection.hpp"

#if defined(YOLO_ONNX_ROS_CUDA_ENABLED) && YOLO_ONNX_ROS_CUDA_ENABLED
#include "yolos/tasks/detection.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

enum class Backend { kOnnx, kTensorRT };

Backend ParseBackend(const std::string& value)
{
    if (value == "onnx")     return Backend::kOnnx;
    if (value == "tensorrt") return Backend::kTensorRT;
    throw std::invalid_argument("Unsupported backend '" + value + "'. Use 'onnx' or 'tensorrt'.");
}

void PrintUsage(const char* executable)
{
    std::cerr << "Usage: " << executable
              << " <model> <images_dir> [--backend=onnx|tensorrt] [--labels=<file>]\n"
              << "  --backend=onnx       ONNX Runtime + CUDA  (default)\n"
              << "  --backend=tensorrt   TensorRT engine (.trt)\n"
              << "  --labels=<file>      Class names file (required for tensorrt)\n";
}

void PrintSummary(const std::vector<double>& frame_times)
{
    if (frame_times.empty()) return;
    double total  = std::accumulate(frame_times.begin(), frame_times.end(), 0.0);
    double avg    = total / frame_times.size();
    double min    = *std::min_element(frame_times.begin(), frame_times.end());
    double max    = *std::max_element(frame_times.begin(), frame_times.end());
    double sq_sum = std::inner_product(frame_times.begin(), frame_times.end(), frame_times.begin(), 0.0);
    double stddev = std::sqrt(sq_sum / frame_times.size() - avg * avg);

    std::cout << "\n--- Summary (excl. first frame) ---\n"
              << "Frames : " << frame_times.size() << "\n"
              << "Avg    : " << avg    << " ms  (" << 1000.0 / avg << " FPS)\n"
              << "Min    : " << min    << " ms\n"
              << "Max    : " << max    << " ms\n"
              << "StdDev : " << stddev << " ms\n"
              << "-----------------------------------" << std::endl;
}

int RunOnnx(const std::filesystem::path& model_path, const std::filesystem::path& imgs_path)
{
    std::unique_ptr<YOLO_V8> yoloDetector;
    DL_INIT_PARAM params;
    std::tie(yoloDetector, params) = Initialize(model_path);

    std::vector<double> frame_times;
    int frame_count = 0;
    for (const auto& i : std::filesystem::directory_iterator(imgs_path))
    {
        const auto& ext = i.path().extension();
        if (ext != ".jpg" && ext != ".png" && ext != ".jpeg") continue;

        cv::Mat img = cv::imread(i.path().string());
        std::vector<DL_RESULT> results;
        auto t0 = std::chrono::high_resolution_clock::now();
        results = Detector(yoloDetector, img);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        ++frame_count;
        if (frame_count > 1) frame_times.push_back(ms);
        std::cout << "[" << frame_count << "] " << i.path().filename().string()
                  << " : " << ms << " ms"
                  << "  detections: " << results.size()
                  << (frame_count == 1 ? "  (warm-up, excluded from stats)" : "")
                  << std::endl;
    }
    PrintSummary(frame_times);
    return 0;
}

#if defined(YOLO_ONNX_ROS_CUDA_ENABLED) && YOLO_ONNX_ROS_CUDA_ENABLED
int RunTensorRT(const std::filesystem::path& engine_path, const std::filesystem::path& imgs_path,
                const std::string& labels_path)
{
    yolos::det::YOLODetector detector(engine_path.string(), labels_path);

    std::vector<double> frame_times;
    int frame_count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(imgs_path))
    {
        const auto& ext = entry.path().extension();
        if (ext != ".jpg" && ext != ".png" && ext != ".jpeg") continue;

        cv::Mat img = cv::imread(entry.path().string());
        if (img.empty()) { std::cerr << "Failed to read: " << entry.path() << std::endl; continue; }

        auto t0 = std::chrono::high_resolution_clock::now();
        auto detections = detector.detect(img);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        ++frame_count;
        if (frame_count > 1) frame_times.push_back(ms);
        std::cout << "[" << frame_count << "] " << entry.path().filename().string()
                  << " : " << ms << " ms"
                  << "  detections: " << detections.size()
                  << (frame_count == 1 ? "  (warm-up, excluded from stats)" : "")
                  << std::endl;
    }
    PrintSummary(frame_times);
    return 0;
}
#endif

} // namespace

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        PrintUsage(argv[0]);
        return 1;
    }

    const std::filesystem::path model_path = argv[1];
    const std::filesystem::path imgs_path  = argv[2];

    Backend backend = Backend::kOnnx;
    std::string labels_path;

    try
    {
        for (int i = 3; i < argc; ++i)
        {
            const std::string arg = argv[i];
            if (arg.rfind("--backend=", 0) == 0)
                backend = ParseBackend(arg.substr(10));
            else if (arg.rfind("--labels=", 0) == 0)
                labels_path = arg.substr(9);
            else if (arg == "--help") { PrintUsage(argv[0]); return 0; }
            else throw std::invalid_argument("Unknown argument '" + arg + "'.");
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        PrintUsage(argv[0]);
        return 1;
    }

#if !defined(YOLO_ONNX_ROS_CUDA_ENABLED) || !YOLO_ONNX_ROS_CUDA_ENABLED
    if (backend == Backend::kTensorRT)
    {
        std::cerr << "This binary was built without TensorRT support. Rebuild with -DCUDA_ENABLED=ON, or use --backend=onnx." << std::endl;
        return 2;
    }
#endif

    if (backend == Backend::kOnnx)
        return RunOnnx(model_path, imgs_path);

#if defined(YOLO_ONNX_ROS_CUDA_ENABLED) && YOLO_ONNX_ROS_CUDA_ENABLED
    return RunTensorRT(model_path, imgs_path, labels_path);
#endif
}

