#include "yolo_onnx_ros/detection.hpp"
#include "yolo_onnx_ros/yolo_inference.hpp"

#include <filesystem>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace
{
YOLO::Backend parseBackend(const std::string& value)
{
    if (value == "onnx")
        return YOLO::Backend::ONNX;
    if (value == "tensorrt" || value == "trt")
        return YOLO::Backend::TENSORRT;

    throw std::runtime_error("Unknown backend: " + value);
}
} // namespace

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <model_path> <images_dir> [onnx|tensorrt]\n";
        return 1;
    }

    const std::filesystem::path model_name = argv[1];
    const std::filesystem::path imgs_path = argv[2];

    YOLO::Backend backend = YOLO::Backend::ONNX;
    if (argc >= 4)
    {
        backend = parseBackend(argv[3]);
    }

    YoloWrapper wrapper;
    DlInitParam params;
    std::tie(wrapper, params) = initialize(model_name, backend);

    for (const auto& entry : std::filesystem::directory_iterator(imgs_path))
    {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png" ||
            entry.path().extension() == ".jpeg")
        {
            const std::string img_path = entry.path().string();
            cv::Mat const img = cv::imread(img_path);
            std::vector<DlResult> const results = detector(wrapper, img);

#ifdef LOGGING
            for (const auto& result : results)
            {
                std::cout << "Image path: " << img_path << "\n"
                          << "class id:   " << result.classId << "\n"
                          << "confidence: " << result.confidence << "\n";
            }
            cv::imshow("detections", img);
            cv::waitKey(0);
#endif // LOGGING
        }
    }

    return 0;
}
