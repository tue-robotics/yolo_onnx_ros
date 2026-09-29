#include "yolo_onnx_ros/detection.hpp"

#include <filesystem>
#include <iostream>
#include <string>

static YOLO::Backend ParseBackend(const std::string& value)
{
    if (value == "onnx")
        return YOLO::Backend::kOnnx;
    if (value == "tensorrt" || value == "trt")
        return YOLO::Backend::kTensorRT;

    throw std::runtime_error("Unknown backend: " + value);
}

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <model_path> <images_dir> [onnx|tensorrt]\n";
        return 1;
    }

    const std::filesystem::path model_name = argv[1];
    const std::filesystem::path imgs_path = argv[2];

    YOLO::Backend backend = YOLO::Backend::kOnnx;
    if (argc >= 4)
    {
        backend = ParseBackend(argv[3]);
    }

    YoloWrapper wrapper;
    DL_INIT_PARAM params;
    std::tie(wrapper, params) = Initialize(model_name, backend);

    for (const auto& entry : std::filesystem::directory_iterator(imgs_path))
    {
        if (entry.path().extension() == ".jpg" ||
            entry.path().extension() == ".png" ||
            entry.path().extension() == ".jpeg")
        {
            const std::string img_path = entry.path().string();
            cv::Mat img = cv::imread(img_path);
            std::vector<DL_RESULT> results = Detector(wrapper, img);

#ifdef LOGGING
            for (const auto& result : results)
            {
                std::cout << "Image path: " << img_path << "\n"
                          << "class id:   " << result.classId << "\n"
                          << "confidence: " << result.confidence << "\n";
            }
#endif // LOGGING
        }
    }

    return 0;
}
