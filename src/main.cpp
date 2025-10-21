#include <iostream>
#include <iomanip>
#include "yolo_onnx_ros/detection.hpp"
#include <filesystem>
#include <fstream>
#include <random>

int main()
{
    std::unique_ptr<YOLO_V8> yoloDetector;
    DL_INIT_PARAM params;

    std::tie(yoloDetector, params) = Initialize();

    std::filesystem::path current_path = std::filesystem::current_path();
    std::filesystem::path imgs_path = current_path / "images";
    for (auto& i : std::filesystem::directory_iterator(imgs_path))
    {
        if (i.path().extension() == ".jpg" || i.path().extension() == ".png" || i.path().extension() == ".jpeg")
        {
            std::string img_path = i.path().string();
            cv::Mat img = cv::imread(img_path);
            std::vector<DL_RESULT> results;
            results = Detector(yoloDetector, img);
            for (const auto& result : results)
            {
                std::cout << "Image path: " << img_path << "\n"
                          << "class id:   " << result.classId << "\n"
                          << "confidence: " << result.confidence << "\n";
            }
        }
    }


    return 0;
}
