#include "yolo_onnx_ros/detection.hpp"

#include <filesystem>
#include <iostream>

int main(int argc, char *argv[])
{
    std::unique_ptr<YOLO_V8> yoloDetector;
    DL_INIT_PARAM params;

    if (argc < 2)
    {
        std::cerr << "Not enough args provided" << std::endl;
        return 1;
    }

    const std::filesystem::path model_name = argv[1];

    std::tie(yoloDetector, params) = Initialize(model_name);

    std::filesystem::path imgs_path = argv[2];
    for (const auto& i : std::filesystem::directory_iterator(imgs_path))
    {
        if (i.path().extension() == ".jpg" || i.path().extension() == ".png" || i.path().extension() == ".jpeg")
        {
            std::string img_path = i.path().string();
            cv::Mat img = cv::imread(img_path);
            std::vector<DL_RESULT> results;
            results = Detector(yoloDetector, img);
            #ifdef LOGGING
            for (const auto& result : results)
            {
                std::cout << "Image path: " << img_path << "\n"
                          << "class id:   " << result.classId << "\n"
                          << "confidence: " << result.confidence << "\n";
            }
            #endif
        }
    }


    return 0;
}
