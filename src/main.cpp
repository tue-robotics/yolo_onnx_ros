#include <iostream>
#include <iomanip>
#include "detection.h"
#include <filesystem>
#include <fstream>
#include <random>

int main()
{
    // An example of how to use the YOLO_V8 class for object detection
    std::unique_ptr<YOLO_V8> yoloDetector;
    std::vector<DL_RESULT> results;
    std::tie(yoloDetector, results) = Initialize();

    std::filesystem::path current_path = std::filesystem::current_path();
    std::filesystem::path imgs_path = current_path / "images";
    for (auto& i : std::filesystem::directory_iterator(imgs_path))
    {
        if (i.path().extension() == ".jpg" || i.path().extension() == ".png" || i.path().extension() == ".jpeg")
        {
            std::string img_path = i.path().string();
            cv::Mat img = cv::imread(img_path);
            DetectObjects(yoloDetector, img, results);
}
    }


    return 0;
}