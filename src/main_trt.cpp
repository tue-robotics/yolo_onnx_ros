#include "yolos/tasks/detection.hpp"

#include <filesystem>
#include <iostream>

int main(int argc, char* argv[])
{
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <engine.trt> <labels.names> <images_dir>" << std::endl;
        std::cerr << "  engine.trt   - TensorRT engine file (.trt / .engine)" << std::endl;
        std::cerr << "  labels.names - Class names file (one per line)" << std::endl;
        std::cerr << "  images_dir   - Directory containing .jpg / .png images" << std::endl;
        return 1;
    }

    const std::string engine_path = argv[1];
    const std::string labels_path = argv[2];
    const std::filesystem::path imgs_path = argv[3];

    if (!std::filesystem::exists(engine_path))
    {
        std::cerr << "Engine file not found: " << engine_path << std::endl;
        return 1;
    }
    if (!std::filesystem::exists(imgs_path))
    {
        std::cerr << "Images directory not found: " << imgs_path << std::endl;
        return 1;
    }

    yolos::det::YOLODetector detector(engine_path, labels_path);

    for (const auto& entry : std::filesystem::directory_iterator(imgs_path))
    {
        const auto& ext = entry.path().extension();
        if (ext != ".jpg" && ext != ".png" && ext != ".jpeg")
            continue;

        cv::Mat img = cv::imread(entry.path().string());
        if (img.empty())
        {
            std::cerr << "Failed to read: " << entry.path() << std::endl;
            continue;
        }

        auto detections = detector.detect(img);

        for (const auto& det : detections)
        {
            std::cout << "Image: "      << entry.path().filename().string()
                      << "  class_id: " << det.classId
                      << "  conf: "     << det.conf
                      << "  box: ["     << det.box.x << "," << det.box.y
                      << " "           << det.box.width << "x" << det.box.height
                      << "]" << std::endl;
        }

        detector.drawDetections(img, detections);
        cv::imshow("TRT Detection", img);
        if (cv::waitKey(0) == 27)  // ESC to quit
            break;
    }

    cv::destroyAllWindows();
    return 0;
}
