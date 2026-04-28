#include "yolos/tasks/detection.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <vector>

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

    std::vector<double> frame_times;
    int frame_count = 0;
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

        auto t0 = std::chrono::high_resolution_clock::now();
        auto detections = detector.detect(img);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        ++frame_count;
        // skip first frame from stats — TRT kernel caching outlier
        if (frame_count > 1)
            frame_times.push_back(ms);

        std::cout << "[" << frame_count << "] " << entry.path().filename().string()
                  << " : " << ms << " ms"
                  << "  detections: " << detections.size()
                  << (frame_count == 1 ? "  (warm-up, excluded from stats)" : "")
                  << std::endl;
    }

    if (!frame_times.empty())
    {
        double total = std::accumulate(frame_times.begin(), frame_times.end(), 0.0);
        double avg = total / frame_times.size();
        double min = *std::min_element(frame_times.begin(), frame_times.end());
        double max = *std::max_element(frame_times.begin(), frame_times.end());
        double sq_sum = std::inner_product(frame_times.begin(), frame_times.end(), frame_times.begin(), 0.0);
        double stddev = std::sqrt(sq_sum / frame_times.size() - avg * avg);

        std::cout << "\n--- Summary (excl. first frame) ---\n"
                  << "Frames : " << frame_times.size() << "\n"
                  << "Avg    : " << avg    << " ms  (" << 1000.0 / avg    << " FPS)\n"
                  << "Min    : " << min    << " ms\n"
                  << "Max    : " << max    << " ms\n"
                  << "StdDev : " << stddev << " ms\n"
                  << "-----------------------------------" << std::endl;
    }

    return 0;
}
