#include "yolo_onnx_ros/detection.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <vector>

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
    std::vector<double> frame_times;
    int frame_count = 0;
    for (const auto& i : std::filesystem::directory_iterator(imgs_path))
    {
        if (i.path().extension() == ".jpg" || i.path().extension() == ".png" || i.path().extension() == ".jpeg")
        {
            std::string img_path = i.path().string();
            cv::Mat img = cv::imread(img_path);
            std::vector<DL_RESULT> results;
            auto t0 = std::chrono::high_resolution_clock::now();
            results = Detector(yoloDetector, img);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            ++frame_count;
            // skip first frame from stats — CUDA kernel caching outlier
            if (frame_count > 1)
                frame_times.push_back(ms);
            std::cout << "[" << frame_count << "] " << i.path().filename().string()
                      << " : " << ms << " ms"
                      << "  detections: " << results.size()
                      << (frame_count == 1 ? "  (warm-up, excluded from stats)" : "")
                      << std::endl;
        }
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
