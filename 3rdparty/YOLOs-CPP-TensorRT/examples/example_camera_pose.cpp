/**
 * @file example_camera_pose.cpp
 * @brief Real-time pose estimation from camera
 * @details Live camera feed with human pose detection
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include "yolos/tasks/pose.hpp"
#include "utils.hpp"

using namespace yolos::pose;

int main(int argc, char* argv[]) {
    std::string modelPath = "../../models/yolo11n-pose.trt";
    std::string labelsPath = "../../models/coco.names";
    int cameraId = 0;
    
    if (argc > 1) modelPath = argv[1];
    if (argc > 2) labelsPath = argv[2];
    if (argc > 3) cameraId = std::stoi(argv[3]);
    
    std::cout << "\n╔════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  YOLOs-CPP Real-Time Camera Pose Estimation" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════╝" << std::endl;
    std::cout << "\nPress 'q' to quit, 's' to save snapshot\n" << std::endl;
    
    std::cout << "🔄 Loading pose estimation model..." << std::endl;
    
    try {
        YOLOPoseDetector detector(modelPath, labelsPath);
        std::cout << "✅ Model loaded!" << std::endl;
        
        cv::VideoCapture cap(cameraId);
        if (!cap.isOpened()) {
            std::cerr << "❌ Could not open camera" << std::endl;
            return -1;
        }
        
        std::cout << "🎬 Starting real-time pose estimation...\n" << std::endl;
        
        cv::Mat frame;
        int frameCount = 0;
        auto startTime = std::chrono::high_resolution_clock::now();
        
        while (true) {
            cap >> frame;
            if (frame.empty()) break;
            
            frameCount++;
            std::vector<PoseResult> detections = detector.detect(frame, 0.4f, 0.5f);
            detector.drawPoses(frame, detections);
            
            auto currentTime = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(currentTime - startTime);
            double fps = elapsed.count() > 0 ? frameCount / static_cast<double>(elapsed.count()) : 0;
            
            std::string info = "FPS: " + std::to_string(static_cast<int>(fps)) + 
                              " | Persons: " + std::to_string(detections.size());
            cv::putText(frame, info, cv::Point(10, 30), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            
            cv::imshow("YOLO Camera Pose (Press 'q' to quit, 's' to save)", frame);
            
            char key = cv::waitKey(1);
            if (key == 'q') break;
            if (key == 's') {
                std::string path = utils::saveImage(frame, "camera_pose.jpg", "../../outputs/pose/");
                std::cout << "📸 Saved: " << path << std::endl;
            }
        }
        
        cap.release();
        cv::destroyAllWindows();
        std::cout << "\n✅ Pose estimation stopped!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
