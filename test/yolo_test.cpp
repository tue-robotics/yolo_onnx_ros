#include "yolo_inference.h"
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

using namespace testing;

// Add a simpler test to check YOLO object creation
TEST(YoloInferenceTest, ObjectCreationTest) {
    // Test if we can create a YOLO_V8 object without crashing
    EXPECT_NO_THROW({
        YOLO_V8 yolo;
    });
}

// Add a test to check if the PreProcess function works correctly
TEST (YoloInferenceTest, PreProcessTest) {
    YOLO_V8 yolo;
    cv::Mat img = cv::Mat::ones(640, 640, CV_8UC3) * 255; // Create a white image
    std::vector<int> imgSize = { 640, 640 };
    cv::Mat processedImg;

    char* result = yolo.PreProcess(img, imgSize, processedImg);

    EXPECT_EQ(result, nullptr) << "PreProcess should return nullptr (RET_OK) on success";
    EXPECT_EQ(processedImg.size(), cv::Size(640, 640)) << "Processed image size should match input size";
}

// Add a test to check if the CreateSession function works correctly
TEST (YoloInferenceTest, CreateSessionTest) {
    std::unique_ptr<YOLO_V8> yolo = std::make_unique<YOLO_V8>();
    DL_INIT_PARAM params;
    params.modelPath = "yolo11m.onnx";
    params.modelType = YOLO_DETECT_V8;
    params.imgSize = { 640, 640 };
    params.rectConfidenceThreshold = 0.6;
    params.iouThreshold = 0.5;
    params.cudaEnable = false;

    const char* result = yolo->CreateSession(params);

    EXPECT_EQ(result, nullptr) << "CreateSession should return nullptr (RET_OK) on success";
}
// Add a test to check if the RunSession function works correctly
TEST (YoloInferenceTest, RunSessionTest) {
    std::unique_ptr<YOLO_V8> yolo = std::make_unique<YOLO_V8>();
    DL_INIT_PARAM params;
    params.modelPath = "yolo11m.onnx"; // Use a dummy model path
    params.modelType = YOLO_DETECT_V8;
    params.imgSize = { 640, 640 };
    params.rectConfidenceThreshold = 0.6;
    params.iouThreshold = 0.5;
    params.cudaEnable = false;

    const char* createResult = yolo->CreateSession(params);
    EXPECT_EQ(createResult, nullptr) << "CreateSession should return nullptr (RET_OK) on success";

    cv::Mat img = cv::Mat::ones(800, 800, CV_8UC3) * 255; // Create a white image
    std::vector<DL_RESULT> results;

    const char* runResult = yolo->RunSession(img, results);
    EXPECT_EQ(runResult, nullptr) << "RunSession should return nullptr (RET_OK) on success";
}
