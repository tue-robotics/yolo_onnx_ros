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
// Import Yolo Inference header
TEST (YoloInferenceTest, PreProcessTest) {
    YOLO_V8 yolo;
    cv::Mat img = cv::Mat::ones(640, 640, CV_8UC3) * 255; // Create a white image
    std::vector<int> imgSize = { 640, 640 };
    cv::Mat processedImg;

    char* result = yolo.PreProcess(img, imgSize, processedImg);

    EXPECT_EQ(result, nullptr) << "PreProcess should return nullptr (RET_OK) on success";
    EXPECT_EQ(processedImg.size(), cv::Size(640, 640));
}
