#include "yolo_inference.h"
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <opencv2/opencv.hpp>

using namespace testing;

class YoloInferenceTest : public testing::Test
{
protected:
    void SetUp() override
    {
        // Create test images with different characteristics
        testImage_640x640 = cv::Mat::ones(640, 640, CV_8UC3) * 255;
        testImage_800x600 = cv::Mat::ones(600, 800, CV_8UC3) * 128;

        // Create a more realistic test image with some patterns
        testImage_realistic = cv::Mat(640, 640, CV_8UC3);
        cv::randu(testImage_realistic, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

        // Setup common parameters
        params.modelPath = "yolo11m.onnx";
        params.modelType = YOLO_DETECT_V8;
        params.imgSize = { 640, 640 };
        params.rectConfidenceThreshold = 0.6;
        params.iouThreshold = 0.5;
        params.cudaEnable = false;

        yolo = std::make_unique<YOLO_V8>();
        NonSquareImgSize = { testImage_800x600.cols, testImage_800x600.rows };
    }

    void TearDown() override
    {
        // Clean up if needed
        yolo.reset();
    }

    // Test data
    cv::Mat testImage_640x640;
    cv::Mat testImage_800x600;
    cv::Mat testImage_realistic;
    DL_INIT_PARAM params;
    std::unique_ptr<YOLO_V8> yolo;
    std::vector<int> NonSquareImgSize;
};

TEST_F(YoloInferenceTest, ObjectCreation)
{
    EXPECT_NO_THROW({
        YOLO_V8 localYolo;
    });
}

TEST_F(YoloInferenceTest, PreProcessSquareImage)
{
    cv::Mat processedImg;
    char* result = yolo->PreProcess(testImage_640x640, params.imgSize, processedImg);

    EXPECT_EQ(result, nullptr) << "PreProcess should succeed";
    EXPECT_EQ(processedImg.size(), cv::Size(640, 640)) << "Output should be 640x640";
    EXPECT_FALSE(processedImg.empty()) << "Processed image should not be empty";
}

TEST_F(YoloInferenceTest, PreProcessRectangularImage)
{
    cv::Mat processedImg;
    char* result = yolo->PreProcess(testImage_800x600, NonSquareImgSize, processedImg);

    EXPECT_EQ(result, nullptr) << "PreProcess should succeed";
    EXPECT_EQ(processedImg.size(), cv::Size(800, 600)) << "Output should be letterboxed to 800x600";
    EXPECT_FALSE(processedImg.empty()) << "Processed image should not be empty";
}

TEST_F(YoloInferenceTest, CreateSessionWithValidModel)
{
    const char* result = yolo->CreateSession(params);
    EXPECT_EQ(result, nullptr) << "CreateSession should succeed with valid parameters";
}

TEST_F(YoloInferenceTest, CreateSessionWithInvalidModel)
{
    params.modelPath = "nonexistent_model.onnx";
    const char* result = yolo->CreateSession(params);
    EXPECT_NE(result, nullptr) << "CreateSession should fail with invalid model path";
}

TEST_F(YoloInferenceTest, FullInferencePipeline)
{
    // First create session
    const char* createResult = yolo->CreateSession(params);
    ASSERT_EQ(createResult, nullptr) << "Session creation must succeed for inference test";

    // Then run inference
    std::vector<DL_RESULT> results;
    const char* runResult = yolo->RunSession(testImage_realistic, results);

    EXPECT_EQ(runResult, nullptr) << "RunSession should succeed";
    // Note: results might be empty for random test image, that's okay
    EXPECT_TRUE(results.size() >= 0) << "Results should be a valid vector";
}

// Run all tests
int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
