#include "yolo_onnx_ros/detection.hpp"
#include "yolo_onnx_ros/yolo_inference.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#if defined(YOLO_ONNX_ROS_CUDA_ENABLED) && YOLO_ONNX_ROS_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

using namespace testing;

class YoloInferenceTest : public testing::Test
{
protected:
    void SetUp() override
    {
#if defined(YOLO_ONNX_ROS_CUDA_ENABLED) && YOLO_ONNX_ROS_CUDA_ENABLED
        // A CUDA build always runs the ONNX backend on the GPU, so skip when no GPU is available (e.g. CI)
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
        {
            GTEST_SKIP() << "Built with CUDA, but no CUDA device is available";
        }
#endif

        // Create test images with different characteristics
        testImage_640x640 = cv::Mat::ones(640, 640, CV_8UC3) * 255;
        testImage_800x600 = cv::Mat::ones(600, 800, CV_8UC3) * 128;

        // Create a more realistic test image with some patterns
        testImage_realistic = cv::Mat(640, 640, CV_8UC3);
        cv::randu(testImage_realistic, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

        // Setup common parameters
        YoloWrapper wrapper;
        std::tie(wrapper, params) = initialize("./yolo26m.onnx", YOLO::Backend::ONNX);
        yolo = std::move(wrapper.onnxDetector);

        NonSquareImgSize = {testImage_800x600.cols, testImage_800x600.rows};
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
    DlInitParam params;
    std::vector<int> NonSquareImgSize;
    std::unique_ptr<YoloV8> yolo;
};

TEST_F(YoloInferenceTest, ObjectCreation)
{
    EXPECT_NO_THROW({ YoloV8 local_yolo; });
}

TEST_F(YoloInferenceTest, PreProcessSquareImage)
{
    cv::Mat processed_img;
    char const* result = yolo->preProcess(testImage_640x640, params.imgSize, processed_img);

    EXPECT_EQ(result, nullptr) << "PreProcess should succeed";
    EXPECT_EQ(processed_img.size(), cv::Size(640, 640)) << "Output should be 640x640";
    EXPECT_FALSE(processed_img.empty()) << "Processed image should not be empty";
}

TEST_F(YoloInferenceTest, PreProcessRectangularImage)
{
    cv::Mat processed_img;
    char const* result = yolo->preProcess(testImage_800x600, NonSquareImgSize, processed_img);

    EXPECT_EQ(result, nullptr) << "PreProcess should succeed";
    EXPECT_EQ(processed_img.size(), cv::Size(800, 600)) << "Output should be letterboxed to 800x600";
    EXPECT_FALSE(processed_img.empty()) << "Processed image should not be empty";
}

TEST_F(YoloInferenceTest, CreateSessionWithValidModel)
{
    const char* result = yolo->createSession(params);
    EXPECT_EQ(result, nullptr) << "CreateSession should succeed with valid parameters";
}

TEST_F(YoloInferenceTest, CreateSessionWithInvalidModel)
{
    params.modelPath = "nonexistent_model.onnx";
    EXPECT_THROW(yolo->createSession(params), std::runtime_error)
        << "CreateSession should throw an exception with invalid model path";
}

TEST_F(YoloInferenceTest, FullInferencePipeline)
{
    // First create session
    const char* create_result = yolo->createSession(params);
    ASSERT_EQ(create_result, nullptr) << "Session creation must succeed for inference test";

    std::vector<DlResult> results;
    const char* run_result = yolo->runSession(testImage_realistic, results);

    EXPECT_EQ(run_result, nullptr) << "RunSession should succeed";
}

// Run all tests
int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
