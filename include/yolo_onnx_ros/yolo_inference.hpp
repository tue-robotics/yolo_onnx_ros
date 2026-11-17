#pragma once

#define RET_OK nullptr

#include "onnxruntime_cxx_api.h"
#include <cstdint>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include <yolo_onnx_ros/config.hpp>

#if defined(YOLO_ONNX_ROS_CUDA_ENABLED) && YOLO_ONNX_ROS_CUDA_ENABLED
#include <cuda_fp16.h>
#endif

enum class ModelType : std::uint8_t
{
    // FLOAT32 MODEL
    YOLO_DETECT_V8 = 1,
    YOLO_POSE = 2,
    YOLO_CLS = 3,

    // FLOAT16 MODEL
    YOLO_DETECT_V8_HALF = 4,
    YOLO_POSE_V8_HALF = 5,
    YOLO_CLS_HALF = 6
};

struct DlInitParam
{
    std::string modelPath;
    ModelType modelType = ModelType::YOLO_DETECT_V8;
    std::vector<int> imgSize = {640, 640};
    float rectConfidenceThreshold = 0.6;
    float iouThreshold = 0.5;
    int keyPointsNum = 2; // Note:kpt number for pose
    bool cudaEnable = false;
    int logSeverityLevel = 3;
    int intraOpNumThreads = 1;
};

struct DlResult
{
    int classId{};
    float confidence{};
    cv::Rect box;
    std::vector<cv::Point2f> keyPoints;
};

class YoloV8
{
public:
    YoloV8() = default;

    ~YoloV8() = default;

    const char* createSession(DlInitParam& iParams);

    const char* runSession(const cv::Mat& iImg, std::vector<DlResult>& oResult);
    // imgSize is [width, height]
    const char* preProcess(const cv::Mat& iImg, const std::vector<int>& iImgSize, cv::Mat& oImg);

    std::vector<std::string> classes;

private:
    const char* warmUpSession();

    // Note: The logic is on the .cpp file since its a private method.
    template <typename N>
    const char*
    tensorProcess(clock_t& starttime_1, N& blob, std::vector<int64_t>& inputNodeDims, std::vector<DlResult>& oResult);

    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    bool cudaEnable_{};
    Ort::RunOptions options;
    // Owns the node name strings; the *NodeNames_ vectors point into these for Ort::Session::Run
    std::vector<std::string> inputNodeNameStorage_;
    std::vector<std::string> outputNodeNameStorage_;
    std::vector<const char*> inputNodeNames_;
    std::vector<const char*> outputNodeNames_;

    ModelType modelType_{ModelType::YOLO_DETECT_V8};
    std::vector<int> imgSize_;
    float rectConfidenceThreshold_{};
    float iouThreshold_{};
    float resizeScales_{}; // letterbox scale
};
