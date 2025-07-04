#pragma once

#define    RET_OK nullptr

#ifdef _WIN32
#include <Windows.h>
#include <direct.h>
#include <io.h>
#endif

#include <string>
#include <vector>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include "onnxruntime_cxx_api.h"

#ifdef USE_CUDA
#include <cuda_fp16.h>
#endif


enum MODEL_TYPE
{
    //FLOAT32 MODEL
    YOLO_DETECT_V8 = 1,
    YOLO_POSE = 2,
    YOLO_CLS = 3,

    //FLOAT16 MODEL
    YOLO_DETECT_V8_HALF = 4,
    YOLO_POSE_V8_HALF = 5,
    YOLO_CLS_HALF = 6
};


typedef struct _DL_INIT_PARAM
{
    std::string modelPath;
    MODEL_TYPE modelType = YOLO_DETECT_V8;
    std::vector<int> imgSize = { 640, 640 };
    float rectConfidenceThreshold = 0.6;
    float iouThreshold = 0.5;
    int	keyPointsNum = 2;//Note:kpt number for pose
    bool cudaEnable = false;
    int logSeverityLevel = 3;
    int intraOpNumThreads = 1;
} DL_INIT_PARAM;


typedef struct _DL_RESULT
{
    int classId;
    float confidence;
    cv::Rect box;
    std::vector<cv::Point2f> keyPoints;
} DL_RESULT;


class YOLO_V8
{
public:
    YOLO_V8();

    ~YOLO_V8();

public:
    const char* CreateSession(DL_INIT_PARAM& iParams);

    const char* RunSession(const cv::Mat& iImg, std::vector<DL_RESULT>& oResult);

    std::vector<std::string> classes{};

private:
    char* WarmUpSession();

    // Note: The logic is on the .cpp file since its a private method.
    template<typename N>
    char* TensorProcess(clock_t& starttime_1, const cv::Mat& iImg, N& blob, std::vector<int64_t>& inputNodeDims,
        std::vector<DL_RESULT>& oResult);

    char* PreProcess(const cv::Mat& iImg, std::vector<int> iImgSize, cv::Mat& oImg);

    Ort::Env env;
    std::unique_ptr<Ort::Session> session;
    bool cudaEnable;
    Ort::RunOptions options;
    std::vector<const char*> inputNodeNames;
    std::vector<const char*> outputNodeNames;

    MODEL_TYPE modelType;
    std::vector<int> imgSize;
    float rectConfidenceThreshold;
    float iouThreshold;
    float resizeScales;//letterbox scale
};