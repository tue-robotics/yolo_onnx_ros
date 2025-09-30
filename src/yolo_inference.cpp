#include "yolo_onnx_ros/yolo_inference.hpp"
#include <regex>

#define benchmark
#define min(a, b) (((a) < (b)) ? (a) : (b))

#ifdef ONNX_YOLO_ROS_CUDA_ENABLED
namespace Ort
{
    template<>
    struct TypeToTensorType<half> { static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16; };
}
#endif

/**
 * @brief Flattened image to blob (and normalized) for deep learning inference. Also reorganize from HWC to CHW.
 *
 * @tparam T
 * @param iImg
 * @param iBlob
 * @return char*
 */
template<typename T>
char* BlobFromImage(const cv::Mat& iImg, T& iBlob) {
    int channels = iImg.channels();
    int imgHeight = iImg.rows;
    int imgWidth = iImg.cols;

    for (int c = 0; c < channels; c++)
    {
        for (int h = 0; h < imgHeight; h++)
        {
            for (int w = 0; w < imgWidth; w++)
            {
                iBlob[c * imgWidth * imgHeight + h * imgWidth + w] = typename std::remove_pointer<T>::type(
                    (iImg.at<cv::Vec3b>(h, w)[c]) / 255.0f);
            }
        }
    }
    return RET_OK;
}


char* YOLO_V8::PreProcess(const cv::Mat& iImg, const std::vector<int>& iImgSize, cv::Mat& oImg)
{
    // Basic validation
    if (iImg.empty()) return (char*)"[YOLO_V8]: Empty input image.";
    if (iImgSize.size() != 2 || iImgSize[0] <= 0 || iImgSize[1] <= 0)
        return (char*)"[YOLO_V8]: Invalid target image size.";

    if (iImg.channels() == 3)
    {
        oImg = iImg.clone();
        cv::cvtColor(oImg, oImg, cv::COLOR_BGR2RGB);
    }
    else
    {
        cv::cvtColor(iImg, oImg, cv::COLOR_GRAY2RGB);
    }

    switch (modelType_)
    {
    case YOLO_DETECT_V8:
    case YOLO_POSE:
    case YOLO_DETECT_V8_HALF:
    case YOLO_POSE_V8_HALF: // LetterBox and Top Left Crop
    {
        if (iImg.cols >= iImg.rows)
        {
            // scale by target width
            resizeScales_ = iImg.cols / (float)iImgSize.at(0);
            cv::resize(oImg, oImg, cv::Size(iImgSize.at(0), int(std::round(iImg.rows / resizeScales_))));
        }
        else
        {
            // scale by target height
            resizeScales_ = iImg.rows / (float)iImgSize.at(1);
            cv::resize(oImg, oImg, cv::Size(int(std::round(iImg.cols / resizeScales_)), iImgSize.at(1)));
        }
        // Note: cv::Mat takes (rows, cols) = (height, width)
        cv::Mat tempImg = cv::Mat::zeros(iImgSize.at(1), iImgSize.at(0), CV_8UC3);
        oImg.copyTo(tempImg(cv::Rect(0, 0, oImg.cols, oImg.rows)));
        oImg = tempImg;
        break;
    }
    case YOLO_CLS: // CenterCrop
    case YOLO_CLS_HALF:
    {
        int h = iImg.rows;
        int w = iImg.cols;
        int m = min(h, w);
        int top = (h - m) / 2;
        int left = (w - m) / 2;
        cv::resize(oImg(cv::Rect(left, top, m, m)), oImg, cv::Size(iImgSize.at(0), iImgSize.at(1)));
        break;
    }
    }
    return RET_OK;
}


const char* YOLO_V8::CreateSession(DL_INIT_PARAM& iParams) {
    const char* Ret = RET_OK;
    if (session_)
    {
        // Clear node names from previous declaration
        for (auto& name : inputNodeNames_) {
            delete[] name;
        }
        inputNodeNames_.clear();

        for (auto& name : outputNodeNames_) {
            delete[] name;
        }
        outputNodeNames_.clear();
    }
    std::regex pattern("[\u4e00-\u9fa5]");
    bool result = std::regex_search(iParams.modelPath, pattern);
    if (result)
    {
        Ret = "[YOLO_V8]:Your model path is error. Change your model path without chinese characters.";
        std::cout << Ret << std::endl;
        return Ret;
    }
    try
    {
        rectConfidenceThreshold_ = iParams.rectConfidenceThreshold;
        iouThreshold_ = iParams.iouThreshold;
        imgSize_ = iParams.imgSize;
        modelType_ = iParams.modelType;
        cudaEnable_ = iParams.cudaEnable;
        env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Yolo");
        Ort::SessionOptions sessionOption;
        if (iParams.cudaEnable)
        {
            OrtCUDAProviderOptions cudaOption;
            cudaOption.device_id = 0;
            sessionOption.AppendExecutionProvider_CUDA(cudaOption);
        }
        sessionOption.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        sessionOption.SetIntraOpNumThreads(iParams.intraOpNumThreads);
        sessionOption.SetLogSeverityLevel(iParams.logSeverityLevel);

        const char *modelPath = iParams.modelPath.c_str();
        session_ = std::make_unique<Ort::Session>(env_, modelPath, sessionOption);
        Ort::AllocatorWithDefaultOptions allocator;
        size_t inputNodesNum = session_->GetInputCount();
        for (size_t i = 0; i < inputNodesNum; i++)
        {
            Ort::AllocatedStringPtr input_node_name = session_->GetInputNameAllocated(i, allocator);
            char* temp_buf = new char[50];
            strcpy(temp_buf, input_node_name.get());
            inputNodeNames_.push_back(temp_buf);
        }
        size_t OutputNodesNum = session_->GetOutputCount();
        for (size_t i = 0; i < OutputNodesNum; i++)
        {
            Ort::AllocatedStringPtr output_node_name = session_->GetOutputNameAllocated(i, allocator);
            char* temp_buf = new char[10];
            strcpy(temp_buf, output_node_name.get());
            outputNodeNames_.push_back(temp_buf);
        }
        options = Ort::RunOptions{ nullptr };
        WarmUpSession();
        return RET_OK;
    }
    catch (const std::exception& e)
    {
        const char* str1 = "[YOLO_V8]:";
        const char* str2 = e.what();
        std::string result = std::string(str1) + std::string(str2);
        char* merged = new char[result.length() + 1];
        std::strcpy(merged, result.c_str());
        std::cout << merged << std::endl;
        delete[] merged;
        return "[YOLO_V8]:Create session failed.";
    }

}


const char* YOLO_V8::RunSession(const cv::Mat& iImg, std::vector<DL_RESULT>& oResult) {
#ifdef benchmark
    clock_t starttime_1 = clock();
#endif
    const char* Ret = RET_OK;
    cv::Mat processedImg;
    PreProcess(iImg, imgSize_, processedImg);
    if (modelType_ < 4)
    {
        float* blob = new float[processedImg.total() * 3];
        BlobFromImage(processedImg, blob);
        // ONNX expects {N, C, H, W} = {1, 3, height, width}
        std::vector<int64_t> inputNodeDims = { 1, 3, imgSize_.at(1), imgSize_.at(0) };
        TensorProcess(starttime_1, iImg, blob, inputNodeDims, oResult);
    }
    else
    {
#ifdef YOLO_ONNX_ROS_CUDA_ENABLED
        half* blob = new half[processedImg.total() * 3];
        BlobFromImage(processedImg, blob);
        std::vector<int64_t> inputNodeDims = { 1, 3, imgSize_.at(1), imgSize_.at(0) };
        TensorProcess(starttime_1, iImg, blob, inputNodeDims, oResult);
#endif
    }

    return Ret;
}


template<typename N>
char* YOLO_V8::TensorProcess(clock_t& starttime_1, const cv::Mat&, N& blob, std::vector<int64_t>& inputNodeDims,
    std::vector<DL_RESULT>& oResult) {
    Ort::Value inputTensor = Ort::Value::CreateTensor<typename std::remove_pointer<N>::type>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * imgSize_.at(0) * imgSize_.at(1),
        inputNodeDims.data(), inputNodeDims.size());
#ifdef benchmark
    clock_t starttime_2 = clock();
#endif // benchmark
    auto outputTensor = session_->Run(options, inputNodeNames_.data(), &inputTensor, 1, outputNodeNames_.data(),
        outputNodeNames_.size());
#ifdef benchmark
    clock_t starttime_3 = clock();
#endif // benchmark

    Ort::TypeInfo typeInfo = outputTensor.front().GetTypeInfo();
    auto tensor_info = typeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> outputNodeDims = tensor_info.GetShape();
    auto output = outputTensor.front().GetTensorMutableData<typename std::remove_pointer<N>::type>();
    delete[] blob;
    switch (modelType_)
    {
    case YOLO_DETECT_V8:
    case YOLO_DETECT_V8_HALF:
    {
        int signalResultNum = outputNodeDims[1];//84
        int strideNum = outputNodeDims[2];//8400
        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        cv::Mat rawData;
        if (modelType_ == YOLO_DETECT_V8)
        {
            // FP32
            rawData = cv::Mat(signalResultNum, strideNum, CV_32F, output);
        }
        else
        {
            // FP16
            rawData = cv::Mat(signalResultNum, strideNum, CV_16F, output);
            rawData.convertTo(rawData, CV_32F);
        }
        // Note:
        // ultralytics add transpose operator to the output of yolov8 model.which make yolov8/v5/v7 has same shape
        // https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
        rawData = rawData.t();

        float* data = (float*)rawData.data;

        for (int i = 0; i < strideNum; ++i)
        {
            float* classesScores = data + 4;
            cv::Mat scores(1, this->classes.size(), CV_32FC1, classesScores);
            cv::Point class_id;
            double maxClassScore;
            cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);
            if (maxClassScore > rectConfidenceThreshold_)
            {
                confidences.push_back(maxClassScore);
                class_ids.push_back(class_id.x);
                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                int left = int((x - 0.5 * w) * resizeScales_);
                int top = int((y - 0.5 * h) * resizeScales_);

                int width = int(w * resizeScales_);
                int height = int(h * resizeScales_);

                boxes.push_back(cv::Rect(left, top, width, height));
            }
            data += signalResultNum;
        }
        std::vector<int> nmsResult;
        cv::dnn::NMSBoxes(boxes, confidences, rectConfidenceThreshold_, iouThreshold_, nmsResult);
        for (uint i = 0; i < nmsResult.size(); ++i)
        {
            int idx = nmsResult[i];
            DL_RESULT result;
            result.classId = class_ids[idx];
            result.confidence = confidences[idx];
            result.box = boxes[idx];
            oResult.push_back(result);
        }

#ifdef benchmark
        clock_t starttime_4 = clock();
        double pre_process_time = (double)(starttime_2 - starttime_1) / CLOCKS_PER_SEC * 1000;
        double process_time = (double)(starttime_3 - starttime_2) / CLOCKS_PER_SEC * 1000;
        double post_process_time = (double)(starttime_4 - starttime_3) / CLOCKS_PER_SEC * 1000;
        if (cudaEnable_)
        {
            std::cout << "[YOLO_V8(CUDA)]: " << pre_process_time << "ms pre-process, " << process_time << "ms inference, " << post_process_time << "ms post-process." << std::endl;
        }
        else
        {
            std::cout << "[YOLO_V8(CPU)]: " << pre_process_time << "ms pre-process, " << process_time << "ms inference, " << post_process_time << "ms post-process." << std::endl;
        }
#endif // benchmark

        break;
    }
    case YOLO_CLS:
    case YOLO_CLS_HALF:
    {
        cv::Mat rawData;
        if (modelType_ == YOLO_CLS) {
            // FP32
            rawData = cv::Mat(1, this->classes.size(), CV_32F, output);
        } else {
            // FP16
            rawData = cv::Mat(1, this->classes.size(), CV_16F, output);
            rawData.convertTo(rawData, CV_32F);
        }
        float *data = (float *) rawData.data;

        DL_RESULT result;
        for (uint i = 0; i < this->classes.size(); i++)
        {
            result.classId = i;
            result.confidence = data[i];
            oResult.push_back(result);
        }
        break;
    }
    default:
        std::cout << "[YOLO_V8]: " << "Not support model type." << std::endl;
    }
    return RET_OK;

}


char* YOLO_V8::WarmUpSession() {
    clock_t starttime_1 = clock();
    // cv::Size takes (width, height)
    cv::Mat iImg = cv::Mat(cv::Size(imgSize_.at(0), imgSize_.at(1)), CV_8UC3);
    cv::Mat processedImg;
    PreProcess(iImg, imgSize_, processedImg);
    if (modelType_ < 4)
    {
        float* blob = new float[iImg.total() * 3];
        BlobFromImage(processedImg, blob);
        std::vector<int64_t> YOLO_input_node_dims = { 1, 3, imgSize_.at(1), imgSize_.at(0) };
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * imgSize_.at(0) * imgSize_.at(1),
            YOLO_input_node_dims.data(), YOLO_input_node_dims.size());
        auto output_tensors = session_->Run(options, inputNodeNames_.data(), &input_tensor, 1, outputNodeNames_.data(),
            outputNodeNames_.size());
        delete[] blob;
        clock_t starttime_4 = clock();
        double post_process_time = (double)(starttime_4 - starttime_1) / CLOCKS_PER_SEC * 1000;
        if (cudaEnable_)
        {
            std::cout << "[YOLO_V8(CUDA)]: " << "Cuda warm-up cost " << post_process_time << " ms. " << std::endl;
        }
    }
    else
    {
#ifdef YOLO_ONNX_ROS_CUDA_ENABLED
        half* blob = new half[iImg.total() * 3];
        BlobFromImage(processedImg, blob);
        std::vector<int64_t> YOLO_input_node_dims = { 1, 3, imgSize_.at(1), imgSize_.at(0) };
        Ort::Value input_tensor = Ort::Value::CreateTensor<half>(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * imgSize_.at(0) * imgSize_.at(1), YOLO_input_node_dims.data(), YOLO_input_node_dims.size());
        auto output_tensors = session_->Run(options, inputNodeNames_.data(), &input_tensor, 1, outputNodeNames_.data(), outputNodeNames_.size());
        delete[] blob;
        clock_t starttime_4 = clock();
        double post_process_time = (double)(starttime_4 - starttime_1) / CLOCKS_PER_SEC * 1000;
        if (cudaEnable_)
        {
            std::cout << "[YOLO_V8(CUDA)]: " << "Cuda warm-up cost " << post_process_time << " ms. " << std::endl;
        }
#endif
    }
    return RET_OK;
}
