#include "yolo_onnx_ros/yolo_inference.hpp"

#include <cmath>
#include <console_bridge/console.h>
#include <cstdint>
#include <ctime>
#include <exception>

#include <cstddef>
#include <memory>
#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#define benchmark
#define min(a, b) (((a) < (b)) ? (a) : (b))

#if defined(YOLO_ONNX_ROS_CUDA_ENABLED) && YOLO_ONNX_ROS_CUDA_ENABLED
namespace Ort
{
template <> struct TypeToTensorType<half>
{
    static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
};
} // namespace Ort
#endif

/**
 * @brief Flattened image to blob (and normalized) for deep learning inference. Also reorganize from HWC to CHW.
 *
 * @tparam T
 * @param iImg
 * @param iBlob
 * @return const char*
 */
namespace
{
/// @brief Whether the model takes FP32 input; the *_HALF model types take FP16 input.
bool isFloat32(ModelType model_type)
{
    return model_type == ModelType::YOLO_DETECT_V8 || model_type == ModelType::YOLO_POSE ||
           model_type == ModelType::YOLO_CLS;
}

template <typename T> const char* blobFromImage(const cv::Mat& iImg, T& iBlob)
{
    int const channels = iImg.channels();
    int const img_height = iImg.rows;
    int const img_width = iImg.cols;

    for (int c = 0; c < channels; c++)
    {
        for (int h = 0; h < img_height; h++)
        {
            for (int w = 0; w < img_width; w++)
            {
                iBlob[(c * img_width * img_height) + (h * img_width) + w] =
                    std::remove_pointer_t<T>(static_cast<float>(iImg.at<cv::Vec3b>(h, w)[c]) / 255.0F);
            }
        }
    }
    return RET_OK;
}
} // namespace

const char* YoloV8::preProcess(const cv::Mat& iImg, const std::vector<int>& iImgSize, cv::Mat& oImg)
{
    // Basic validation
    if (iImg.empty())
        return "[YOLO_V8]: Empty input image.";
    if (iImgSize.size() != 2 || iImgSize[0] <= 0 || iImgSize[1] <= 0)
        return "[YOLO_V8]: Invalid target image size.";

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
    case ModelType::YOLO_DETECT_V8:
    case ModelType::YOLO_POSE:
    case ModelType::YOLO_DETECT_V8_HALF:
    case ModelType::YOLO_POSE_V8_HALF: // LetterBox and Top Left Crop
    {
        if (iImg.cols >= iImg.rows)
        {
            // scale by target width
            resizeScales_ = static_cast<float>(iImg.cols) / static_cast<float>(iImgSize.at(0));
            cv::resize(
                oImg,
                oImg,
                cv::Size(iImgSize.at(0), static_cast<int>(std::round(static_cast<float>(iImg.rows) / resizeScales_))));
        }
        else
        {
            // scale by target height
            resizeScales_ = static_cast<float>(iImg.rows) / static_cast<float>(iImgSize.at(1));
            cv::resize(
                oImg,
                oImg,
                cv::Size(static_cast<int>(std::round(static_cast<float>(iImg.cols) / resizeScales_)), iImgSize.at(1)));
        }
        // Note: cv::Mat takes (rows, cols) = (height, width)
        cv::Mat const temp_img = cv::Mat::zeros(iImgSize.at(1), iImgSize.at(0), CV_8UC3);
        oImg.copyTo(temp_img(cv::Rect(0, 0, oImg.cols, oImg.rows)));
        oImg = temp_img;
        break;
    }
    case ModelType::YOLO_CLS: // CenterCrop
    case ModelType::YOLO_CLS_HALF:
    {
        int const h = iImg.rows;
        int const w = iImg.cols;
        int const m = min(h, w);
        int const top = (h - m) / 2;
        int const left = (w - m) / 2;
        cv::resize(oImg(cv::Rect(left, top, m, m)), oImg, cv::Size(iImgSize.at(0), iImgSize.at(1)));
        break;
    }
    }
    return RET_OK;
}

const char* YoloV8::createSession(DlInitParam& iParams)
{
    const char* ret = RET_OK;
    // Clear node names from a previous session
    inputNodeNameStorage_.clear();
    outputNodeNameStorage_.clear();
    inputNodeNames_.clear();
    outputNodeNames_.clear();
    std::regex const pattern("[\u4e00-\u9fa5]");
    bool const result = std::regex_search(iParams.modelPath, pattern);
    if (result)
    {
        CONSOLE_BRIDGE_logInform(
            "[YOLO_V8]:Your model path is error. Change your model path without chinese characters.");
        return ret;
    }
    try
    {
        rectConfidenceThreshold_ = iParams.rectConfidenceThreshold;
        iouThreshold_ = iParams.iouThreshold;
        imgSize_ = iParams.imgSize;
        modelType_ = iParams.modelType;
        cudaEnable_ = iParams.cudaEnable;
        env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Yolo");
        Ort::SessionOptions session_option;
        if (iParams.cudaEnable)
        {
            OrtCUDAProviderOptions cuda_option;
            cuda_option.device_id = 0;
            session_option.AppendExecutionProvider_CUDA(cuda_option);
        }
        session_option.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_option.SetIntraOpNumThreads(iParams.intraOpNumThreads);
        session_option.SetLogSeverityLevel(iParams.logSeverityLevel);

        const char* model_path = iParams.modelPath.c_str();
        session_ = std::make_unique<Ort::Session>(env_, model_path, session_option);
        Ort::AllocatorWithDefaultOptions const allocator;
        size_t const input_nodes_num = session_->GetInputCount();
        for (size_t i = 0; i < input_nodes_num; i++)
        {
            Ort::AllocatedStringPtr const input_node_name = session_->GetInputNameAllocated(i, allocator);
            inputNodeNameStorage_.emplace_back(input_node_name.get());
        }
        size_t const output_nodes_num = session_->GetOutputCount();
        for (size_t i = 0; i < output_nodes_num; i++)
        {
            Ort::AllocatedStringPtr const output_node_name = session_->GetOutputNameAllocated(i, allocator);
            outputNodeNameStorage_.emplace_back(output_node_name.get());
        }
        // Take the pointers only after the storage vectors are complete, so no reallocation invalidates them
        for (const auto& name : inputNodeNameStorage_)
        {
            inputNodeNames_.push_back(name.c_str());
        }
        for (const auto& name : outputNodeNameStorage_)
        {
            outputNodeNames_.push_back(name.c_str());
        }
        options = Ort::RunOptions{nullptr};
        warmUpSession();
        return RET_OK;
    }
    catch (const std::exception& e)
    {
        std::string const error_msg = "[YOLO_V8]: Failed to create session: " + std::string(e.what());
        CONSOLE_BRIDGE_logError(error_msg.c_str());
        throw std::runtime_error(error_msg);
    }
}

const char* YoloV8::runSession(const cv::Mat& iImg, std::vector<DlResult>& oResult)
{
#ifdef benchmark
    clock_t starttime_1 = clock();
#endif
    const char* ret = RET_OK;
    cv::Mat processed_img;
    preProcess(iImg, imgSize_, processed_img);
    if (isFloat32(modelType_))
    {
        auto* blob = new float[processed_img.total() * 3];
        blobFromImage(processed_img, blob);
        // ONNX expects {N, C, H, W} = {1, 3, height, width}
        std::vector<int64_t> input_node_dims = {1, 3, imgSize_.at(1), imgSize_.at(0)};
        tensorProcess(starttime_1, blob, input_node_dims, oResult);
    }
    else
    {
#if defined(YOLO_ONNX_ROS_CUDA_ENABLED) && YOLO_ONNX_ROS_CUDA_ENABLED
        auto* blob = new half[processed_img.total() * 3];
        blobFromImage(processed_img, blob);
        std::vector<int64_t> input_node_dims = {1, 3, imgSize_.at(1), imgSize_.at(0)};
        tensorProcess(starttime_1, blob, input_node_dims, oResult);
#endif
    }

    return ret;
}

template <typename N>
const char* YoloV8::tensorProcess(clock_t& starttime_1,
                                  N& blob,
                                  std::vector<int64_t>& inputNodeDims,
                                  std::vector<DlResult>& oResult)
{
    Ort::Value const input_tensor = Ort::Value::CreateTensor<std::remove_pointer_t<N>>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
        blob,
        3 * imgSize_.at(0) * imgSize_.at(1),
        inputNodeDims.data(),
        inputNodeDims.size());
#ifdef benchmark
    clock_t const starttime_2 = clock();
#endif // benchmark
    auto output_tensor = session_->Run(
        options, inputNodeNames_.data(), &input_tensor, 1, outputNodeNames_.data(), outputNodeNames_.size());
#ifdef benchmark
    clock_t const starttime_3 = clock();
#endif // benchmark

    Ort::TypeInfo const type_info = output_tensor.front().GetTypeInfo();
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> output_node_dims = tensor_info.GetShape();
    auto output = output_tensor.front().GetTensorMutableData<std::remove_pointer_t<N>>();
    delete[] blob;
    switch (modelType_)
    {
    case ModelType::YOLO_DETECT_V8:
    case ModelType::YOLO_DETECT_V8_HALF:
    {
        // If the model is not trained with NMS (like YOLO26), the output shape will be [1, N, 6] = x1, y1, x2, y2,
        // conf, class where N is the number of detections.
        if (output_node_dims.size() == 3 && output_node_dims[2] == 6 && output_node_dims[1] > 6)
        {
            cv::Mat raw(static_cast<int>(output_node_dims[1]), 6, CV_32F);
            if (modelType_ == ModelType::YOLO_DETECT_V8)
                raw = cv::Mat(static_cast<int>(output_node_dims[1]), 6, CV_32F, output).clone();
            else
            {
                cv::Mat const half(static_cast<int>(output_node_dims[1]), 6, CV_16F, output);
                half.convertTo(raw, CV_32F);
            }
            for (int i = 0; i < raw.rows; ++i)
            {
                const float* d = raw.ptr<float>(i);
                if (d[4] <= rectConfidenceThreshold_)
                    continue;
                DlResult result;
                result.classId = static_cast<int>(d[5]);
                result.confidence = d[4];
                result.box = cv::Rect(static_cast<int>(d[0] * resizeScales_),
                                      static_cast<int>(d[1] * resizeScales_),
                                      static_cast<int>((d[2] - d[0]) * resizeScales_),
                                      static_cast<int>((d[3] - d[1]) * resizeScales_));
                oResult.push_back(result);
            }
            break;
        }
        // If the model is trained with NMS, the output shape will be [1, M, 6] where M is the number of detections
        // after NMS.
        int const signal_result_num =
            static_cast<int>(output_node_dims[1]); // Dataset-specific: e.g., 605 for OIV7 (4 bbox + 601 classes),
                                                   // 84 for COCO (4 bbox + 80 classes)
        int const stride_num = static_cast<int>(output_node_dims[2]); // 8400
        int const num_classes = signal_result_num - 4; // 601 for OIV7, 80 for COCO (4 rows are bbox)
        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        cv::Mat raw_data;
        if (modelType_ == ModelType::YOLO_DETECT_V8)
        {
            // FP32
            raw_data = cv::Mat(signal_result_num, stride_num, CV_32F, output);
        }
        else
        {
            // FP16
            raw_data = cv::Mat(signal_result_num, stride_num, CV_16F, output);
            raw_data.convertTo(raw_data, CV_32F);
        }
        // Note:
        // ultralytics add transpose operator to the output of yolov8 model.which make yolov8/v5/v7 has same shape
        // https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
        raw_data = raw_data.t();

        auto* data = reinterpret_cast<float*>(raw_data.data);

        for (int i = 0; i < stride_num; ++i)
        {
            float* classes_scores = data + 4;
            cv::Mat const scores(
                1, num_classes, CV_32FC1, classes_scores); // Use numClasses instead of this->classes.size()
            cv::Point class_id;
            double max_class_score = NAN;
            cv::minMaxLoc(scores, nullptr, &max_class_score, nullptr, &class_id);
            if (max_class_score > rectConfidenceThreshold_)
            {
                confidences.push_back(static_cast<float>(max_class_score));
                class_ids.push_back(class_id.x);
                float const x = data[0];
                float const y = data[1];
                float const w = data[2];
                float const h = data[3];

                int const left = static_cast<int>((x - 0.5 * w) * resizeScales_);
                int const top = static_cast<int>((y - 0.5 * h) * resizeScales_);

                int const width = static_cast<int>(w * resizeScales_);
                int const height = static_cast<int>(h * resizeScales_);

                boxes.emplace_back(left, top, width, height);
            }
            data += signal_result_num;
        }
        std::vector<int> nms_result;
        cv::dnn::NMSBoxes(boxes, confidences, rectConfidenceThreshold_, iouThreshold_, nms_result);
        for (int const idx : nms_result)
        {
            DlResult result;
            result.classId = class_ids[idx];
            result.confidence = confidences[idx];
            result.box = boxes[idx];
            oResult.push_back(result);
        }

#ifdef benchmark
        clock_t const starttime_4 = clock();
        double const pre_process_time = static_cast<double>(starttime_2 - starttime_1) / CLOCKS_PER_SEC * 1000;
        double const process_time = static_cast<double>(starttime_3 - starttime_2) / CLOCKS_PER_SEC * 1000;
        double const post_process_time = static_cast<double>(starttime_4 - starttime_3) / CLOCKS_PER_SEC * 1000;
        std::stringstream ss;
        if (cudaEnable_)
        {

            ss << "[YOLO_V8(CUDA)]: ";
        }
        else
        {
            ss << "[YOLO_V8(CPU)]: ";
        }
        ss << pre_process_time << "ms pre-process, " << process_time << "ms inference, " << post_process_time
           << "ms post-process." << '\n';
        CONSOLE_BRIDGE_logInform(ss.str().c_str());
#endif // benchmark

        break;
    }
    case ModelType::YOLO_CLS:
    case ModelType::YOLO_CLS_HALF:
    {
        cv::Mat raw_data;
        if (modelType_ == ModelType::YOLO_CLS)
        {
            // FP32
            raw_data = cv::Mat(1, this->classes.size(), CV_32F, output);
        }
        else
        {
            // FP16
            raw_data = cv::Mat(1, this->classes.size(), CV_16F, output);
            raw_data.convertTo(raw_data, CV_32F);
        }
        auto* data = reinterpret_cast<float*>(raw_data.data);

        DlResult result;
        for (size_t i = 0; i < this->classes.size(); i++)
        {
            result.classId = static_cast<int>(i);
            result.confidence = data[i];
            oResult.push_back(result);
        }
        break;
    }
    default: CONSOLE_BRIDGE_logError("[YOLO_V8]: Not support model type.");
    }
    return RET_OK;
}

const char* YoloV8::warmUpSession()
{
    clock_t const starttime_1 = clock();
    // cv::Size takes (width, height)
    cv::Mat const i_img = cv::Mat(cv::Size(imgSize_.at(0), imgSize_.at(1)), CV_8UC3);
    cv::Mat processed_img;
    preProcess(i_img, imgSize_, processed_img);
    if (isFloat32(modelType_))
    {
        auto* blob = new float[i_img.total() * 3];
        blobFromImage(processed_img, blob);
        std::vector<int64_t> yolo_input_node_dims = {1, 3, imgSize_.at(1), imgSize_.at(0)};
        Ort::Value const input_tensor = Ort::Value::CreateTensor<float>(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
            blob,
            3 * static_cast<size_t>(imgSize_.at(0)) * static_cast<size_t>(imgSize_.at(1)),
            yolo_input_node_dims.data(),
            yolo_input_node_dims.size());
        auto output_tensors = session_->Run(
            options, inputNodeNames_.data(), &input_tensor, 1, outputNodeNames_.data(), outputNodeNames_.size());
        delete[] blob;
        clock_t const starttime_4 = clock();
        double const post_process_time = static_cast<double>(starttime_4 - starttime_1) / CLOCKS_PER_SEC * 1000;
        if (cudaEnable_)
        {
            CONSOLE_BRIDGE_logInform("[YOLO_V8(CUDA)]: CUDA warm-up cost %f ms.", post_process_time);
        }
    }
    else
    {
#if defined(YOLO_ONNX_ROS_CUDA_ENABLED) && YOLO_ONNX_ROS_CUDA_ENABLED
        auto* blob = new half[i_img.total() * 3];
        blobFromImage(processed_img, blob);
        std::vector<int64_t> yolo_input_node_dims = {1, 3, imgSize_.at(1), imgSize_.at(0)};
        Ort::Value const input_tensor = Ort::Value::CreateTensor<half>(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
            blob,
            3 * static_cast<size_t>(imgSize_.at(0)) * static_cast<size_t>(imgSize_.at(1)),
            yolo_input_node_dims.data(),
            yolo_input_node_dims.size());
        auto output_tensors = session_->Run(
            options, inputNodeNames_.data(), &input_tensor, 1, outputNodeNames_.data(), outputNodeNames_.size());
        delete[] blob;
        clock_t const starttime_4 = clock();
        double const post_process_time = static_cast<double>(starttime_4 - starttime_1) / CLOCKS_PER_SEC * 1000;
        if (cudaEnable_)
        {
            CONSOLE_BRIDGE_logInform("[YOLO_V8(CUDA)]: CUDA warm-up cost %f ms.", post_process_time);
        }
#endif
    }
    return RET_OK;
}
