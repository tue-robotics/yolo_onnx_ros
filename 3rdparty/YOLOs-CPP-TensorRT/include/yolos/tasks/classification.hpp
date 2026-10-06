#pragma once

// ============================================================================
// YOLO Image Classification (TensorRT Backend)
// ============================================================================
// Image classification using YOLO models (v11, v12, YOLO26).
// Supports efficient classification with Ultralytics-style preprocessing.
// Refactored to inherit TrtSessionBase for unified TensorRT inference.
//
// Author: YOLOs-TRT Team
// ============================================================================

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "yolos/core/types.hpp"
#include "yolos/core/version.hpp"
#include "yolos/core/utils.hpp"
#include "yolos/core/trt_session_base.hpp"

namespace yolos {
namespace cls {

// ============================================================================
// Classification Result Structure
// ============================================================================

/// @brief Classification result containing class ID, confidence, and class name
struct ClassificationResult {
    int classId{-1};
    float confidence{0.0f};
    std::string className{};

    ClassificationResult() = default;
    ClassificationResult(int id, float conf, std::string name)
        : classId(id), confidence(conf), className(std::move(name)) {}
};

// ============================================================================
// Drawing Utility for Classification
// ============================================================================

/// @brief Draw classification result on an image
inline void drawClassificationResult(cv::Mat& image,
                                     const ClassificationResult& result,
                                     const cv::Point& position = cv::Point(10, 30),
                                     const cv::Scalar& textColor = cv::Scalar(0, 255, 0),
                                     const cv::Scalar& bgColor = cv::Scalar(0, 0, 0)) {
    if (image.empty() || result.classId == -1) return;

    std::ostringstream ss;
    ss << result.className << ": " << std::fixed << std::setprecision(1) << result.confidence * 100 << "%";
    std::string text = ss.str();

    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = std::min(image.rows, image.cols) * 0.001;
    fontScale = std::max(fontScale, 0.5);
    int thickness = std::max(1, static_cast<int>(fontScale * 2));
    int baseline = 0;

    cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);

    cv::Point textPos = position;
    textPos.y = std::max(textPos.y, textSize.height + 5);

    cv::Point bgTopLeft(textPos.x - 2, textPos.y - textSize.height - 5);
    cv::Point bgBottomRight(textPos.x + textSize.width + 2, textPos.y + 5);

    bgTopLeft.x = utils::clamp(bgTopLeft.x, 0, image.cols - 1);
    bgTopLeft.y = utils::clamp(bgTopLeft.y, 0, image.rows - 1);
    bgBottomRight.x = utils::clamp(bgBottomRight.x, 0, image.cols - 1);
    bgBottomRight.y = utils::clamp(bgBottomRight.y, 0, image.rows - 1);

    cv::rectangle(image, bgTopLeft, bgBottomRight, bgColor, cv::FILLED);
    cv::putText(image, text, textPos, fontFace, fontScale, textColor, thickness, cv::LINE_AA);
}

// ============================================================================
// YOLOClassifier Base Class
// ============================================================================

/// @brief YOLO classifier for image classification (TensorRT backend)
class YOLOClassifier : public TrtSessionBase {
public:
    /// @brief Constructor
    /// @param enginePath Path to the TensorRT engine file
    /// @param labelsPath Path to the class names file
    /// @param targetInputShape Target input shape for preprocessing
    /// @param dlaCore DLA core index (-1 = GPU, 0/1 = DLA on Jetson)
    YOLOClassifier(const std::string& enginePath,
                   const std::string& labelsPath,
                   const cv::Size& targetInputShape = cv::Size(224, 224),
                   int dlaCore = -1)
        : TrtSessionBase(enginePath, dlaCore),
          inputImageShape_(targetInputShape) {

        classNames_ = utils::getClassNames(labelsPath);

        // Derive input shape from engine if available (override user default)
        if (inputShape_.width > 0 && inputShape_.height > 0) {
            inputImageShape_ = inputShape_;
        }

        // Determine number of classes from output shape
        const auto& outShape = getOutputShape(0);
        if (outShape.size() >= 2) {
            numClasses_ = static_cast<int>(outShape.back());
        } else if (outShape.size() == 1) {
            numClasses_ = static_cast<int>(outShape[0]);
        }

        std::cout << "[INFO] Classification model loaded" << std::endl;
        std::cout << "[INFO] Input shape: " << inputImageShape_.width << "x" << inputImageShape_.height << std::endl;
        std::cout << "[INFO] Number of classes: " << numClasses_ << std::endl;
    }

    virtual ~YOLOClassifier() = default;

    /// @brief Run classification on an image
    ClassificationResult classify(const cv::Mat& image) {
        if (image.empty()) return {};

        // Preprocess
        std::vector<int64_t> inputTensorShape;
        preprocess(image, inputTensorShape);

        // Run TensorRT inference
        size_t inputCount = utils::vectorProduct(inputTensorShape);
        infer(inputBuffer_.data(), inputCount);

        return postprocess();
    }

    /// @brief Draw classification result on an image
    void drawResult(cv::Mat& image, const ClassificationResult& result,
                    const cv::Point& position = cv::Point(10, 30)) const {
        drawClassificationResult(image, result, position);
    }

    [[nodiscard]] cv::Size getClsInputShape() const { return inputImageShape_; }
    [[nodiscard]] const std::vector<std::string>& getClassNames() const { return classNames_; }

protected:
    cv::Size inputImageShape_;
    std::vector<float> inputBuffer_;
    int numClasses_{0};
    std::vector<std::string> classNames_;

    /// @brief Preprocess image for classification (Ultralytics-style)
    void preprocess(const cv::Mat& image, std::vector<int64_t>& inputTensorShape) {
        int targetSize = inputImageShape_.width;
        int h = image.rows;
        int w = image.cols;

        // Resize: shortest side to target_size, maintaining aspect ratio
        int newH, newW;
        if (h < w) {
            newH = targetSize;
            newW = static_cast<int>(w * targetSize / h);
        } else {
            newW = targetSize;
            newH = static_cast<int>(h * targetSize / w);
        }

        cv::Mat rgbImage;
        cv::cvtColor(image, rgbImage, cv::COLOR_BGR2RGB);

        cv::Mat resized;
        cv::resize(rgbImage, resized, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR);

        // Center crop to target_size x target_size
        int yStart = std::max(0, (newH - targetSize) / 2);
        int xStart = std::max(0, (newW - targetSize) / 2);
        cv::Mat cropped = resized(cv::Rect(xStart, yStart, targetSize, targetSize));

        // Normalize to [0, 1]
        cv::Mat floatImage;
        cropped.convertTo(floatImage, CV_32F, 1.0 / 255.0);

        inputTensorShape = {1, 3, floatImage.rows, floatImage.cols};
        const int finalH = floatImage.rows;
        const int finalW = floatImage.cols;
        size_t tensorSize = 3 * finalH * finalW;
        inputBuffer_.resize(tensorSize);

        // Convert HWC to CHW format
        std::vector<cv::Mat> channels(3);
        cv::split(floatImage, channels);
        for (int c = 0; c < 3; ++c) {
            std::memcpy(inputBuffer_.data() + c * finalH * finalW,
                       channels[c].ptr<float>(), finalH * finalW * sizeof(float));
        }
    }

    /// @brief Postprocess classification output
    ClassificationResult postprocess() {
        const float* rawOutput = getOutputData(0);

        int numScores = numClasses_ > 0 ? numClasses_ : static_cast<int>(classNames_.size());
        if (numScores <= 0) return {};

        // Find max score
        int bestClassId = 0;
        float maxProb = rawOutput[0];

        for (int i = 1; i < numScores; ++i) {
            if (rawOutput[i] > maxProb) {
                maxProb = rawOutput[i];
                bestClassId = i;
            }
        }

        std::string className = (bestClassId >= 0 && static_cast<size_t>(bestClassId) < classNames_.size())
                               ? classNames_[bestClassId]
                               : ("Class_" + std::to_string(bestClassId));

        return ClassificationResult(bestClassId, maxProb, className);
    }
};

// ============================================================================
// Version-Specific Classifier Subclasses
// ============================================================================

/// @brief YOLOv11 classifier
class YOLO11Classifier : public YOLOClassifier {
public:
    YOLO11Classifier(const std::string& enginePath, const std::string& labelsPath, int dlaCore = -1)
        : YOLOClassifier(enginePath, labelsPath, cv::Size(224, 224), dlaCore) {}
};

/// @brief YOLOv12 classifier
class YOLO12Classifier : public YOLOClassifier {
public:
    YOLO12Classifier(const std::string& enginePath, const std::string& labelsPath, int dlaCore = -1)
        : YOLOClassifier(enginePath, labelsPath, cv::Size(224, 224), dlaCore) {}
};

/// @brief YOLO26 classifier
class YOLO26Classifier : public YOLOClassifier {
public:
    YOLO26Classifier(const std::string& enginePath, const std::string& labelsPath, int dlaCore = -1)
        : YOLOClassifier(enginePath, labelsPath, cv::Size(224, 224), dlaCore) {}
};

// ============================================================================
// Factory Function
// ============================================================================

/// @brief Create a classifier with explicit version selection
inline std::unique_ptr<YOLOClassifier> createClassifier(const std::string& enginePath,
                                                        const std::string& labelsPath,
                                                        YOLOVersion version = YOLOVersion::V11,
                                                        int dlaCore = -1) {
    switch (version) {
        case YOLOVersion::V26:
            return std::make_unique<YOLO26Classifier>(enginePath, labelsPath, dlaCore);
        case YOLOVersion::V12:
            return std::make_unique<YOLO12Classifier>(enginePath, labelsPath, dlaCore);
        default:
            return std::make_unique<YOLO11Classifier>(enginePath, labelsPath, dlaCore);
    }
}

} // namespace cls
} // namespace yolos
