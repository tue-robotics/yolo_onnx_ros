#pragma once

// ============================================================================
// YOLO Object Detection (TensorRT Backend)
// ============================================================================
// Object detection using YOLO models with support for multiple versions
// (v7, v8, v10, v11, v26, NAS) through runtime auto-detection or explicit
// selection. Uses TensorRT for high-performance GPU inference.
//
// Author: YOLOs-TRT Team
// ============================================================================

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>
#include <cfloat>

#include "yolos/core/types.hpp"
#include "yolos/core/version.hpp"
#include "yolos/core/utils.hpp"
#include "yolos/core/preprocessing.hpp"
#include "yolos/core/nms.hpp"
#include "yolos/core/drawing.hpp"
#include "yolos/core/trt_session_base.hpp"

namespace yolos {
namespace det {

// ============================================================================
// Detection Result Structure
// ============================================================================

/// @brief Detection result containing bounding box, confidence, and class ID
struct Detection {
    BoundingBox box;    ///< Axis-aligned bounding box
    float conf{0.0f};   ///< Confidence score
    int classId{-1};    ///< Class ID

    Detection() = default;
    Detection(const BoundingBox& box_, float conf_, int classId_)
        : box(box_), conf(conf_), classId(classId_) {}
};

// ============================================================================
// YOLODetector Base Class
// ============================================================================

/// @brief Base YOLO detector with runtime version auto-detection
class YOLODetector : public TrtSessionBase {
public:
    /// @brief Constructor
    /// @param enginePath Path to the TensorRT engine file (.trt / .engine)
    /// @param labelsPath Path to the class names file
    /// @param version YOLO version (Auto for runtime detection)
    /// @param dlaCore DLA core index (-1 = GPU, 0/1 = DLA on Jetson)
    YOLODetector(const std::string& enginePath,
                 const std::string& labelsPath,
                 YOLOVersion version = YOLOVersion::Auto,
                 int dlaCore = -1)
        : TrtSessionBase(enginePath, dlaCore),
          version_(version) {
        classNames_ = utils::getClassNames(labelsPath);
        classColors_ = drawing::generateColors(classNames_);

        // Resolve version once at construction if Auto (avoids per-frame branch)
        if (version_ == YOLOVersion::Auto) {
            version_ = detectVersion();
        }
    }

    virtual ~YOLODetector() = default;

    /// @brief Run detection on an image — full GPU pipeline
    /// @param image Input image (BGR format)
    /// @param confThreshold Confidence threshold
    /// @param iouThreshold IoU threshold for NMS
    /// @return Vector of detections
    virtual std::vector<Detection> detect(const cv::Mat& image,
                                          float confThreshold = 0.4f,
                                          float iouThreshold = 0.45f) {
        // Full GPU pipeline: pinned H2D → CUDA letterbox → TRT infer → D2H
        inferGpu(image);

        // Postprocess using cached scale/pad from GPU preprocess
        return postprocess(image.size(), inputShape_, version_, confThreshold, iouThreshold);
    }

    /// @brief Draw detections on an image
    void drawDetections(cv::Mat& image, const std::vector<Detection>& detections) const {
        for (const auto& det : detections) {
            if (det.classId >= 0 && static_cast<size_t>(det.classId) < classNames_.size()) {
                std::string label = classNames_[det.classId] + ": " +
                                   std::to_string(static_cast<int>(det.conf * 100)) + "%";
                const cv::Scalar& color = classColors_[det.classId % classColors_.size()];
                drawing::drawBoundingBox(image, det.box, label, color);
            }
        }
    }

    /// @brief Draw detections with semi-transparent mask fill
    void drawDetectionsWithMask(cv::Mat& image, const std::vector<Detection>& detections, float alpha = 0.4f) const {
        for (const auto& det : detections) {
            if (det.classId >= 0 && static_cast<size_t>(det.classId) < classNames_.size()) {
                std::string label = classNames_[det.classId] + ": " +
                                   std::to_string(static_cast<int>(det.conf * 100)) + "%";
                const cv::Scalar& color = classColors_[det.classId % classColors_.size()];
                drawing::drawBoundingBoxWithMask(image, det.box, label, color, alpha);
            }
        }
    }

    /// @brief Get class names
    [[nodiscard]] const std::vector<std::string>& getClassNames() const { return classNames_; }

    /// @brief Get class colors
    [[nodiscard]] const std::vector<cv::Scalar>& getClassColors() const { return classColors_; }

protected:
    YOLOVersion version_{YOLOVersion::Auto};
    std::vector<std::string> classNames_;
    std::vector<cv::Scalar> classColors_;

    /// @brief Detect YOLO version from output tensor shape
    YOLOVersion detectVersion() {
        const auto& outputShape = getOutputShape(0);
        return version::detectFromOutputShape(outputShape, numOutputs());
    }

    /// @brief Postprocess dispatch — uses cached scale/pad from GPU preprocess
    virtual std::vector<Detection> postprocess(const cv::Size& originalSize,
                                               const cv::Size& /*resizedShape*/,
                                               YOLOVersion version,
                                               float confThreshold,
                                               float iouThreshold) {
        // Use cached scale/pad computed during GPU preprocessing
        const float invScale = 1.0f / getCachedScale();
        const float padX = getCachedPadX();
        const float padY = getCachedPadY();

        switch (version) {
            case YOLOVersion::V7:
                return postprocessV7(originalSize, invScale, padX, padY, confThreshold, iouThreshold);
            case YOLOVersion::V10:
            case YOLOVersion::V26:
                return postprocessV10(originalSize, invScale, padX, padY, confThreshold);
            case YOLOVersion::NAS:
                return postprocessNAS(originalSize, invScale, padX, padY, confThreshold, iouThreshold);
            default:
                return postprocessStandard(originalSize, invScale, padX, padY, confThreshold, iouThreshold);
        }
    }

    /// @brief Standard postprocess for YOLOv8/v11 format [batch, features, boxes]
    std::vector<Detection> postprocessStandard(const cv::Size& originalSize,
                                               float invScale, float padX, float padY,
                                               float confThreshold, float iouThreshold) {
        const float* __restrict__ rawOutput = getOutputData(0);
        const auto& outputShape = getOutputShape(0);

        const int numFeatures = static_cast<int>(outputShape[1]);
        const int numDets     = static_cast<int>(outputShape[2]);
        const int numClasses  = numFeatures - 4;

        if (numDets == 0 || numClasses <= 0) return {};

        const int imgW = originalSize.width;
        const int imgH = originalSize.height;

        // Pre-size with headroom to avoid realloc in hot loop
        std::vector<BoundingBox> boxes;
        std::vector<float> confs;
        std::vector<int> classIds;
        boxes.reserve(512);
        confs.reserve(512);
        classIds.reserve(512);

        // Pointers to xywh rows for cache-friendly strided access
        const float* pCx = rawOutput;
        const float* pCy = rawOutput + numDets;
        const float* pW  = rawOutput + 2 * numDets;
        const float* pH  = rawOutput + 3 * numDets;

        for (int d = 0; d < numDets; ++d) {
            // Find argmax class (unrolled scan)
            int bestClass = 0;
            float bestScore = rawOutput[4 * numDets + d];
            for (int c = 1; c < numClasses; ++c) {
                const float s = rawOutput[(4 + c) * numDets + d];
                if (s > bestScore) { bestScore = s; bestClass = c; }
            }
            if (bestScore <= confThreshold) continue;

            const float cx = pCx[d], cy = pCy[d], w = pW[d], h = pH[d];
            const float left = (cx - w * 0.5f - padX) * invScale;
            const float top  = (cy - h * 0.5f - padY) * invScale;
            const float bw   = w * invScale;
            const float bh   = h * invScale;

            BoundingBox box;
            box.x      = utils::clamp(static_cast<int>(left), 0, imgW - 1);
            box.y      = utils::clamp(static_cast<int>(top),  0, imgH - 1);
            box.width  = utils::clamp(static_cast<int>(bw),   1, imgW - box.x);
            box.height = utils::clamp(static_cast<int>(bh),   1, imgH - box.y);

            boxes.push_back(box);
            confs.push_back(bestScore);
            classIds.push_back(bestClass);
        }

        std::vector<int> indices;
        nms::NMSBoxesBatched(boxes, confs, classIds, confThreshold, iouThreshold, indices);

        std::vector<Detection> detections;
        detections.reserve(indices.size());
        for (int idx : indices) {
            detections.emplace_back(boxes[idx], confs[idx], classIds[idx]);
        }
        return detections;
    }

    /// @brief Postprocess for YOLOv7 format [batch, boxes, features]
    std::vector<Detection> postprocessV7(const cv::Size& originalSize,
                                         float invScale, float padX, float padY,
                                         float confThreshold, float iouThreshold) {
        const float* rawOutput = getOutputData(0);
        const auto& outputShape = getOutputShape(0);

        const int numDets    = static_cast<int>(outputShape[1]);
        const int numFeats   = static_cast<int>(outputShape[2]);
        const int numClasses = numFeats - 5;

        if (numDets == 0 || numClasses <= 0) return {};

        const int imgW = originalSize.width, imgH = originalSize.height;

        std::vector<BoundingBox> boxes;  boxes.reserve(512);
        std::vector<float> confs;        confs.reserve(512);
        std::vector<int> classIds;       classIds.reserve(512);

        for (int d = 0; d < numDets; ++d) {
            const float* row = rawOutput + d * numFeats;
            if (row[4] <= confThreshold) continue;

            int bestC = 0; float bestS = row[5];
            for (int c = 1; c < numClasses; ++c) {
                if (row[5 + c] > bestS) { bestS = row[5 + c]; bestC = c; }
            }

            const float left = (row[0] - row[2] * 0.5f - padX) * invScale;
            const float top  = (row[1] - row[3] * 0.5f - padY) * invScale;

            BoundingBox box;
            box.x      = utils::clamp(static_cast<int>(left), 0, imgW - 1);
            box.y      = utils::clamp(static_cast<int>(top),  0, imgH - 1);
            box.width  = utils::clamp(static_cast<int>(row[2] * invScale), 1, imgW - box.x);
            box.height = utils::clamp(static_cast<int>(row[3] * invScale), 1, imgH - box.y);

            boxes.push_back(box); confs.push_back(row[4]); classIds.push_back(bestC);
        }

        std::vector<int> indices;
        nms::NMSBoxesBatched(boxes, confs, classIds, confThreshold, iouThreshold, indices);

        std::vector<Detection> dets; dets.reserve(indices.size());
        for (int idx : indices) dets.emplace_back(boxes[idx], confs[idx], classIds[idx]);
        return dets;
    }

    /// @brief Postprocess for YOLOv10/v26 format [batch, boxes, 6] (end-to-end, no NMS)
    std::vector<Detection> postprocessV10(const cv::Size& originalSize,
                                          float invScale, float padX, float padY,
                                          float confThreshold) {
        const float* rawOutput = getOutputData(0);
        const int numDets = static_cast<int>(getOutputShape(0)[1]);
        const int imgW = originalSize.width, imgH = originalSize.height;

        std::vector<Detection> dets; dets.reserve(numDets);

        for (int i = 0; i < numDets; ++i) {
            const float* r = rawOutput + i * 6;
            if (r[4] <= confThreshold) continue;

            const float x1 = (r[0] - padX) * invScale;
            const float y1 = (r[1] - padY) * invScale;
            const float x2 = (r[2] - padX) * invScale;
            const float y2 = (r[3] - padY) * invScale;

            BoundingBox box;
            box.x      = utils::clamp(static_cast<int>(x1), 0, imgW - 1);
            box.y      = utils::clamp(static_cast<int>(y1), 0, imgH - 1);
            box.width  = utils::clamp(static_cast<int>(x2 - x1), 1, imgW - box.x);
            box.height = utils::clamp(static_cast<int>(y2 - y1), 1, imgH - box.y);

            dets.emplace_back(box, r[4], static_cast<int>(r[5]));
        }
        return dets;
    }

    /// @brief Postprocess for YOLO-NAS format (two outputs: boxes and scores)
    std::vector<Detection> postprocessNAS(const cv::Size& originalSize,
                                          float invScale, float padX, float padY,
                                          float confThreshold, float iouThreshold) {
        if (numOutputs() < 2)
            return postprocessStandard(originalSize, invScale, padX, padY, confThreshold, iouThreshold);

        const float* boxOut   = getOutputData(0);
        const float* scoreOut = getOutputData(1);
        const int numDets     = static_cast<int>(getOutputShape(0)[1]);
        const int numClasses  = static_cast<int>(getOutputShape(1)[2]);
        const int imgW = originalSize.width, imgH = originalSize.height;

        std::vector<BoundingBox> boxes; boxes.reserve(512);
        std::vector<float> confs;       confs.reserve(512);
        std::vector<int> classIds;      classIds.reserve(512);

        for (int i = 0; i < numDets; ++i) {
            int bestC = 0; float bestS = scoreOut[i * numClasses];
            for (int c = 1; c < numClasses; ++c) {
                const float s = scoreOut[i * numClasses + c];
                if (s > bestS) { bestS = s; bestC = c; }
            }
            if (bestS <= confThreshold) continue;

            const float x1 = (boxOut[i * 4 + 0] - padX) * invScale;
            const float y1 = (boxOut[i * 4 + 1] - padY) * invScale;
            const float x2 = (boxOut[i * 4 + 2] - padX) * invScale;
            const float y2 = (boxOut[i * 4 + 3] - padY) * invScale;

            BoundingBox box;
            box.x      = utils::clamp(static_cast<int>(x1), 0, imgW - 1);
            box.y      = utils::clamp(static_cast<int>(y1), 0, imgH - 1);
            box.width  = utils::clamp(static_cast<int>(x2 - x1), 1, imgW - box.x);
            box.height = utils::clamp(static_cast<int>(y2 - y1), 1, imgH - box.y);

            boxes.push_back(box); confs.push_back(bestS); classIds.push_back(bestC);
        }

        std::vector<int> indices;
        nms::NMSBoxesBatched(boxes, confs, classIds, confThreshold, iouThreshold, indices);

        std::vector<Detection> dets; dets.reserve(indices.size());
        for (int idx : indices) dets.emplace_back(boxes[idx], confs[idx], classIds[idx]);
        return dets;
    }
};

// ============================================================================
// Version-Specific Detector Subclasses
// ============================================================================

/// @brief YOLOv7 detector (forces V7 postprocessing)
class YOLOv7Detector : public YOLODetector {
public:
    YOLOv7Detector(const std::string& enginePath, const std::string& labelsPath, int dlaCore = -1)
        : YOLODetector(enginePath, labelsPath, YOLOVersion::V7, dlaCore) {}
};

/// @brief YOLOv8 detector (forces standard postprocessing)
class YOLOv8Detector : public YOLODetector {
public:
    YOLOv8Detector(const std::string& enginePath, const std::string& labelsPath, int dlaCore = -1)
        : YOLODetector(enginePath, labelsPath, YOLOVersion::V8, dlaCore) {}
};

/// @brief YOLOv10 detector (forces V10 end-to-end postprocessing)
class YOLOv10Detector : public YOLODetector {
public:
    YOLOv10Detector(const std::string& enginePath, const std::string& labelsPath, int dlaCore = -1)
        : YOLODetector(enginePath, labelsPath, YOLOVersion::V10, dlaCore) {}
};

/// @brief YOLOv11 detector (forces standard postprocessing)
class YOLOv11Detector : public YOLODetector {
public:
    YOLOv11Detector(const std::string& enginePath, const std::string& labelsPath, int dlaCore = -1)
        : YOLODetector(enginePath, labelsPath, YOLOVersion::V11, dlaCore) {}
};

/// @brief YOLO-NAS detector (forces NAS postprocessing)
class YOLONASDetector : public YOLODetector {
public:
    YOLONASDetector(const std::string& enginePath, const std::string& labelsPath, int dlaCore = -1)
        : YOLODetector(enginePath, labelsPath, YOLOVersion::NAS, dlaCore) {}
};

/// @brief YOLOv26 detector (forces V26 end-to-end postprocessing)
class YOLO26Detector : public YOLODetector {
public:
    YOLO26Detector(const std::string& enginePath, const std::string& labelsPath, int dlaCore = -1)
        : YOLODetector(enginePath, labelsPath, YOLOVersion::V26, dlaCore) {}
};

// ============================================================================
// Factory Function
// ============================================================================

/// @brief Create a detector with explicit version selection
inline std::unique_ptr<YOLODetector> createDetector(const std::string& enginePath,
                                                    const std::string& labelsPath,
                                                    YOLOVersion version = YOLOVersion::Auto,
                                                    int dlaCore = -1) {
    switch (version) {
        case YOLOVersion::V7:
            return std::make_unique<YOLOv7Detector>(enginePath, labelsPath, dlaCore);
        case YOLOVersion::V8:
            return std::make_unique<YOLOv8Detector>(enginePath, labelsPath, dlaCore);
        case YOLOVersion::V10:
            return std::make_unique<YOLOv10Detector>(enginePath, labelsPath, dlaCore);
        case YOLOVersion::V11:
            return std::make_unique<YOLOv11Detector>(enginePath, labelsPath, dlaCore);
        case YOLOVersion::V26:
            return std::make_unique<YOLO26Detector>(enginePath, labelsPath, dlaCore);
        case YOLOVersion::NAS:
            return std::make_unique<YOLONASDetector>(enginePath, labelsPath, dlaCore);
        default:
            return std::make_unique<YOLODetector>(enginePath, labelsPath, YOLOVersion::Auto, dlaCore);
    }
}

} // namespace det
} // namespace yolos
