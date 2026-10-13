#pragma once

// ============================================================================
// YOLO Oriented Bounding Box Detection — OBB (TensorRT Backend)
// ============================================================================
// Object detection with rotated/oriented bounding boxes for aerial imagery
// and other scenarios requiring rotation-aware detection.
// Supports YOLOv8-obb, YOLOv11-obb, and YOLO26-obb models.
//
// Author: YOLOs-TRT Team
// ============================================================================

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <string>
#include <memory>
#include <cfloat>
#include <cmath>

#include "yolos/core/types.hpp"
#include "yolos/core/version.hpp"
#include "yolos/core/utils.hpp"
#include "yolos/core/preprocessing.hpp"
#include "yolos/core/nms.hpp"
#include "yolos/core/drawing.hpp"
#include "yolos/core/trt_session_base.hpp"

namespace yolos {
namespace obb {

// ============================================================================
// OBB Detection Result Structure
// ============================================================================

/// @brief OBB detection result containing oriented bounding box, confidence, and class ID
struct OBBResult {
    OrientedBoundingBox box;
    float conf{0.0f};
    int classId{-1};

    OBBResult() = default;
    OBBResult(const OrientedBoundingBox& box_, float conf_, int classId_)
        : box(box_), conf(conf_), classId(classId_) {}
};

// ============================================================================
// YOLOOBBDetector Class
// ============================================================================

/// @brief YOLO oriented bounding box detector for rotated object detection
class YOLOOBBDetector : public TrtSessionBase {
public:
    /// @brief Constructor
    /// @param enginePath Path to the TensorRT engine file
    /// @param labelsPath Path to the class names file
    /// @param dlaCore DLA core index (-1 = GPU, 0/1 = DLA on Jetson)
    YOLOOBBDetector(const std::string& enginePath,
                    const std::string& labelsPath,
                    int dlaCore = -1)
        : TrtSessionBase(enginePath, dlaCore) {
        classNames_ = utils::getClassNames(labelsPath);
        classColors_ = drawing::generateColors(classNames_);
    }

    virtual ~YOLOOBBDetector() = default;

    /// @brief Run OBB detection on an image
    std::vector<OBBResult> detect(const cv::Mat& image,
                                  float confThreshold = 0.25f,
                                  float iouThreshold = 0.45f,
                                  int maxDet = 300) {
        // Full GPU pipeline
        inferGpu(image);
        return postprocess(image.size(), inputShape_, confThreshold, iouThreshold, maxDet);
    }

    /// @brief Draw OBB detections on an image
    void drawDetections(cv::Mat& image,
                        const std::vector<OBBResult>& results,
                        int thickness = 2) const {
        for (const auto& det : results) {
            if (det.classId >= 0 && static_cast<size_t>(det.classId) < classNames_.size()) {
                std::string label = classNames_[det.classId] + ": " +
                                   std::to_string(static_cast<int>(det.conf * 100)) + "%";
                const cv::Scalar& color = classColors_[det.classId % classColors_.size()];
                drawing::drawOrientedBoundingBox(image, det.box, label, color, thickness);
            }
        }
    }

    [[nodiscard]] const std::vector<std::string>& getClassNames() const { return classNames_; }
    [[nodiscard]] const std::vector<cv::Scalar>& getClassColors() const { return classColors_; }

protected:
    std::vector<std::string> classNames_;
    std::vector<cv::Scalar> classColors_;

    /// @brief Postprocess OBB detection outputs
    std::vector<OBBResult> postprocess(const cv::Size& originalSize,
                                       const cv::Size& resizedShape,
                                       float confThreshold,
                                       float iouThreshold,
                                       int maxDet) {
        const float* rawOutput = getOutputData(0);
        const auto& outputShape = getOutputShape(0);

        if (outputShape.size() == 3 && outputShape[2] == 7) {
            return postprocessV26(originalSize, resizedShape, rawOutput, outputShape, confThreshold, maxDet);
        } else {
            return postprocessV8(originalSize, resizedShape, rawOutput, outputShape, confThreshold, iouThreshold, maxDet);
        }
    }

    /// @brief Postprocess YOLOv8/v11 OBB outputs (requires NMS)
    std::vector<OBBResult> postprocessV8(const cv::Size& originalSize,
                                         const cv::Size& resizedShape,
                                         const float* rawOutput,
                                         const std::vector<int64_t>& outputShape,
                                         float confThreshold,
                                         float iouThreshold,
                                         int maxDet) {
        std::vector<OBBResult> results;

        const int numFeatures = static_cast<int>(outputShape[1]);
        const int numDetections = static_cast<int>(outputShape[2]);

        if (numDetections == 0) return results;

        const int numLabels = numFeatures - 5;
        if (numLabels <= 0) return results;

        float scale, padX, padY;
        preprocessing::getScalePad(originalSize, resizedShape, scale, padX, padY);
        const float invScale = 1.0f / scale;

        cv::Mat output = cv::Mat(numFeatures, numDetections, CV_32F, const_cast<float*>(rawOutput));
        output = output.t();

        std::vector<OrientedBoundingBox> letterboxBoxes;
        std::vector<float> confidences;
        std::vector<int> classIds;
        letterboxBoxes.reserve(256);
        confidences.reserve(256);
        classIds.reserve(256);

        for (int i = 0; i < numDetections; ++i) {
            const float* row = output.ptr<float>(i);

            float maxScore = row[4];
            int classId = 0;
            for (int j = 1; j < numLabels; ++j) {
                const float score = row[4 + j];
                if (score > maxScore) {
                    maxScore = score;
                    classId = j;
                }
            }

            if (maxScore <= confThreshold) continue;

            const float x = row[0];
            const float y = row[1];
            const float w = row[2];
            const float h = row[3];
            const float angle = row[4 + numLabels];

            letterboxBoxes.emplace_back(x, y, w, h, angle);
            confidences.push_back(maxScore);
            classIds.push_back(classId);
        }

        if (letterboxBoxes.empty()) return results;

        std::vector<int> keepIndices = nms::NMSRotatedBatched(letterboxBoxes, confidences, classIds, iouThreshold, maxDet);

        results.reserve(keepIndices.size());
        for (int idx : keepIndices) {
            const OrientedBoundingBox& lbBox = letterboxBoxes[idx];
            const float cx = (lbBox.x - padX) * invScale;
            const float cy = (lbBox.y - padY) * invScale;
            const float bw = lbBox.width * invScale;
            const float bh = lbBox.height * invScale;

            results.emplace_back(OrientedBoundingBox(cx, cy, bw, bh, lbBox.angle), confidences[idx], classIds[idx]);
        }

        return results;
    }

    /// @brief Postprocess YOLO26 OBB outputs (end-to-end, NMS-free)
    std::vector<OBBResult> postprocessV26(const cv::Size& originalSize,
                                          const cv::Size& resizedShape,
                                          const float* rawOutput,
                                          const std::vector<int64_t>& outputShape,
                                          float confThreshold,
                                          int maxDet) {
        std::vector<OBBResult> results;

        const size_t numDetections = outputShape[1];
        const size_t numFeatures = outputShape[2];

        float scale, padX, padY;
        preprocessing::getScalePad(originalSize, resizedShape, scale, padX, padY);
        const float invScale = 1.0f / scale;

        for (size_t d = 0; d < numDetections && static_cast<int>(results.size()) < maxDet; ++d) {
            const size_t base = d * numFeatures;

            const float x = rawOutput[base + 0];
            const float y = rawOutput[base + 1];
            const float w = rawOutput[base + 2];
            const float h = rawOutput[base + 3];
            const float conf = rawOutput[base + 4];
            const int classId = static_cast<int>(rawOutput[base + 5]);
            const float angle = rawOutput[base + 6];

            if (conf < confThreshold) continue;

            const float cx = (x - padX) * invScale;
            const float cy = (y - padY) * invScale;
            const float bw = w * invScale;
            const float bh = h * invScale;

            results.emplace_back(OrientedBoundingBox(cx, cy, bw, bh, angle), conf, classId);
        }

        return results;
    }
};

} // namespace obb
} // namespace yolos
