#pragma once

// ============================================================================
// YOLO Pose Estimation (TensorRT Backend)
// ============================================================================
// Human pose estimation using YOLO models with keypoint detection.
// Supports YOLOv8-pose, YOLOv11-pose, and YOLO26-pose models.
//
// Author: YOLOs-TRT Team
// ============================================================================

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>
#include <cmath>

#include "yolos/core/types.hpp"
#include "yolos/core/version.hpp"
#include "yolos/core/utils.hpp"
#include "yolos/core/preprocessing.hpp"
#include "yolos/core/nms.hpp"
#include "yolos/core/drawing.hpp"
#include "yolos/core/trt_session_base.hpp"

namespace yolos {
namespace pose {

// ============================================================================
// Pose Result Structure
// ============================================================================

/// @brief Pose estimation result containing bounding box, confidence, and keypoints
struct PoseResult {
    BoundingBox box;
    float conf{0.0f};
    int classId{0};
    std::vector<KeyPoint> keypoints;

    PoseResult() = default;
    PoseResult(const BoundingBox& box_, float conf_, int classId_, const std::vector<KeyPoint>& kpts)
        : box(box_), conf(conf_), classId(classId_), keypoints(kpts) {}
};

// ============================================================================
// YOLOPoseDetector Class
// ============================================================================

/// @brief YOLO pose estimation detector with keypoint detection
class YOLOPoseDetector : public TrtSessionBase {
public:
    /// @brief Constructor
    /// @param enginePath Path to the TensorRT engine file
    /// @param labelsPath Path to the class names file (optional for pose)
    /// @param dlaCore DLA core index (-1 = GPU, 0/1 = DLA on Jetson)
    YOLOPoseDetector(const std::string& enginePath,
                     const std::string& labelsPath = "",
                     int dlaCore = -1)
        : TrtSessionBase(enginePath, dlaCore) {

        if (!labelsPath.empty()) {
            classNames_ = utils::getClassNames(labelsPath);
        } else {
            classNames_ = {"person"};
        }
        classColors_ = drawing::generateColors(classNames_);
    }

    virtual ~YOLOPoseDetector() = default;

    /// @brief Run pose detection on an image
    std::vector<PoseResult> detect(const cv::Mat& image,
                                   float confThreshold = 0.4f,
                                   float iouThreshold = 0.5f) {
        // Full GPU pipeline
        inferGpu(image);
        return postprocess(image.size(), inputShape_, confThreshold, iouThreshold);
    }

    /// @brief Draw pose estimations on an image
    void drawPoses(cv::Mat& image,
                   const std::vector<PoseResult>& results,
                   int kptRadius = 4,
                   float kptThreshold = 0.5f,
                   int lineThickness = 2) const {
        for (const auto& pose : results) {
            cv::rectangle(image,
                         cv::Point(pose.box.x, pose.box.y),
                         cv::Point(pose.box.x + pose.box.width, pose.box.y + pose.box.height),
                         cv::Scalar(0, 255, 0), lineThickness);
            drawing::drawPoseSkeleton(image, pose.keypoints, getPoseSkeleton(),
                                     kptRadius, kptThreshold, lineThickness);
        }
    }

    /// @brief Draw only skeletons (no bounding boxes)
    void drawSkeletonsOnly(cv::Mat& image,
                           const std::vector<PoseResult>& results,
                           int kptRadius = 4,
                           float kptThreshold = 0.5f,
                           int lineThickness = 2) const {
        for (const auto& pose : results) {
            drawing::drawPoseSkeleton(image, pose.keypoints, getPoseSkeleton(),
                                     kptRadius, kptThreshold, lineThickness);
        }
    }

    [[nodiscard]] const std::vector<std::string>& getClassNames() const { return classNames_; }

    [[nodiscard]] static const std::vector<std::pair<int, int>>& getPoseSkeleton() {
        static const std::vector<std::pair<int, int>> skeleton = {
            {0, 1}, {0, 2}, {1, 3}, {2, 4},
            {3, 5}, {4, 6},
            {5, 7}, {7, 9}, {6, 8}, {8, 10},
            {5, 6}, {5, 11}, {6, 12}, {11, 12},
            {11, 13}, {13, 15}, {12, 14}, {14, 16}
        };
        return skeleton;
    }

protected:
    std::vector<std::string> classNames_;
    std::vector<cv::Scalar> classColors_;
    static constexpr int NUM_KEYPOINTS = 17;
    static constexpr int FEATURES_PER_KEYPOINT = 3;

    /// @brief Postprocess pose detection outputs
    std::vector<PoseResult> postprocess(const cv::Size& originalSize,
                                        const cv::Size& resizedShape,
                                        float confThreshold,
                                        float iouThreshold) {
        const float* rawOutput = getOutputData(0);
        const auto& outputShape = getOutputShape(0);

        const int expectedFeaturesV8 = 4 + 1 + NUM_KEYPOINTS * FEATURES_PER_KEYPOINT;  // 56
        const int expectedFeaturesV26 = 4 + 1 + 1 + NUM_KEYPOINTS * FEATURES_PER_KEYPOINT;  // 57

        if (outputShape.size() == 3 && outputShape[2] == expectedFeaturesV26) {
            return postprocessV26(originalSize, resizedShape, rawOutput, outputShape, confThreshold);
        } else if (outputShape.size() == 3 && outputShape[1] == expectedFeaturesV8) {
            return postprocessV8(originalSize, resizedShape, rawOutput, outputShape, confThreshold, iouThreshold);
        } else {
            std::cerr << "[ERROR] Unsupported pose model output shape: ["
                      << outputShape[0] << ", " << outputShape[1] << ", " << outputShape[2] << "]" << std::endl;
            return {};
        }
    }

    /// @brief Postprocess YOLOv8/v11 pose outputs (requires NMS)
    std::vector<PoseResult> postprocessV8(const cv::Size& originalSize,
                                          const cv::Size& resizedShape,
                                          const float* rawOutput,
                                          const std::vector<int64_t>& outputShape,
                                          float confThreshold,
                                          float iouThreshold) {
        std::vector<PoseResult> results;

        const size_t numDetections = outputShape[2];

        float scale, padX, padY;
        preprocessing::getScalePad(originalSize, resizedShape, scale, padX, padY);
        const float invScale = 1.0f / scale;

        std::vector<BoundingBox> boxes;
        std::vector<float> confidences;
        std::vector<std::vector<KeyPoint>> allKeypoints;
        boxes.reserve(64);
        confidences.reserve(64);
        allKeypoints.reserve(64);

        for (size_t d = 0; d < numDetections; ++d) {
            const float objConfidence = rawOutput[4 * numDetections + d];
            if (objConfidence < confThreshold) continue;

            const float cx = rawOutput[0 * numDetections + d];
            const float cy = rawOutput[1 * numDetections + d];
            const float w = rawOutput[2 * numDetections + d];
            const float h = rawOutput[3 * numDetections + d];

            BoundingBox box;
            box.x = utils::clamp(static_cast<int>((cx - w * 0.5f - padX) * invScale), 0, originalSize.width - 1);
            box.y = utils::clamp(static_cast<int>((cy - h * 0.5f - padY) * invScale), 0, originalSize.height - 1);
            box.width = utils::clamp(static_cast<int>(w * invScale), 1, originalSize.width - box.x);
            box.height = utils::clamp(static_cast<int>(h * invScale), 1, originalSize.height - box.y);

            std::vector<KeyPoint> keypoints;
            keypoints.reserve(NUM_KEYPOINTS);
            for (int k = 0; k < NUM_KEYPOINTS; ++k) {
                const int offset = 5 + k * FEATURES_PER_KEYPOINT;
                KeyPoint kpt;
                kpt.x = (rawOutput[offset * numDetections + d] - padX) * invScale;
                kpt.y = (rawOutput[(offset + 1) * numDetections + d] - padY) * invScale;
                const float rawConf = rawOutput[(offset + 2) * numDetections + d];
                kpt.confidence = 1.0f / (1.0f + std::exp(-rawConf));

                kpt.x = utils::clamp(kpt.x, 0.0f, static_cast<float>(originalSize.width - 1));
                kpt.y = utils::clamp(kpt.y, 0.0f, static_cast<float>(originalSize.height - 1));

                keypoints.push_back(kpt);
            }

            boxes.push_back(box);
            confidences.push_back(objConfidence);
            allKeypoints.push_back(std::move(keypoints));
        }

        if (boxes.empty()) return results;

        std::vector<int> indices;
        nms::NMSBoxes(boxes, confidences, confThreshold, iouThreshold, indices);

        results.reserve(indices.size());
        for (int idx : indices) {
            results.emplace_back(boxes[idx], confidences[idx], 0, allKeypoints[idx]);
        }

        return results;
    }

    /// @brief Postprocess YOLO26 pose outputs (end-to-end, NMS-free)
    std::vector<PoseResult> postprocessV26(const cv::Size& originalSize,
                                           const cv::Size& resizedShape,
                                           const float* rawOutput,
                                           const std::vector<int64_t>& outputShape,
                                           float confThreshold) {
        std::vector<PoseResult> results;

        const size_t numDetections = outputShape[1];
        const size_t numFeatures = outputShape[2];

        float scale, padX, padY;
        preprocessing::getScalePad(originalSize, resizedShape, scale, padX, padY);
        const float invScale = 1.0f / scale;

        for (size_t d = 0; d < numDetections; ++d) {
            const size_t base = d * numFeatures;

            const float x1 = rawOutput[base + 0];
            const float y1 = rawOutput[base + 1];
            const float x2 = rawOutput[base + 2];
            const float y2 = rawOutput[base + 3];
            const float conf = rawOutput[base + 4];

            if (conf < confThreshold) continue;

            BoundingBox box;
            box.x = utils::clamp(static_cast<int>((x1 - padX) * invScale), 0, originalSize.width - 1);
            box.y = utils::clamp(static_cast<int>((y1 - padY) * invScale), 0, originalSize.height - 1);
            const int x2_scaled = utils::clamp(static_cast<int>((x2 - padX) * invScale), 0, originalSize.width - 1);
            const int y2_scaled = utils::clamp(static_cast<int>((y2 - padY) * invScale), 0, originalSize.height - 1);
            box.width = std::max(1, x2_scaled - box.x);
            box.height = std::max(1, y2_scaled - box.y);

            std::vector<KeyPoint> keypoints;
            keypoints.reserve(NUM_KEYPOINTS);
            for (int k = 0; k < NUM_KEYPOINTS; ++k) {
                const size_t kptBase = base + 6 + k * FEATURES_PER_KEYPOINT;
                KeyPoint kpt;
                kpt.x = (rawOutput[kptBase + 0] - padX) * invScale;
                kpt.y = (rawOutput[kptBase + 1] - padY) * invScale;
                kpt.confidence = rawOutput[kptBase + 2];

                kpt.x = utils::clamp(kpt.x, 0.0f, static_cast<float>(originalSize.width - 1));
                kpt.y = utils::clamp(kpt.y, 0.0f, static_cast<float>(originalSize.height - 1));

                keypoints.push_back(kpt);
            }

            results.emplace_back(box, conf, 0, std::move(keypoints));
        }

        return results;
    }
};

} // namespace pose
} // namespace yolos
