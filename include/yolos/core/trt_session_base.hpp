#pragma once

// ============================================================================
// TensorRT Session Base — High-Performance Edition
// ============================================================================
// Zero-copy GPU preprocessing pipeline:
//   cv::Mat (host) → pinned staging → device uint8 → CUDA letterbox kernel
//   → TRT input buffer (device float NCHW) → enqueueV3 → D2H output
//
// Optimisations:
//   • Pinned staging buffer for truly async H2D of raw BGR pixels
//   • CUDA preprocessing kernel writes directly into TRT input buffer
//   • CUDA graph capture eliminates per-frame kernel launch overhead
//   • 10 warm-up iterations for stable TRT autotuner
//   • Single stream, minimal sync points
//
// Requires TensorRT 10.x+ (tensor-based API).
//
// Author: YOLOs-TRT Team
// ============================================================================

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <opencv2/core.hpp>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "yolos/core/trt_utils.hpp"
#include "yolos/core/cuda_preprocessing.hpp"
#include "yolos/core/version.hpp"
#include "yolos/core/utils.hpp"

namespace yolos {

// ============================================================================
// TrtSessionBase
// ============================================================================

class TrtSessionBase {
public:
    TrtSessionBase(const std::string& enginePath,
                   int dlaCore = -1,
                   int warmupRuns = 10)
    {
        initEngine(enginePath, dlaCore);
        allocGpuPreprocessBuffers();
        warmUp(warmupRuns);
        captureInferenceGraph();
    }

    virtual ~TrtSessionBase() {
        // Destroy CUDA graph resources before other members
        if (graphExec_) { cudaGraphExecDestroy(graphExec_); graphExec_ = nullptr; }
        if (graph_)     { cudaGraphDestroy(graph_); graph_ = nullptr; }
    }

    TrtSessionBase(const TrtSessionBase&) = delete;
    TrtSessionBase& operator=(const TrtSessionBase&) = delete;
    TrtSessionBase(TrtSessionBase&&) = default;
    TrtSessionBase& operator=(TrtSessionBase&&) = default;

    // ── Accessors ───────────────────────────────────────────────────────

    [[nodiscard]] cv::Size getInputShape() const noexcept { return inputShape_; }
    [[nodiscard]] size_t getNumInputNodes() const noexcept { return inputTensors_.size(); }
    [[nodiscard]] size_t getNumOutputNodes() const noexcept { return outputTensors_.size(); }
    [[nodiscard]] bool isDynamicInputShape() const noexcept { return isDynamicInputShape_; }
    [[nodiscard]] bool isDynamicBatchSize() const noexcept { return isDynamicBatchSize_; }
    [[nodiscard]] const std::string& getDevice() const noexcept {
        static const std::string dev = "gpu";
        return dev;
    }

protected:
    // ── Tensor Descriptor ───────────────────────────────────────────────

    struct TensorInfo {
        std::string name;
        nvinfer1::Dims dims;
        nvinfer1::DataType dtype{nvinfer1::DataType::kFLOAT};
        bool isInput{false};
        trt::DeviceBuffer deviceBuf;
        trt::PinnedBuffer hostBuf;
        size_t bytes{0};
        std::vector<int64_t> shape;
    };

    trt::TrtUniquePtr<nvinfer1::IRuntime>          runtime_{nullptr};
    trt::TrtUniquePtr<nvinfer1::ICudaEngine>       engine_{nullptr};
    trt::TrtUniquePtr<nvinfer1::IExecutionContext>  context_{nullptr};

    trt::CudaStream stream_;

    std::vector<TensorInfo> inputTensors_;
    std::vector<TensorInfo> outputTensors_;

    cv::Size inputShape_;
    bool isDynamicInputShape_{false};
    bool isDynamicBatchSize_{false};

    // ── GPU Preprocessing Staging ───────────────────────────────────────
    trt::PinnedBuffer pinnedSrc_;     // pinned host buffer for raw BGR image
    trt::DeviceBuffer deviceSrc_;     // device buffer for raw BGR image
    size_t maxSrcBytes_{0};           // current allocation capacity

    // ── CUDA Graph ──────────────────────────────────────────────────────
    cudaGraph_t graph_{nullptr};
    cudaGraphExec_t graphExec_{nullptr};
    bool graphCaptured_{false};

    // ── Scale/pad cache (set during GPU preprocess, read by postprocess) ──
    mutable float cachedScale_{1.0f};
    mutable float cachedPadX_{0.0f};
    mutable float cachedPadY_{0.0f};

    // ====================================================================
    // GPU-accelerated full pipeline: cv::Mat → detect
    // ====================================================================

    /// @brief Upload raw BGR image to GPU, run CUDA letterbox kernel that
    ///        writes directly into TRT input buffer, then run inference
    ///        and copy outputs to host. Entire preprocess is on GPU.
    ///
    /// @param image  Input BGR cv::Mat (any size)
    void inferGpu(const cv::Mat& image) {
        const int srcH = image.rows;
        const int srcW = image.cols;
        const size_t srcBytes = static_cast<size_t>(srcH) * srcW * 3;

        // Ensure staging buffers are big enough
        ensureSrcCapacity(srcBytes);

        // 1. Copy raw BGR pixels host→device via pinned staging (truly async)
        std::memcpy(pinnedSrc_.data(), image.data, srcBytes);
        CUDA_CHECK(cudaMemcpyAsync(
            deviceSrc_.data(), pinnedSrc_.data(), srcBytes,
            cudaMemcpyHostToDevice, stream_.get()));

        // 2. CUDA letterbox+BGR→RGB+normalize → directly into TRT input buffer
        float* trtInput = static_cast<float*>(inputTensors_[0].deviceBuf.data());
        cuda::letterboxPreprocess(
            static_cast<const uint8_t*>(deviceSrc_.data()),
            srcW, srcH,
            trtInput,
            inputShape_.width, inputShape_.height,
            stream_.get());

        // 3. Inference (CUDA graph if available, else enqueueV3)
        if (graphCaptured_ && graphExec_) {
            CUDA_CHECK(cudaGraphLaunch(graphExec_, stream_.get()));
        } else {
            context_->enqueueV3(stream_.get());
        }

        // 4. D2H copy of outputs
        for (auto& out : outputTensors_) {
            CUDA_CHECK(cudaMemcpyAsync(
                out.hostBuf.data(), out.deviceBuf.data(), out.bytes,
                cudaMemcpyDeviceToHost, stream_.get()));
        }

        // 5. Single sync point
        stream_.synchronize();

        // 6. Cache scale/pad for postprocessing (computed on host, negligible)
        computeScalePad(srcW, srcH);
    }

    // ====================================================================
    // Legacy CPU-blob path (kept for classification & fallback)
    // ====================================================================

    void infer(const float* blob, size_t count) {
        // Copy from pinned input host buffer for speed
        TensorInfo& inp = inputTensors_[0];
        const size_t bytes = count * sizeof(float);
        assert(bytes <= inp.bytes);

        // Copy blob into pinned buffer first, then async H2D
        std::memcpy(inp.hostBuf.data(), blob, bytes);
        CUDA_CHECK(cudaMemcpyAsync(
            inp.deviceBuf.data(), inp.hostBuf.data(), bytes,
            cudaMemcpyHostToDevice, stream_.get()));

        if (graphCaptured_ && graphExec_) {
            CUDA_CHECK(cudaGraphLaunch(graphExec_, stream_.get()));
        } else {
            context_->enqueueV3(stream_.get());
        }

        for (auto& out : outputTensors_) {
            CUDA_CHECK(cudaMemcpyAsync(
                out.hostBuf.data(), out.deviceBuf.data(), out.bytes,
                cudaMemcpyDeviceToHost, stream_.get()));
        }
        stream_.synchronize();
    }

    // ── Output Accessors ────────────────────────────────────────────────

    [[nodiscard]] const float* getOutputData(size_t idx = 0) const {
        assert(idx < outputTensors_.size());
        return outputTensors_[idx].hostBuf.as<float>();
    }

    [[nodiscard]] const std::vector<int64_t>& getOutputShape(size_t idx = 0) const {
        assert(idx < outputTensors_.size());
        return outputTensors_[idx].shape;
    }

    [[nodiscard]] size_t numOutputs() const noexcept {
        return outputTensors_.size();
    }

    // ── Cached scale/pad accessors ──────────────────────────────────────

    [[nodiscard]] float getCachedScale() const noexcept { return cachedScale_; }
    [[nodiscard]] float getCachedPadX()  const noexcept { return cachedPadX_; }
    [[nodiscard]] float getCachedPadY()  const noexcept { return cachedPadY_; }

private:
    // ── Engine Init ─────────────────────────────────────────────────────

    void initEngine(const std::string& enginePath, int dlaCore) {
        std::vector<char> engineData = trt::readFile(enginePath);
        std::cout << "[INFO] Engine file size: "
                  << (engineData.size() / (1024 * 1024)) << " MB" << std::endl;

        runtime_.reset(nvinfer1::createInferRuntime(trt::getLogger()));
        if (!runtime_) throw std::runtime_error("Failed to create TensorRT runtime");

        if (dlaCore >= 0) {
            if (runtime_->getNbDLACores() > 0) {
                runtime_->setDLACore(dlaCore);
                std::cout << "[INFO] Using DLA core " << dlaCore << std::endl;
            } else {
                std::cout << "[WARN] DLA not available, using GPU." << std::endl;
            }
        }

        engine_.reset(runtime_->deserializeCudaEngine(engineData.data(), engineData.size()));
        if (!engine_) throw std::runtime_error("Failed to deserialize engine: " + enginePath);

        context_.reset(engine_->createExecutionContext());
        if (!context_) throw std::runtime_error("Failed to create execution context");

        const int numIO = engine_->getNbIOTensors();
        for (int i = 0; i < numIO; ++i) {
            const char* name = engine_->getIOTensorName(i);
            nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(name);
            nvinfer1::Dims dims = engine_->getTensorShape(name);
            nvinfer1::DataType dtype = engine_->getTensorDataType(name);

            TensorInfo ti;
            ti.name = name;
            ti.dims = dims;
            ti.dtype = dtype;
            ti.isInput = (mode == nvinfer1::TensorIOMode::kINPUT);

            ti.shape.resize(dims.nbDims);
            bool hasDynamic = false;
            for (int d = 0; d < dims.nbDims; ++d) {
                ti.shape[d] = dims.d[d];
                if (dims.d[d] == -1) hasDynamic = true;
            }

            if (ti.isInput && hasDynamic) {
                nvinfer1::Dims optDims = dims;
                for (int d = 0; d < dims.nbDims; ++d) {
                    if (dims.d[d] == -1) {
                        if (d == 0) optDims.d[d] = 1;
                        else if (d == 1) optDims.d[d] = 3;
                        else optDims.d[d] = 640;
                    }
                }
                context_->setInputShape(name, optDims);
                for (int d = 0; d < optDims.nbDims; ++d) ti.shape[d] = optDims.d[d];
            }

            int64_t vol = 1;
            for (auto s : ti.shape) vol *= (s > 0 ? s : 1);
            ti.bytes = static_cast<size_t>(vol) * trt::dataTypeBytes(dtype);

            ti.deviceBuf = trt::DeviceBuffer(ti.bytes);
            ti.hostBuf   = trt::PinnedBuffer(ti.bytes);
            context_->setTensorAddress(name, ti.deviceBuf.data());

            if (ti.isInput) inputTensors_.push_back(std::move(ti));
            else            outputTensors_.push_back(std::move(ti));
        }

        if (inputTensors_.empty()) throw std::runtime_error("Engine has no inputs");
        if (outputTensors_.empty()) throw std::runtime_error("Engine has no outputs");

        // Resolve output shapes
        for (auto& out : outputTensors_) {
            nvinfer1::Dims outDims = context_->getTensorShape(out.name.c_str());
            out.shape.resize(outDims.nbDims);
            int64_t vol = 1;
            for (int d = 0; d < outDims.nbDims; ++d) {
                out.shape[d] = outDims.d[d];
                vol *= (outDims.d[d] > 0 ? outDims.d[d] : 1);
            }
            size_t needed = static_cast<size_t>(vol) * trt::dataTypeBytes(out.dtype);
            if (needed > out.bytes) {
                out.bytes = needed;
                out.deviceBuf = trt::DeviceBuffer(needed);
                out.hostBuf   = trt::PinnedBuffer(needed);
                context_->setTensorAddress(out.name.c_str(), out.deviceBuf.data());
            }
        }

        // Derive input image shape
        const auto& inShape = inputTensors_[0].shape;
        if (inShape.size() >= 4) {
            isDynamicBatchSize_ = (inputTensors_[0].dims.d[0] == -1);
            isDynamicInputShape_ = (inputTensors_[0].dims.d[2] == -1 || inputTensors_[0].dims.d[3] == -1);
            inputShape_ = cv::Size(static_cast<int>(inShape[3]), static_cast<int>(inShape[2]));
        } else {
            throw std::runtime_error("Expected 4D input [N,C,H,W]");
        }

        std::cout << "[INFO] Engine: " << enginePath << std::endl;
        std::cout << "[INFO] Input: " << inputShape_.width << "x" << inputShape_.height
                  << (isDynamicInputShape_ ? " (dynamic)" : "") << std::endl;
        std::cout << "[INFO] Outputs: " << outputTensors_.size();
        for (const auto& out : outputTensors_) {
            std::cout << "  " << out.name << "[";
            for (size_t d = 0; d < out.shape.size(); ++d) { if (d) std::cout << ","; std::cout << out.shape[d]; }
            std::cout << "]";
        }
        std::cout << std::endl;
    }

    // ── GPU Preprocessing Buffers ───────────────────────────────────────

    void allocGpuPreprocessBuffers() {
        // Pre-allocate for a common input size (1080p = 1920x1080x3 ≈ 6MB)
        constexpr size_t INITIAL_CAPACITY = 1920 * 1080 * 3;
        maxSrcBytes_ = INITIAL_CAPACITY;
        pinnedSrc_ = trt::PinnedBuffer(maxSrcBytes_);
        deviceSrc_ = trt::DeviceBuffer(maxSrcBytes_);
    }

    void ensureSrcCapacity(size_t bytes) {
        if (bytes > maxSrcBytes_) {
            maxSrcBytes_ = bytes + bytes / 4; // 25% headroom
            pinnedSrc_ = trt::PinnedBuffer(maxSrcBytes_);
            deviceSrc_ = trt::DeviceBuffer(maxSrcBytes_);
        }
    }

    // ── Scale/Pad Computation (matches CUDA kernel exactly) ─────────────

    void computeScalePad(int srcW, int srcH) const {
        const int dstW = inputShape_.width;
        const int dstH = inputShape_.height;
        cachedScale_ = std::min(
            static_cast<float>(dstH) / srcH,
            static_cast<float>(dstW) / srcW);
        const int newW = static_cast<int>(std::round(srcW * cachedScale_));
        const int newH = static_cast<int>(std::round(srcH * cachedScale_));
        cachedPadX_ = std::round((dstW - newW) / 2.0f - 0.1f);
        cachedPadY_ = std::round((dstH - newH) / 2.0f - 0.1f);
    }

    // ── CUDA Graph Capture ──────────────────────────────────────────────
    // Captures enqueueV3 into a CUDA graph so subsequent frames replay
    // the entire inference with a single cudaGraphLaunch — eliminating
    // per-kernel dispatch overhead (~0.1-0.3ms saved per frame).

    void captureInferenceGraph() {
        // Only capture for fixed-shape, non-dynamic models
        if (isDynamicInputShape_ || isDynamicBatchSize_) {
            std::cout << "[INFO] Dynamic shapes — CUDA graph disabled." << std::endl;
            return;
        }

        CUDA_CHECK(cudaStreamBeginCapture(stream_.get(), cudaStreamCaptureModeGlobal));
        bool ok = context_->enqueueV3(stream_.get());
        CUDA_CHECK(cudaStreamEndCapture(stream_.get(), &graph_));

        if (ok && graph_) {
            CUDA_CHECK(cudaGraphInstantiate(&graphExec_, graph_, 0));
            graphCaptured_ = true;
            std::cout << "[INFO] CUDA graph captured — inference dispatch optimised." << std::endl;
        } else {
            std::cout << "[WARN] CUDA graph capture failed — using enqueueV3 fallback." << std::endl;
            if (graph_) { cudaGraphDestroy(graph_); graph_ = nullptr; }
        }
    }

    // ── Warm-up ─────────────────────────────────────────────────────────

    void warmUp(int runs) {
        if (runs <= 0) return;
        std::cout << "[INFO] Warming up (" << runs << " iterations)..." << std::endl;

        CUDA_CHECK(cudaMemsetAsync(
            inputTensors_[0].deviceBuf.data(), 0,
            inputTensors_[0].bytes, stream_.get()));

        for (int i = 0; i < runs; ++i) {
            context_->enqueueV3(stream_.get());
        }
        for (auto& out : outputTensors_) {
            CUDA_CHECK(cudaMemcpyAsync(
                out.hostBuf.data(), out.deviceBuf.data(), out.bytes,
                cudaMemcpyDeviceToHost, stream_.get()));
        }
        stream_.synchronize();
        std::cout << "[INFO] Warm-up complete." << std::endl;
    }
};

} // namespace yolos
