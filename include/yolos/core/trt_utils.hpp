#pragma once

// ============================================================================
// TensorRT Utilities
// ============================================================================
// CUDA error checking macros, TensorRT logger wrapper, and memory management
// helpers for production-grade TensorRT inference.
//
// Author: YOLOs-TRT Team
// ============================================================================

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace yolos {
namespace trt {

// ============================================================================
// CUDA Error Checking
// ============================================================================

/// @brief Check CUDA call and throw on failure with file/line info
inline void cudaCheck(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        std::ostringstream ss;
        ss << "CUDA error at " << file << ":" << line
           << " — " << cudaGetErrorName(code)
           << ": " << cudaGetErrorString(code);
        throw std::runtime_error(ss.str());
    }
}

#define CUDA_CHECK(call) ::yolos::trt::cudaCheck((call), __FILE__, __LINE__)

// ============================================================================
// TensorRT Logger
// ============================================================================

/// @brief TensorRT logger that maps severity to std::cerr / std::cout
class TrtLogger : public nvinfer1::ILogger {
public:
    explicit TrtLogger(Severity severity = Severity::kWARNING)
        : reportableSeverity_(severity) {}

    void log(Severity severity, const char* msg) noexcept override {
        if (severity > reportableSeverity_) return;

        const char* tag = "";
        switch (severity) {
            case Severity::kINTERNAL_ERROR: tag = "[FATAL] "; break;
            case Severity::kERROR:          tag = "[ERROR] "; break;
            case Severity::kWARNING:        tag = "[WARN]  "; break;
            case Severity::kINFO:           tag = "[INFO]  "; break;
            case Severity::kVERBOSE:        tag = "[DEBUG] "; break;
        }

        if (severity <= Severity::kWARNING) {
            std::cerr << "TensorRT " << tag << msg << std::endl;
        } else {
            std::cout << "TensorRT " << tag << msg << std::endl;
        }
    }

    void setSeverity(Severity severity) noexcept { reportableSeverity_ = severity; }

private:
    Severity reportableSeverity_;
};

/// @brief Get the singleton TRT logger
inline TrtLogger& getLogger() {
    static TrtLogger logger(nvinfer1::ILogger::Severity::kWARNING);
    return logger;
}

// ============================================================================
// TensorRT Object Deleters (for unique_ptr RAII)
// ============================================================================

struct TrtDeleter {
    template <typename T>
    void operator()(T* obj) const {
        delete obj;
    }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDeleter>;

// ============================================================================
// CUDA Memory Helpers
// ============================================================================

/// @brief RAII wrapper for pinned (page-locked) host memory
class PinnedBuffer {
public:
    PinnedBuffer() = default;

    explicit PinnedBuffer(size_t bytes) : bytes_(bytes) {
        if (bytes_ > 0) {
            CUDA_CHECK(cudaMallocHost(&ptr_, bytes_));
        }
    }

    ~PinnedBuffer() { free(); }

    // Move only
    PinnedBuffer(PinnedBuffer&& other) noexcept
        : ptr_(other.ptr_), bytes_(other.bytes_) {
        other.ptr_ = nullptr;
        other.bytes_ = 0;
    }

    PinnedBuffer& operator=(PinnedBuffer&& other) noexcept {
        if (this != &other) {
            free();
            ptr_ = other.ptr_;
            bytes_ = other.bytes_;
            other.ptr_ = nullptr;
            other.bytes_ = 0;
        }
        return *this;
    }

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    void resize(size_t bytes) {
        if (bytes > bytes_) {
            free();
            bytes_ = bytes;
            CUDA_CHECK(cudaMallocHost(&ptr_, bytes_));
        }
    }

    [[nodiscard]] void* data() noexcept { return ptr_; }
    [[nodiscard]] const void* data() const noexcept { return ptr_; }
    [[nodiscard]] size_t size() const noexcept { return bytes_; }

    template <typename T>
    [[nodiscard]] T* as() noexcept { return static_cast<T*>(ptr_); }

    template <typename T>
    [[nodiscard]] const T* as() const noexcept { return static_cast<const T*>(ptr_); }

private:
    void free() {
        if (ptr_) {
            cudaFreeHost(ptr_);
            ptr_ = nullptr;
            bytes_ = 0;
        }
    }

    void* ptr_{nullptr};
    size_t bytes_{0};
};

/// @brief RAII wrapper for device (GPU) memory
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(size_t bytes) : bytes_(bytes) {
        if (bytes_ > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, bytes_));
        }
    }

    ~DeviceBuffer() { free(); }

    // Move only
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(other.ptr_), bytes_(other.bytes_) {
        other.ptr_ = nullptr;
        other.bytes_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            free();
            ptr_ = other.ptr_;
            bytes_ = other.bytes_;
            other.ptr_ = nullptr;
            other.bytes_ = 0;
        }
        return *this;
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void resize(size_t bytes) {
        if (bytes > bytes_) {
            free();
            bytes_ = bytes;
            CUDA_CHECK(cudaMalloc(&ptr_, bytes_));
        }
    }

    [[nodiscard]] void* data() noexcept { return ptr_; }
    [[nodiscard]] const void* data() const noexcept { return ptr_; }
    [[nodiscard]] size_t size() const noexcept { return bytes_; }

private:
    void free() {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
            bytes_ = 0;
        }
    }

    void* ptr_{nullptr};
    size_t bytes_{0};
};

// ============================================================================
// CUDA Stream RAII Wrapper
// ============================================================================

class CudaStream {
public:
    CudaStream() { CUDA_CHECK(cudaStreamCreate(&stream_)); }

    ~CudaStream() {
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
    }

    CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) {
        other.stream_ = nullptr;
    }

    CudaStream& operator=(CudaStream&& other) noexcept {
        if (this != &other) {
            if (stream_) cudaStreamDestroy(stream_);
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    void synchronize() const { CUDA_CHECK(cudaStreamSynchronize(stream_)); }

    [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }
    [[nodiscard]] operator cudaStream_t() const noexcept { return stream_; }

private:
    cudaStream_t stream_{nullptr};
};

// ============================================================================
// CUDA Event RAII Wrapper (for profiling)
// ============================================================================

class CudaEvent {
public:
    CudaEvent() { CUDA_CHECK(cudaEventCreate(&event_)); }

    ~CudaEvent() {
        if (event_) cudaEventDestroy(event_);
    }

    CudaEvent(CudaEvent&& other) noexcept : event_(other.event_) {
        other.event_ = nullptr;
    }

    CudaEvent& operator=(CudaEvent&& other) noexcept {
        if (this != &other) {
            if (event_) cudaEventDestroy(event_);
            event_ = other.event_;
            other.event_ = nullptr;
        }
        return *this;
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    void record(cudaStream_t stream = nullptr) {
        CUDA_CHECK(cudaEventRecord(event_, stream));
    }

    void synchronize() { CUDA_CHECK(cudaEventSynchronize(event_)); }

    /// @brief Elapsed time in milliseconds between this event and a prior event
    static float elapsed(const CudaEvent& start, const CudaEvent& end) {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start.event_, end.event_));
        return ms;
    }

    [[nodiscard]] cudaEvent_t get() const noexcept { return event_; }

private:
    cudaEvent_t event_{nullptr};
};

// ============================================================================
// File I/O Helpers
// ============================================================================

/// @brief Read an entire binary file into a vector
inline std::vector<char> readFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    const auto size = file.tellg();
    if (size <= 0) {
        throw std::runtime_error("File is empty or unreadable: " + path);
    }
    std::vector<char> data(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    file.read(data.data(), size);
    return data;
}

// ============================================================================
// Data-type Utilities
// ============================================================================

/// @brief Bytes per element for a TensorRT data type
inline size_t dataTypeBytes(nvinfer1::DataType dtype) {
    switch (dtype) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF:  return 2;
        case nvinfer1::DataType::kINT8:  return 1;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kBOOL:  return 1;
        default:
            throw std::runtime_error("Unsupported TensorRT data type");
    }
}

/// @brief Compute total number of elements from a TensorRT Dims
inline int64_t volume(const nvinfer1::Dims& dims) {
    int64_t v = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        v *= dims.d[i];
    }
    return v;
}

} // namespace trt
} // namespace yolos
