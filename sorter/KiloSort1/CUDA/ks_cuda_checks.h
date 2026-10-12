#ifndef KS_CUDA_CHECKS_H
#define KS_CUDA_CHECKS_H

#include <cuda_runtime.h>
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <vector>

class KsCudaError : public std::runtime_error {
public:
    KsCudaError(cudaError_t status, const char *operation, const char *file, int line)
        : std::runtime_error(message(status, operation, file, line)) {}
private:
    static std::string message(cudaError_t status, const char *operation, const char *file, int line) {
        std::ostringstream out;
        out << operation << " failed: " << cudaGetErrorName(status) << " (" << int(status)
            << "), " << cudaGetErrorString(status) << ". " << file << ":" << line;
        return out.str();
    }
};

inline void ksCudaCheck(cudaError_t status, const char *operation, const char *file, int line) {
    if (status != cudaSuccess) throw KsCudaError(status, operation, file, line);
}

#define KS_CUDA(call) ksCudaCheck((call), #call, __FILE__, __LINE__)
#ifndef KS_CUDA_DEBUG_SYNCHRONIZE
#define KS_CUDA_DEBUG_SYNCHRONIZE 0
#endif
#define KS_KERNEL(name) do { \
    ksCudaCheck(cudaGetLastError(), name " launch", __FILE__, __LINE__); \
    if (KS_CUDA_DEBUG_SYNCHRONIZE) \
        ksCudaCheck(cudaDeviceSynchronize(), name " execution", __FILE__, __LINE__); \
} while (0)

// CUDA errors throw through this scope so allocations are released before the
// MEX entry point converts the exception to a MATLAB error. Never reset a GPU.
class KsCudaResources {
    std::vector<void *> device_;
    std::vector<void *> host_;
    std::vector<const mxGPUArray *> gpu_;
public:
    KsCudaResources() { device_.reserve(24); host_.reserve(4); gpu_.reserve(16); }
    KsCudaResources(const KsCudaResources &) = delete;
    KsCudaResources &operator=(const KsCudaResources &) = delete;
    ~KsCudaResources() {
        for (auto p : device_) cudaFree(p);
        for (auto p : gpu_) mxGPUDestroyGPUArray(p);
        for (auto p : host_) std::free(p);
    }
    template<class T> cudaError_t allocate(T **pointer, size_t bytes) {
        cudaError_t status = cudaMalloc(reinterpret_cast<void **>(pointer), bytes);
        if (status == cudaSuccess) device_.push_back(*pointer);
        return status;
    }
    cudaError_t releaseDevice(void *pointer) {
        cudaError_t status = cudaFree(pointer);
        if (status == cudaSuccess)
            device_.erase(std::remove(device_.begin(), device_.end(), pointer), device_.end());
        return status;
    }
    template<class T> T *gpu(T *array) { gpu_.push_back(array); return array; }
    void releaseGpu(const mxGPUArray *array) {
        mxGPUDestroyGPUArray(array);
        gpu_.erase(std::remove(gpu_.begin(), gpu_.end(), array), gpu_.end());
    }
    template<class T> T *host(T *pointer) {
        if (!pointer) throw std::bad_alloc();
        host_.push_back(pointer);
        return pointer;
    }
};
#endif
