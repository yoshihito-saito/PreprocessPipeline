"""Fault-inject the real CUDA error/ownership helper without requiring a GPU."""

from pathlib import Path
import os
import shutil
import subprocess

import pytest


@pytest.fixture(scope="module")
def checked_cuda_harness(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    folder = tmp_path_factory.mktemp("checked-cuda")
    (folder / "gpu").mkdir()
    (folder / "mex.h").write_text("#pragma once\n")
    (folder / "gpu" / "mxGPUArray.h").write_text(
        "#pragma once\nstruct mxGPUArray {};\n"
        "void mxGPUDestroyGPUArray(const mxGPUArray *);\n"
    )
    (folder / "cuda_runtime.h").write_text(
        """#pragma once
#include <cstddef>
enum cudaError_t { cudaSuccess=0, cudaErrorMemoryAllocation=2,
                  cudaErrorNoKernelImageForDevice=209, cudaErrorLaunchFailure=719 };
cudaError_t cudaMalloc(void **, size_t);
cudaError_t cudaFree(void *);
cudaError_t cudaGetLastError();
cudaError_t cudaDeviceSynchronize();
const char *cudaGetErrorName(cudaError_t);
const char *cudaGetErrorString(cudaError_t);
"""
    )
    source = folder / "main.cpp"
    source.write_text(
        """#include "ks_cuda_checks.h"
#include <cassert>
#include <string>
int allocations=0, frees=0, destroyed=0;
cudaError_t mallocError=cudaSuccess, launchError=cudaSuccess, syncError=cudaSuccess;
cudaError_t cudaMalloc(void **p, size_t n) {
    if (mallocError != cudaSuccess) return mallocError;
    *p=std::malloc(n); ++allocations; return cudaSuccess;
}
cudaError_t cudaFree(void *p) { ++frees; std::free(p); return cudaSuccess; }
cudaError_t cudaGetLastError() { return launchError; }
cudaError_t cudaDeviceSynchronize() { return syncError; }
const char *cudaGetErrorName(cudaError_t s) {
    if (s==cudaErrorNoKernelImageForDevice) return "cudaErrorNoKernelImageForDevice";
    if (s==cudaErrorLaunchFailure) return "cudaErrorLaunchFailure";
    if (s==cudaErrorMemoryAllocation) return "cudaErrorMemoryAllocation";
    return "cudaSuccess";
}
const char *cudaGetErrorString(cudaError_t s) {
    if (s==cudaErrorNoKernelImageForDevice) return "no kernel image is available";
    if (s==cudaErrorLaunchFailure) return "unspecified launch failure";
    if (s==cudaErrorMemoryAllocation) return "out of memory";
    return "success";
}
void mxGPUDestroyGPUArray(const mxGPUArray *p) { ++destroyed; delete p; }
int main(int argc, char **argv) {
    assert(argc==2);
    const std::string scenario=argv[1];
    if (scenario=="success") {
        {
            KsCudaResources scope;
            float *data=nullptr;
            KS_CUDA(scope.allocate(&data,32));
            auto gpu=scope.gpu(new mxGPUArray);
            scope.host(static_cast<int *>(std::calloc(1,sizeof(int))));
            KS_KERNEL("Conv1D");
            KS_CUDA(scope.releaseDevice(data));
            scope.releaseGpu(gpu);
        }
        assert(allocations==1 && frees==1 && destroyed==1);
        return 0;
    }
    bool caught=false;
    try {
        KsCudaResources scope;
        float *data=nullptr;
        KS_CUDA(scope.allocate(&data,32));
        scope.gpu(new mxGPUArray);
        scope.host(static_cast<int *>(std::calloc(1,sizeof(int))));
        if (scenario=="allocation") {
            mallocError=cudaErrorMemoryAllocation;
            float *other=nullptr;
            KS_CUDA(scope.allocate(&other,32));
        } else {
            if (scenario=="launch") launchError=cudaErrorNoKernelImageForDevice;
            else { assert(scenario=="execution"); syncError=cudaErrorLaunchFailure; }
            KS_KERNEL("Conv1D");
        }
    } catch (const KsCudaError &error) {
        caught=true;
        const std::string message=error.what();
        if (scenario=="launch") {
            assert(message.find("Conv1D launch")!=std::string::npos);
            assert(message.find("cudaErrorNoKernelImageForDevice (209)")!=std::string::npos);
        } else if (scenario=="execution") {
            assert(message.find("Conv1D execution")!=std::string::npos);
            assert(message.find("cudaErrorLaunchFailure (719)")!=std::string::npos);
        } else {
            assert(message.find("scope.allocate")!=std::string::npos);
            assert(message.find("cudaErrorMemoryAllocation (2)")!=std::string::npos);
        }
    }
    assert(caught && allocations==1 && frees==1 && destroyed==1);
}
"""
    )
    cuda_sources = Path(__file__).resolve().parents[2] / "sorter" / "KiloSort1" / "CUDA"
    binary = folder / ("checked-cuda.exe" if os.name == "nt" else "checked-cuda")
    subprocess.run(
        [compiler, "-std=c++11", "-DKS_CUDA_DEBUG_SYNCHRONIZE=1", "-I", str(folder),
         "-I", str(cuda_sources), str(source), "-o", str(binary)],
        check=True, capture_output=True, text=True,
    )
    return binary


@pytest.mark.parametrize("scenario", ["success", "launch", "execution", "allocation"])
def test_checked_cuda_failure_and_cleanup(checked_cuda_harness, scenario):
    subprocess.run([str(checked_cuda_harness), scenario], check=True, capture_output=True, text=True)
