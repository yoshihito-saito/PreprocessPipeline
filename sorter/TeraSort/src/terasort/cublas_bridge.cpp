// Small C ABI around native cuBLAS. No Torch C++ ABI or CUDA toolkit dependency.
// The caller supplies the absolute cuBLAS DLL shipped with its Torch runtime.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <new>
#include <mutex>

using Status = int;
using Handle = void*;
using Create = Status (__stdcall*)(Handle*);
using Destroy = Status (__stdcall*)(Handle);
using SetStream = Status (__stdcall*)(Handle, void*);
using SetMath = Status (__stdcall*)(Handle, int);
using BlasVersion = Status (__stdcall*)(Handle, int*);
using Sgemm = Status (__stdcall*)(Handle, int, int, int, int, int, const float*,
                                 const float*, int, const float*, int,
                                 const float*, float*, int);
struct Context {
    HMODULE module = nullptr;
    Handle handle = nullptr;
    Destroy destroy = nullptr;
    SetStream stream = nullptr;
    Sgemm sgemm = nullptr;
    std::mutex mutex;
};

extern "C" __declspec(dllexport) int pp_cublas_create(
        const wchar_t* path, void** output, int* version) {
    if (!path || !output || !version) return -1;
    *output = nullptr;
    auto* ctx = new (std::nothrow) Context;
    if (!ctx) return -2;
    ctx->module = LoadLibraryExW(path, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!ctx->module) { delete ctx; return -3; }
    auto create = reinterpret_cast<Create>(GetProcAddress(ctx->module, "cublasCreate_v2"));
    ctx->destroy = reinterpret_cast<Destroy>(GetProcAddress(ctx->module, "cublasDestroy_v2"));
    ctx->stream = reinterpret_cast<SetStream>(GetProcAddress(ctx->module, "cublasSetStream_v2"));
    ctx->sgemm = reinterpret_cast<Sgemm>(GetProcAddress(ctx->module, "cublasSgemm_v2"));
    auto math = reinterpret_cast<SetMath>(GetProcAddress(ctx->module, "cublasSetMathMode"));
    auto get_version = reinterpret_cast<BlasVersion>(GetProcAddress(ctx->module, "cublasGetVersion_v2"));
    if (!create || !ctx->destroy || !ctx->stream || !ctx->sgemm || !math || !get_version) {
        FreeLibrary(ctx->module); delete ctx; return -4;
    }
    int status = create(&ctx->handle);
    // CUBLAS_DEFAULT_MATH: no explicit TF32/half-precision fast-math opt-in.
    if (!status) status = math(ctx->handle, 0);
    if (!status) status = get_version(ctx->handle, version);
    if (status) {
        if (ctx->handle) ctx->destroy(ctx->handle);
        FreeLibrary(ctx->module); delete ctx; return status;
    }
    *output = ctx;
    return 0;
}

extern "C" __declspec(dllexport) int pp_cublas_sgemm(
        void* context, void* stream, const float* weights, const float* signal,
        float* output, int units, int features, int samples) {
    if (!context || !weights || !signal || !output || units <= 0 || features <= 0 || samples <= 0)
        return -1;
    auto* ctx = static_cast<Context*>(context);
    std::lock_guard<std::mutex> guard(ctx->mutex);
    int status = ctx->stream(ctx->handle, stream);
    if (status) return status;
    const float alpha = 1.0f, beta = 0.0f;
    // Row-major output[U,T] = weights[U,K] * signal[K,T].
    // Interpret these buffers as column-major: output^T = signal^T * weights^T.
    return ctx->sgemm(ctx->handle, 0, 0, samples, units, features, &alpha,
                      signal, samples, weights, features, &beta, output, samples);
}

extern "C" __declspec(dllexport) int pp_cublas_destroy(void* context) {
    if (!context) return 0;
    auto* ctx = static_cast<Context*>(context);
    const int status = ctx->destroy(ctx->handle);
    FreeLibrary(ctx->module);
    delete ctx;
    return status;
}
