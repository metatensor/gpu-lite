// gpulite: singleton classes dynamically loading the CUDA runtime, CUDA driver
// and NVRTC libraries, and macros to check the result of calls to them
#ifndef GPULITE_CUDA_LIB_HPP
#define GPULITE_CUDA_LIB_HPP

#include "./cuda_api.hpp"
#include "./dlopen.hpp"

namespace gpulite {

/*
This class allows us to dynamically load the CUDA runtime and reference the functions contained
within the libcudart.so library (see CUDA Runtime API:
https://docs.nvidia.com/cuda/cuda-runtime-api/index.html).
*/
class CUDART {
  public:
    static CUDART& instance() {
        static CUDART instance;
        return instance;
    }

    static bool loaded() { return instance().cudartHandle != nullptr; }

    using cudaGetDeviceCount_t = cudaError_t (*)(int*);
    using cudaGetDevice_t = cudaError_t (*)(int*);
    using cudaSetDevice_t = cudaError_t (*)(int);
    using cudaMalloc_t = cudaError_t (*)(void**, size_t);
    using cudaMemcpy_t = cudaError_t (*)(void*, const void*, size_t, cudaMemcpyKind);
    using cudaMemset_t = cudaError_t (*)(void*, int, size_t);
    using cudaGetErrorName_t = const char* (*)(cudaError_t);
    using cudaGetErrorString_t = const char* (*)(cudaError_t);
    using cudaDeviceSynchronize_t = cudaError_t (*)(void);
    using cudaPointerGetAttributes_t = cudaError_t (*)(cudaPointerAttributes*, const void*);
    using cudaFree_t = cudaError_t (*)(void*);
    using cudaRuntimeGetVersion_t = cudaError_t (*)(int*);
    using cudaStreamCreate_t = cudaError_t (*)(cudaStream_t*);
    using cudaStreamDestroy_t = cudaError_t (*)(cudaStream_t);
    using cudaStreamSynchronize_t = cudaError_t (*)(cudaStream_t);
    using cudaHostAlloc_t = cudaError_t (*)(void**, size_t, unsigned int);
    using cudaFreeHost_t = cudaError_t (*)(void*);
    using cudaHostGetDevicePointer_t = cudaError_t (*)(void**, void*, unsigned int);
    using cudaMemcpyAsync_t = cudaError_t (*)(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t);
    using cudaMemsetAsync_t = cudaError_t (*)(void*, int, size_t, cudaStream_t);
    using cudaMallocAsync_t = cudaError_t (*)(void**, size_t, cudaStream_t);
    using cudaFreeAsync_t = cudaError_t (*)(void*, cudaStream_t);
    using cudaEventCreate_t = cudaError_t (*)(cudaEvent_t*);
    using cudaEventCreateWithFlags_t = cudaError_t (*)(cudaEvent_t*, unsigned int);
    using cudaEventDestroy_t = cudaError_t (*)(cudaEvent_t);
    using cudaEventRecord_t = cudaError_t (*)(cudaEvent_t, cudaStream_t);
    using cudaEventSynchronize_t = cudaError_t (*)(cudaEvent_t);
    using cudaEventQuery_t = cudaError_t (*)(cudaEvent_t);
    using cudaEventElapsedTime_t = cudaError_t (*)(float*, cudaEvent_t, cudaEvent_t);
    using cudaStreamWaitEvent_t = cudaError_t (*)(cudaStream_t, cudaEvent_t, unsigned int);

    cudaGetDeviceCount_t cudaGetDeviceCount;
    cudaGetDevice_t cudaGetDevice;
    cudaSetDevice_t cudaSetDevice;
    cudaMalloc_t cudaMalloc;
    cudaMemset_t cudaMemset;
    cudaMemcpy_t cudaMemcpy;
    cudaGetErrorName_t cudaGetErrorName;
    cudaGetErrorString_t cudaGetErrorString;
    cudaDeviceSynchronize_t cudaDeviceSynchronize;
    cudaPointerGetAttributes_t cudaPointerGetAttributes;
    cudaFree_t cudaFree;
    cudaRuntimeGetVersion_t cudaRuntimeGetVersion;
    cudaStreamCreate_t cudaStreamCreate;
    cudaStreamDestroy_t cudaStreamDestroy;
    cudaStreamSynchronize_t cudaStreamSynchronize;
    cudaHostAlloc_t cudaHostAlloc;
    cudaFreeHost_t cudaFreeHost;
    cudaHostGetDevicePointer_t cudaHostGetDevicePointer;
    cudaMemcpyAsync_t cudaMemcpyAsync;
    cudaMemsetAsync_t cudaMemsetAsync;
    cudaMallocAsync_t cudaMallocAsync;
    cudaFreeAsync_t cudaFreeAsync;
    cudaEventCreate_t cudaEventCreate;
    cudaEventCreateWithFlags_t cudaEventCreateWithFlags;
    cudaEventDestroy_t cudaEventDestroy;
    cudaEventRecord_t cudaEventRecord;
    cudaEventSynchronize_t cudaEventSynchronize;
    cudaEventQuery_t cudaEventQuery;
    cudaEventElapsedTime_t cudaEventElapsedTime;
    cudaStreamWaitEvent_t cudaStreamWaitEvent;

    CUDART() {
#if defined(__linux__) || defined(__APPLE__)
        ownedHandle = true;
        // First try to find an already loaded libcudart to avoid loading
        // multiple versions in the same process
        cudartHandle = details::findLoadedLibrary("libcudart.");

        // otherwise, try multiple candidate names to maximize compatibility
        // with different CUDA versions
        if (!cudartHandle) {
            static const char* CANDIDATES[] = {
                "libcudart.so",
                "libcudart.so.11",
                "libcudart.so.12",
                "libcudart.so.13",
                "libcudart.so.14",
                "libcudart.so.15",
            };
            for (auto* candidate: CANDIDATES) {
                cudartHandle = dlopen(candidate, RTLD_NOW);
                if (cudartHandle) {
                    break;
                }
            }
        }
#elif defined(_WIN32)
        // First try to find an already loaded cudart.dll to avoid loading
        // multiple versions in the same process
        cudartHandle = details::findLoadedLibrary(L"cudart64");

        // then look into known path and pick the most recent version if
        // multiple are found (e.g. cudart64_90.dll, cudart64_120.dll, etc.)
        if (cudartHandle != nullptr) {
            ownedHandle = false;
        } else {
            auto dllPathOpt = details::FindBestCudaDll(L"cudart64");
            if (dllPathOpt) {
                auto dllPath = *dllPathOpt;
                auto dir = dllPath.parent_path();
                // add the directory containing the DLL to the search path
                SetDllDirectoryW(dir.c_str());

                cudartHandle = LoadLibraryExW(
                    dllPath.c_str(),
                    nullptr,
                    LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
                    LOAD_LIBRARY_SEARCH_DEFAULT_DIRS |
                    LOAD_LIBRARY_SEARCH_USER_DIRS
                );
                ownedHandle = true;
            }
        }
#else
#error "Platform not supported"
#endif
        if (cudartHandle) {
            // load cudart function pointers using template
            cudaGetDeviceCount = details::loadSymbol<cudaGetDeviceCount_t>(cudartHandle, "cudaGetDeviceCount");
            cudaGetDevice = details::loadSymbol<cudaGetDevice_t>(cudartHandle, "cudaGetDevice");
            cudaSetDevice = details::loadSymbol<cudaSetDevice_t>(cudartHandle, "cudaSetDevice");
            cudaMalloc = details::loadSymbol<cudaMalloc_t>(cudartHandle, "cudaMalloc");
            cudaMemset = details::loadSymbol<cudaMemset_t>(cudartHandle, "cudaMemset");
            cudaMemcpy = details::loadSymbol<cudaMemcpy_t>(cudartHandle, "cudaMemcpy");
            cudaGetErrorName = details::loadSymbol<cudaGetErrorName_t>(cudartHandle, "cudaGetErrorName");
            cudaGetErrorString = details::loadSymbol<cudaGetErrorString_t>(cudartHandle, "cudaGetErrorString");
            cudaDeviceSynchronize = details::loadSymbol<cudaDeviceSynchronize_t>(cudartHandle, "cudaDeviceSynchronize");
            cudaPointerGetAttributes = details::loadSymbol<cudaPointerGetAttributes_t>(cudartHandle, "cudaPointerGetAttributes");
            cudaFree = details::loadSymbol<cudaFree_t>(cudartHandle, "cudaFree");
            cudaRuntimeGetVersion = details::loadSymbol<cudaRuntimeGetVersion_t>(cudartHandle, "cudaRuntimeGetVersion");
            cudaStreamCreate = details::loadSymbol<cudaStreamCreate_t>(cudartHandle, "cudaStreamCreate");
            cudaStreamDestroy = details::loadSymbol<cudaStreamDestroy_t>(cudartHandle, "cudaStreamDestroy");
            cudaStreamSynchronize = details::loadSymbol<cudaStreamSynchronize_t>(cudartHandle, "cudaStreamSynchronize");
            cudaHostAlloc = details::loadSymbol<cudaHostAlloc_t>(cudartHandle, "cudaHostAlloc");
            cudaFreeHost = details::loadSymbol<cudaFreeHost_t>(cudartHandle, "cudaFreeHost");
            cudaHostGetDevicePointer = details::loadSymbol<cudaHostGetDevicePointer_t>(cudartHandle, "cudaHostGetDevicePointer");
            cudaMemcpyAsync = details::loadSymbol<cudaMemcpyAsync_t>(cudartHandle, "cudaMemcpyAsync");
            cudaMemsetAsync = details::loadSymbol<cudaMemsetAsync_t>(cudartHandle, "cudaMemsetAsync");
            cudaMallocAsync = details::loadSymbol<cudaMallocAsync_t>(cudartHandle, "cudaMallocAsync");
            cudaFreeAsync = details::loadSymbol<cudaFreeAsync_t>(cudartHandle, "cudaFreeAsync");
            cudaEventCreate = details::loadSymbol<cudaEventCreate_t>(cudartHandle, "cudaEventCreate");
            cudaEventCreateWithFlags = details::loadSymbol<cudaEventCreateWithFlags_t>(cudartHandle, "cudaEventCreateWithFlags");
            cudaEventDestroy = details::loadSymbol<cudaEventDestroy_t>(cudartHandle, "cudaEventDestroy");
            cudaEventRecord = details::loadSymbol<cudaEventRecord_t>(cudartHandle, "cudaEventRecord");
            cudaEventSynchronize = details::loadSymbol<cudaEventSynchronize_t>(cudartHandle, "cudaEventSynchronize");
            cudaEventQuery = details::loadSymbol<cudaEventQuery_t>(cudartHandle, "cudaEventQuery");
            cudaEventElapsedTime = details::loadSymbol<cudaEventElapsedTime_t>(cudartHandle, "cudaEventElapsedTime");
            cudaStreamWaitEvent = details::loadSymbol<cudaStreamWaitEvent_t>(cudartHandle, "cudaStreamWaitEvent");
        }
    }

    ~CUDART() {
#if defined(__linux__) || defined(__APPLE__)
        if (cudartHandle) {
            dlclose(cudartHandle);
        }
#elif defined(_WIN32)
        if (cudartHandle && ownedHandle) {
            FreeLibrary(static_cast<HMODULE>(cudartHandle));
        }
#else
#error "Platform not supported"
#endif
    }

    // Prevent copying
    CUDART(const CUDART&) = delete;
    CUDART& operator=(const CUDART&) = delete;

    void* cudartHandle = nullptr;
    bool ownedHandle = false;
};

/*
This class allows us to dynamically load the CUDA Driver and reference the functions contained
within the libcuda.so library (CUDA Driver API:
https://docs.nvidia.com/cuda/cuda-driver-api/index.html).
*/
class CUDADriver {

  public:
    static CUDADriver& instance() {
        static CUDADriver instance;
        return instance;
    }

    static bool loaded() { return instance().cudaHandle != nullptr; }

    using cuInit_t = CUresult (*)(unsigned int);
    using cuDeviceGetCount_t = CUresult (*)(int*);
    using cuDevicePrimaryCtxRetain_t = CUresult (*)(CUcontext*, CUdevice);
    using cuDevicePrimaryCtxRelease_t = CUresult (*)(CUdevice);
    using cuCtxCreate_t = CUresult (*)(CUcontext*, unsigned int, CUdevice);
    using cuCtxDestroy_t = CUresult (*)(CUcontext);
    using cuCtxGetCurrent_t = CUresult (*)(CUcontext*);
    using cuCtxSetCurrent_t = CUresult (*)(CUcontext);
    using cuModuleLoadDataEx_t = CUresult (*)(CUmodule*, const void*, unsigned int, CUjit_option*, void**);
    using cuModuleGetFunction_t = CUresult (*)(CUfunction*, CUmodule, const char*);
    using cuFuncSetAttribute_t = CUresult (*)(CUfunction, CUfunction_attribute, int);
    using cuFuncGetAttribute_t = CUresult (*)(int*, CUfunction_attribute, CUfunction);
    using cuCtxGetDevice_t = CUresult (*)(CUdevice*);
    using cuDeviceGetAttribute_t = CUresult (*)(int*, CUdevice_attribute, CUdevice);
    using cuDeviceGetName_t = CUresult (*)(char*, int, CUdevice);
    using cuDeviceTotalMem_t = CUresult (*)(unsigned int*, CUdevice);
    using cuStreamCreate_t = CUresult (*)(CUstream*, unsigned int);
    using cuStreamDestroy_t = CUresult (*)(CUstream);
    using cuCtxSynchronize_t = CUresult (*)(void);
    using cuGetErrorName_t = CUresult (*)(CUresult, const char**);
    using cuCtxPushCurrent_t = CUresult (*)(CUcontext);
    using cuPointerGetAttribute_t = CUresult (*)(void*, CUpointer_attribute, CUdeviceptr);
    using cuCtxCreate_v2_t = CUresult (*)(CUcontext*, unsigned int, CUdevice);
    using cuCtxDestroy_v2_t = CUresult (*)(CUcontext);
    using cuCtxPushCurrent_v2_t = CUresult (*)(CUcontext);
    using cuStreamDestroy_v2_t = CUresult (*)(CUstream);
    using cuDevicePrimaryCtxRelease_v2_t = CUresult (*)(CUdevice);
    using cuDeviceTotalMem_v2_t = CUresult (*)(size_t*, CUdevice);
    using cuMemsetD8Async_t = CUresult (*)(CUdeviceptr, unsigned char, size_t, CUstream);
    using cuMemsetD16Async_t = CUresult (*)(CUdeviceptr, unsigned short, size_t, CUstream);
    using cuMemsetD32Async_t = CUresult (*)(CUdeviceptr, unsigned int, size_t, CUstream);
    using cuLaunchKernel_t = CUresult (*)(
        CUfunction,
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int,
        size_t,
        CUstream,
        void**,
        void*
    );

    cuInit_t cuInit;
    cuDeviceGetCount_t cuDeviceGetCount;
    cuCtxCreate_t cuCtxCreate;
    cuCtxDestroy_t cuCtxDestroy;
    cuDevicePrimaryCtxRetain_t cuDevicePrimaryCtxRetain;
    cuDevicePrimaryCtxRelease_t cuDevicePrimaryCtxRelease;
    cuCtxGetCurrent_t cuCtxGetCurrent;
    cuCtxSetCurrent_t cuCtxSetCurrent;
    cuModuleLoadDataEx_t cuModuleLoadDataEx;
    cuModuleGetFunction_t cuModuleGetFunction;
    cuFuncSetAttribute_t cuFuncSetAttribute;
    cuFuncGetAttribute_t cuFuncGetAttribute;
    cuCtxGetDevice_t cuCtxGetDevice;
    cuDeviceGetAttribute_t cuDeviceGetAttribute;
    cuDeviceGetName_t cuDeviceGetName;
    cuDeviceTotalMem_t cuDeviceTotalMem;
    cuStreamCreate_t cuStreamCreate;
    cuStreamDestroy_t cuStreamDestroy;
    cuGetErrorName_t cuGetErrorName;
    cuCtxSynchronize_t cuCtxSynchronize;
    cuCtxCreate_v2_t cuCtxCreate_v2;
    cuCtxDestroy_v2_t cuCtxDestroy_v2;
    cuCtxPushCurrent_v2_t cuCtxPushCurrent_v2;
    cuStreamDestroy_v2_t cuStreamDestroy_v2;
    cuDevicePrimaryCtxRelease_v2_t cuDevicePrimaryCtxRelease_v2;
    cuDeviceTotalMem_v2_t cuDeviceTotalMem_v2;
    cuMemsetD8Async_t cuMemsetD8Async;
    cuMemsetD16Async_t cuMemsetD16Async;
    cuMemsetD32Async_t cuMemsetD32Async;
    cuCtxPushCurrent_t cuCtxPushCurrent;
    cuPointerGetAttribute_t cuPointerGetAttribute;
    cuLaunchKernel_t cuLaunchKernel;

    CUDADriver() {
#if defined(__linux__) || defined(__APPLE__)
        ownedHandle = true;
        // check if libcuda is already loaded to avoid loading multiple versions
        // in the same process
        cudaHandle = details::findLoadedLibrary("libcuda.");

        if (cudaHandle == nullptr) {
            cudaHandle = dlopen("libcuda.so", RTLD_NOW);
        }
#elif defined(_WIN32)
        cudaHandle = details::findLoadedLibrary(L"nvcuda.dll");
        if (cudaHandle) {
            ownedHandle = false;
        } else {
            cudaHandle = LoadLibraryA("nvcuda.dll");
            ownedHandle = true;
        }
#else
#error "Platform not supported"
#endif
        if (cudaHandle) {
            // Load CUDA driver function pointers using template
            cuInit = details::loadSymbol<cuInit_t>(cudaHandle, "cuInit");
            cuDeviceGetCount = details::loadSymbol<cuDeviceGetCount_t>(cudaHandle, "cuDeviceGetCount");
            cuCtxCreate = details::loadSymbol<cuCtxCreate_t>(cudaHandle, "cuCtxCreate");
            cuCtxDestroy = details::loadSymbol<cuCtxDestroy_t>(cudaHandle, "cuCtxDestroy");
            cuDevicePrimaryCtxRetain = details::loadSymbol<cuDevicePrimaryCtxRetain_t>(cudaHandle, "cuDevicePrimaryCtxRetain");
            cuDevicePrimaryCtxRelease = details::loadSymbol<cuDevicePrimaryCtxRelease_t>(cudaHandle, "cuDevicePrimaryCtxRelease");
            cuCtxGetCurrent = details::loadSymbol<cuCtxGetCurrent_t>(cudaHandle, "cuCtxGetCurrent");
            cuCtxSetCurrent = details::loadSymbol<cuCtxSetCurrent_t>(cudaHandle, "cuCtxSetCurrent");
            cuModuleLoadDataEx = details::loadSymbol<cuModuleLoadDataEx_t>(cudaHandle, "cuModuleLoadDataEx");
            cuModuleGetFunction = details::loadSymbol<cuModuleGetFunction_t>(cudaHandle, "cuModuleGetFunction");
            cuFuncSetAttribute = details::loadSymbol<cuFuncSetAttribute_t>(cudaHandle, "cuFuncSetAttribute");
            cuFuncGetAttribute = details::loadSymbol<cuFuncGetAttribute_t>(cudaHandle, "cuFuncGetAttribute");
            cuCtxGetDevice = details::loadSymbol<cuCtxGetDevice_t>(cudaHandle, "cuCtxGetDevice");
            cuDeviceGetAttribute = details::loadSymbol<cuDeviceGetAttribute_t>(cudaHandle, "cuDeviceGetAttribute");
            cuDeviceGetName = details::loadSymbol<cuDeviceGetName_t>(cudaHandle, "cuDeviceGetName");
            cuDeviceTotalMem = details::loadSymbol<cuDeviceTotalMem_t>(cudaHandle, "cuDeviceTotalMem");
            cuStreamCreate = details::loadSymbol<cuStreamCreate_t>(cudaHandle, "cuStreamCreate");
            cuStreamDestroy = details::loadSymbol<cuStreamDestroy_t>(cudaHandle, "cuStreamDestroy");
            cuCtxSynchronize = details::loadSymbol<cuCtxSynchronize_t>(cudaHandle, "cuCtxSynchronize");
            cuGetErrorName = details::loadSymbol<cuGetErrorName_t>(cudaHandle, "cuGetErrorName");
            cuCtxPushCurrent = details::loadSymbol<cuCtxPushCurrent_t>(cudaHandle, "cuCtxPushCurrent");
            cuPointerGetAttribute = details::loadSymbol<cuPointerGetAttribute_t>(cudaHandle, "cuPointerGetAttribute");
            cuLaunchKernel = details::loadSymbol<cuLaunchKernel_t>(cudaHandle, "cuLaunchKernel");
            cuCtxCreate_v2 = details::loadSymbol<cuCtxCreate_v2_t>(cudaHandle, "cuCtxCreate_v2");
            cuCtxDestroy_v2 = details::loadSymbol<cuCtxDestroy_v2_t>(cudaHandle, "cuCtxDestroy_v2");
            cuCtxPushCurrent_v2 = details::loadSymbol<cuCtxPushCurrent_v2_t>(cudaHandle, "cuCtxPushCurrent_v2");
            cuStreamDestroy_v2 = details::loadSymbol<cuStreamDestroy_v2_t>(cudaHandle, "cuStreamDestroy_v2");
            cuDevicePrimaryCtxRelease_v2 = details::loadSymbol<cuDevicePrimaryCtxRelease_v2_t>(cudaHandle, "cuDevicePrimaryCtxRelease_v2");
            cuDeviceTotalMem_v2 = details::loadSymbol<cuDeviceTotalMem_v2_t>(cudaHandle, "cuDeviceTotalMem_v2");
            cuMemsetD8Async = details::loadSymbol<cuMemsetD8Async_t>(cudaHandle, "cuMemsetD8Async");
            cuMemsetD16Async = details::loadSymbol<cuMemsetD16Async_t>(cudaHandle, "cuMemsetD16Async");
            cuMemsetD32Async = details::loadSymbol<cuMemsetD32Async_t>(cudaHandle, "cuMemsetD32Async");
        }
    }

    ~CUDADriver() {
#if defined(__linux__) || defined(__APPLE__)
        if (cudaHandle) {
            dlclose(cudaHandle);
        }
#elif defined(_WIN32)
        if (cudaHandle && ownedHandle) {
            FreeLibrary(static_cast<HMODULE>(cudaHandle));
        }
#else
#error "Platform not supported"
#endif
    }

    // Prevent copying
    CUDADriver(const CUDADriver&) = delete;
    CUDADriver& operator=(const CUDADriver&) = delete;

    void* cudaHandle = nullptr;
    bool ownedHandle = false;
};

/*
This class allows us to dynamically load NVRTC and reference the functions contained within the
libnvrtc.so library (see NVRTC API: https://docs.nvidia.com/cuda/nvrtc/index.html).
*/
class NVRTC {

  public:
    static NVRTC& instance() {
        static NVRTC instance;
        return instance;
    }

    static bool loaded() { return instance().nvrtcHandle != nullptr; }

    using nvrtcCreateProgram_t = nvrtcResult (*)(nvrtcProgram*, const char*, const char*, int, const char*[], const char*[]);
    using nvrtcCompileProgram_t = nvrtcResult (*)(nvrtcProgram, int, const char*[]);
    using nvrtcGetPTX_t = nvrtcResult (*)(nvrtcProgram, char*);
    using nvrtcGetPTXSize_t = nvrtcResult (*)(nvrtcProgram, size_t*);
    using nvrtcGetCUBIN_t = nvrtcResult (*)(nvrtcProgram, char*);
    using nvrtcGetCUBINSize_t = nvrtcResult (*)(nvrtcProgram, size_t*);
    using nvrtcGetProgramLog_t = nvrtcResult (*)(nvrtcProgram, char*);
    using nvrtcGetProgramLogSize_t = nvrtcResult (*)(nvrtcProgram, size_t*);
    using nvrtcAddNameExpression_t = nvrtcResult (*)(nvrtcProgram, const char* const);
    using nvrtcGetLoweredName_t = nvrtcResult (*)(nvrtcProgram, const char*, const char**);
    using nvrtcDestroyProgram_t = nvrtcResult (*)(nvrtcProgram*);
    using nvrtcGetErrorString_t = const char* (*)(nvrtcResult);

    nvrtcCreateProgram_t nvrtcCreateProgram;
    nvrtcCompileProgram_t nvrtcCompileProgram;
    nvrtcGetPTX_t nvrtcGetPTX;
    nvrtcGetPTXSize_t nvrtcGetPTXSize;
    nvrtcGetCUBIN_t nvrtcGetCUBIN;
    nvrtcGetCUBINSize_t nvrtcGetCUBINSize;
    nvrtcGetProgramLog_t nvrtcGetProgramLog;
    nvrtcGetProgramLogSize_t nvrtcGetProgramLogSize;
    nvrtcGetLoweredName_t nvrtcGetLoweredName;
    nvrtcAddNameExpression_t nvrtcAddNameExpression;
    nvrtcDestroyProgram_t nvrtcDestroyProgram;
    nvrtcGetErrorString_t nvrtcGetErrorString;

    NVRTC() {
#if defined(__linux__) || defined(__APPLE__)
        ownedHandle = true;
        // check if libnvrtc is already loaded to avoid loading multiple versions
        // in the same process
        nvrtcHandle = details::findLoadedLibrary("libnvrtc");
        if (!nvrtcHandle) {
            static const char* CANDIDATES[] = {
                "libnvrtc.so",
                "libnvrtc.so.11",
                "libnvrtc.so.12",
                "libnvrtc.so.13",
                "libnvrtc.so.14",
                "libnvrtc.so.15",
            };
            for (auto* candidate: CANDIDATES) {
                nvrtcHandle = dlopen(candidate, RTLD_NOW);
                if (nvrtcHandle != nullptr) {
                    break;
                }
            }
        }

#elif defined(_WIN32)
        // check if nvrtc.dll is already loaded to avoid loading multiple versions
        nvrtcHandle = details::findLoadedLibrary(L"nvrtc64");
        if (nvrtcHandle) {
            ownedHandle = false;
        } else {
            // otherwise, look into known path and pick the most recent version
            // if multiple are found
            auto dllPathOpt = details::FindBestCudaDll(L"nvrtc64");
            if (dllPathOpt) {
                auto dllPath = *dllPathOpt;
                // add the directory containing the DLL to the search path
                auto dir = dllPath.parent_path();
                SetDllDirectoryW(dir.c_str());

                nvrtcHandle = LoadLibraryExW(
                    dllPath.c_str(),
                    nullptr,
                    LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
                    LOAD_LIBRARY_SEARCH_DEFAULT_DIRS |
                    LOAD_LIBRARY_SEARCH_USER_DIRS
                );
                ownedHandle = true;
            }
        }
#else
#error "Platform not supported"
#endif

        if (nvrtcHandle) {
            // Load NVRTC function pointers using template
            nvrtcCreateProgram = details::loadSymbol<nvrtcCreateProgram_t>(nvrtcHandle, "nvrtcCreateProgram");
            nvrtcCompileProgram = details::loadSymbol<nvrtcCompileProgram_t>(nvrtcHandle, "nvrtcCompileProgram");
            nvrtcGetPTX = details::loadSymbol<nvrtcGetPTX_t>(nvrtcHandle, "nvrtcGetPTX");
            nvrtcGetPTXSize = details::loadSymbol<nvrtcGetPTXSize_t>(nvrtcHandle, "nvrtcGetPTXSize");
            nvrtcGetCUBIN = details::loadSymbol<nvrtcGetCUBIN_t>(nvrtcHandle, "nvrtcGetCUBIN");
            nvrtcGetCUBINSize = details::loadSymbol<nvrtcGetCUBINSize_t>(nvrtcHandle, "nvrtcGetCUBINSize");
            nvrtcGetProgramLog = details::loadSymbol<nvrtcGetProgramLog_t>(nvrtcHandle, "nvrtcGetProgramLog");
            nvrtcGetProgramLogSize = details::loadSymbol<nvrtcGetProgramLogSize_t>(nvrtcHandle, "nvrtcGetProgramLogSize");
            nvrtcGetLoweredName = details::loadSymbol<nvrtcGetLoweredName_t>(nvrtcHandle, "nvrtcGetLoweredName");
            nvrtcAddNameExpression = details::loadSymbol<nvrtcAddNameExpression_t>(nvrtcHandle, "nvrtcAddNameExpression");
            nvrtcDestroyProgram = details::loadSymbol<nvrtcDestroyProgram_t>(nvrtcHandle, "nvrtcDestroyProgram");
            nvrtcGetErrorString = details::loadSymbol<nvrtcGetErrorString_t>(nvrtcHandle, "nvrtcGetErrorString");
        }
    }

    ~NVRTC() {
#if defined(__linux__) || defined(__APPLE__)
        if (nvrtcHandle) {
            dlclose(nvrtcHandle);
        }
#elif defined(_WIN32)
        if (nvrtcHandle && ownedHandle) {
            FreeLibrary(static_cast<HMODULE>(nvrtcHandle));
        }
#else
#error "Platform not supported"
#endif
    }

    // Prevent copying
    NVRTC(const NVRTC&) = delete;
    NVRTC& operator=(const NVRTC&) = delete;

    void* nvrtcHandle = nullptr;
    bool ownedHandle = false;
};

namespace details {

inline const char* cudaDriverErrorString(CUresult error) {
    if (CUDADriver::instance().loaded()) {
        const char* errorStr = nullptr;
        CUDADriver::instance().cuGetErrorName(error, &errorStr);
        return errorStr ? errorStr : "Unknown CUDA driver error";
    }
    return "CUDA driver library not loaded";
}

inline const char* cudartErrorString(cudaError_t error) {
    if (CUDART::instance().loaded()) {
        const char* errorStr = CUDART::instance().cudaGetErrorString(error);
        return errorStr ? errorStr : "Unknown CUDA runtime error";
    }
    return "CUDA runtime library not loaded";
}

inline const char* nvrtcErrorString(nvrtcResult error) {
    if (NVRTC::instance().loaded()) {
        const char* errorStr = NVRTC::instance().nvrtcGetErrorString(error);
        return errorStr ? errorStr : "Unknown NVRTC error";
    }
    return "NVRTC library not loaded";
}

} // namespace details


#define GPULITE_CUDA_DRIVER_CALL(func) \
    gpulite::details::checkCall(gpulite::CUDADriver::instance().func, CUDA_SUCCESS, gpulite::details::cudaDriverErrorString, __FILE__, __LINE__, #func)

#define GPULITE_CUDART_CALL(func) \
    gpulite::details::checkCall(gpulite::CUDART::instance().func, cudaSuccess, gpulite::details::cudartErrorString, __FILE__, __LINE__, #func)

#define GPULITE_NVRTC_CALL(func) \
    gpulite::details::checkCall(gpulite::NVRTC::instance().func, NVRTC_SUCCESS, gpulite::details::nvrtcErrorString, __FILE__, __LINE__, #func)

} // namespace gpulite

#endif // GPULITE_CUDA_LIB_HPP
