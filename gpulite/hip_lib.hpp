// gpulite: singleton classes dynamically loading the HIP runtime and HIPRTC
// libraries, and macros to check the result of calls to them
#ifndef GPULITE_HIP_LIB_HPP
#define GPULITE_HIP_LIB_HPP

#include "./hip_api.hpp"
#include "./dlopen.hpp"

namespace gpulite {

#ifdef _WIN32
namespace details {

/// Try to load the first DLL that exists in `candidates`, looking first inside
/// `%HIP_PATH%\bin` and then in the default DLL search path.
inline HMODULE loadHipDll(const std::vector<std::wstring>& candidates) {
    auto hipPath = GetEnvVar(L"HIP_PATH");
    if (!hipPath.empty()) {
        auto bin = std::filesystem::path(hipPath) / L"bin";
        for (const auto& candidate: candidates) {
            auto dllPath = bin / candidate;
            std::error_code ec;
            if (!std::filesystem::exists(dllPath, ec) || ec) {
                continue;
            }

            auto handle = LoadLibraryExW(
                dllPath.c_str(),
                nullptr,
                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
            );
            if (handle != nullptr) {
                return handle;
            }
        }
    }

    for (const auto& candidate: candidates) {
        auto handle = LoadLibraryW(candidate.c_str());
        if (handle != nullptr) {
            return handle;
        }
    }

    return nullptr;
}

} // namespace details
#endif

/*
This class allows us to dynamically load the HIP runtime and reference the
functions contained within the libamdhip64.so library (see HIP Runtime API:
https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/modules.html).

Unlike CUDA, HIP does not have a separate driver library, and both the
runtime-style and driver-style (module, context, ...) functions live in the
same library.
*/
class HIPRuntime {
  public:
    static HIPRuntime& instance() {
        static HIPRuntime instance;
        return instance;
    }

    static bool loaded() { return instance().hipHandle != nullptr; }

    // runtime-style API
    using hipGetDeviceCount_t = hipError_t (*)(int*);
    using hipGetDevice_t = hipError_t (*)(int*);
    using hipSetDevice_t = hipError_t (*)(int);
    using hipMalloc_t = hipError_t (*)(void**, size_t);
    using hipMemcpy_t = hipError_t (*)(void*, const void*, size_t, hipMemcpyKind);
    using hipMemset_t = hipError_t (*)(void*, int, size_t);
    using hipGetErrorName_t = const char* (*)(hipError_t);
    using hipGetErrorString_t = const char* (*)(hipError_t);
    using hipDeviceSynchronize_t = hipError_t (*)(void);
    using hipPointerGetAttributes_t = hipError_t (*)(hipPointerAttribute_t*, const void*);
    using hipFree_t = hipError_t (*)(void*);
    using hipRuntimeGetVersion_t = hipError_t (*)(int*);
    using hipDriverGetVersion_t = hipError_t (*)(int*);
    using hipStreamCreate_t = hipError_t (*)(hipStream_t*);
    using hipStreamCreateWithFlags_t = hipError_t (*)(hipStream_t*, unsigned int);
    using hipStreamDestroy_t = hipError_t (*)(hipStream_t);
    using hipStreamSynchronize_t = hipError_t (*)(hipStream_t);
    using hipHostMalloc_t = hipError_t (*)(void**, size_t, unsigned int);
    using hipHostFree_t = hipError_t (*)(void*);
    using hipHostGetDevicePointer_t = hipError_t (*)(void**, void*, unsigned int);
    using hipMemcpyAsync_t = hipError_t (*)(void*, const void*, size_t, hipMemcpyKind, hipStream_t);
    using hipMemsetAsync_t = hipError_t (*)(void*, int, size_t, hipStream_t);
    using hipMallocAsync_t = hipError_t (*)(void**, size_t, hipStream_t);
    using hipFreeAsync_t = hipError_t (*)(void*, hipStream_t);
    using hipEventCreate_t = hipError_t (*)(hipEvent_t*);
    using hipEventCreateWithFlags_t = hipError_t (*)(hipEvent_t*, unsigned int);
    using hipEventDestroy_t = hipError_t (*)(hipEvent_t);
    using hipEventRecord_t = hipError_t (*)(hipEvent_t, hipStream_t);
    using hipEventSynchronize_t = hipError_t (*)(hipEvent_t);
    using hipEventQuery_t = hipError_t (*)(hipEvent_t);
    using hipEventElapsedTime_t = hipError_t (*)(float*, hipEvent_t, hipEvent_t);
    using hipStreamWaitEvent_t = hipError_t (*)(hipStream_t, hipEvent_t, unsigned int);

    // driver-style API
    using hipInit_t = hipError_t (*)(unsigned int);
    using hipDeviceGet_t = hipError_t (*)(hipDevice_t*, int);
    using hipCtxCreate_t = hipError_t (*)(hipCtx_t*, unsigned int, hipDevice_t);
    using hipCtxDestroy_t = hipError_t (*)(hipCtx_t);
    using hipDevicePrimaryCtxRetain_t = hipError_t (*)(hipCtx_t*, hipDevice_t);
    using hipDevicePrimaryCtxRelease_t = hipError_t (*)(hipDevice_t);
    using hipCtxGetCurrent_t = hipError_t (*)(hipCtx_t*);
    using hipCtxSetCurrent_t = hipError_t (*)(hipCtx_t);
    using hipCtxPushCurrent_t = hipError_t (*)(hipCtx_t);
    using hipCtxPopCurrent_t = hipError_t (*)(hipCtx_t*);
    using hipCtxGetDevice_t = hipError_t (*)(hipDevice_t*);
    using hipCtxSynchronize_t = hipError_t (*)(void);
    using hipModuleLoadData_t = hipError_t (*)(hipModule_t*, const void*);
    using hipModuleLoadDataEx_t = hipError_t (*)(hipModule_t*, const void*, unsigned int, hipJitOption*, void**);
    using hipModuleUnload_t = hipError_t (*)(hipModule_t);
    using hipModuleGetFunction_t = hipError_t (*)(hipFunction_t*, hipModule_t, const char*);
    using hipFuncGetAttribute_t = hipError_t (*)(int*, hipFunction_attribute, hipFunction_t);
    using hipFuncSetAttribute_t = hipError_t (*)(const void*, hipFuncAttribute, int);
    using hipDeviceGetAttribute_t = hipError_t (*)(int*, hipDeviceAttribute_t, int);
    using hipDeviceGetName_t = hipError_t (*)(char*, int, hipDevice_t);
    using hipDeviceTotalMem_t = hipError_t (*)(size_t*, hipDevice_t);
    using hipPointerGetAttribute_t = hipError_t (*)(void*, hipPointer_attribute, hipDeviceptr_t);
    using hipMemsetD8Async_t = hipError_t (*)(hipDeviceptr_t, unsigned char, size_t, hipStream_t);
    using hipMemsetD16Async_t = hipError_t (*)(hipDeviceptr_t, unsigned short, size_t, hipStream_t);
    using hipMemsetD32Async_t = hipError_t (*)(hipDeviceptr_t, int, size_t, hipStream_t);
    using hipModuleLaunchKernel_t = hipError_t (*)(
        hipFunction_t,
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int,
        hipStream_t,
        void**,
        void**
    );

    hipGetDeviceCount_t hipGetDeviceCount;
    hipGetDevice_t hipGetDevice;
    hipSetDevice_t hipSetDevice;
    hipMalloc_t hipMalloc;
    hipMemset_t hipMemset;
    hipMemcpy_t hipMemcpy;
    hipGetErrorName_t hipGetErrorName;
    hipGetErrorString_t hipGetErrorString;
    hipDeviceSynchronize_t hipDeviceSynchronize;
    hipPointerGetAttributes_t hipPointerGetAttributes;
    hipFree_t hipFree;
    hipRuntimeGetVersion_t hipRuntimeGetVersion;
    hipDriverGetVersion_t hipDriverGetVersion;
    hipStreamCreate_t hipStreamCreate;
    hipStreamCreateWithFlags_t hipStreamCreateWithFlags;
    hipStreamDestroy_t hipStreamDestroy;
    hipStreamSynchronize_t hipStreamSynchronize;
    hipHostMalloc_t hipHostMalloc;
    hipHostFree_t hipHostFree;
    hipHostGetDevicePointer_t hipHostGetDevicePointer;
    hipMemcpyAsync_t hipMemcpyAsync;
    hipMemsetAsync_t hipMemsetAsync;
    hipMallocAsync_t hipMallocAsync;
    hipFreeAsync_t hipFreeAsync;
    hipEventCreate_t hipEventCreate;
    hipEventCreateWithFlags_t hipEventCreateWithFlags;
    hipEventDestroy_t hipEventDestroy;
    hipEventRecord_t hipEventRecord;
    hipEventSynchronize_t hipEventSynchronize;
    hipEventQuery_t hipEventQuery;
    hipEventElapsedTime_t hipEventElapsedTime;
    hipStreamWaitEvent_t hipStreamWaitEvent;

    hipInit_t hipInit;
    hipDeviceGet_t hipDeviceGet;
    hipCtxCreate_t hipCtxCreate;
    hipCtxDestroy_t hipCtxDestroy;
    hipDevicePrimaryCtxRetain_t hipDevicePrimaryCtxRetain;
    hipDevicePrimaryCtxRelease_t hipDevicePrimaryCtxRelease;
    hipCtxGetCurrent_t hipCtxGetCurrent;
    hipCtxSetCurrent_t hipCtxSetCurrent;
    hipCtxPushCurrent_t hipCtxPushCurrent;
    hipCtxPopCurrent_t hipCtxPopCurrent;
    hipCtxGetDevice_t hipCtxGetDevice;
    hipCtxSynchronize_t hipCtxSynchronize;
    hipModuleLoadData_t hipModuleLoadData;
    hipModuleLoadDataEx_t hipModuleLoadDataEx;
    hipModuleUnload_t hipModuleUnload;
    hipModuleGetFunction_t hipModuleGetFunction;
    hipFuncGetAttribute_t hipFuncGetAttribute;
    hipFuncSetAttribute_t hipFuncSetAttribute;
    hipDeviceGetAttribute_t hipDeviceGetAttribute;
    hipDeviceGetName_t hipDeviceGetName;
    hipDeviceTotalMem_t hipDeviceTotalMem;
    hipPointerGetAttribute_t hipPointerGetAttribute;
    hipMemsetD8Async_t hipMemsetD8Async;
    hipMemsetD16Async_t hipMemsetD16Async;
    hipMemsetD32Async_t hipMemsetD32Async;
    hipModuleLaunchKernel_t hipModuleLaunchKernel;

    HIPRuntime() {
#if defined(__linux__) || defined(__APPLE__)
        ownedHandle = true;
        // First try to find an already loaded libamdhip64 to avoid loading
        // multiple versions in the same process
        hipHandle = details::findLoadedLibrary("libamdhip64.");

        // otherwise, try multiple candidate names to maximize compatibility
        // with different ROCm versions
        if (!hipHandle) {
            static const char* CANDIDATES[] = {
                "libamdhip64.so",
                "libamdhip64.so.7",
                "libamdhip64.so.6",
                "libamdhip64.so.8",
            };
            for (auto* candidate: CANDIDATES) {
                hipHandle = dlopen(candidate, RTLD_NOW);
                if (hipHandle) {
                    break;
                }
            }
        }
#elif defined(_WIN32)
        // First try to find an already loaded amdhip64.dll to avoid loading
        // multiple versions in the same process
        hipHandle = details::findLoadedLibrary(L"amdhip64");
        if (hipHandle != nullptr) {
            ownedHandle = false;
        } else {
            hipHandle = details::loadHipDll({
                L"amdhip64_7.dll",
                L"amdhip64_6.dll",
                L"amdhip64_8.dll",
                L"amdhip64.dll",
            });
            ownedHandle = true;
        }
#else
#error "Platform not supported"
#endif
        if (hipHandle) {
            // load HIP function pointers using template
            hipGetDeviceCount = details::loadSymbol<hipGetDeviceCount_t>(hipHandle, "hipGetDeviceCount");
            hipGetDevice = details::loadSymbol<hipGetDevice_t>(hipHandle, "hipGetDevice");
            hipSetDevice = details::loadSymbol<hipSetDevice_t>(hipHandle, "hipSetDevice");
            hipMalloc = details::loadSymbol<hipMalloc_t>(hipHandle, "hipMalloc");
            hipMemset = details::loadSymbol<hipMemset_t>(hipHandle, "hipMemset");
            hipMemcpy = details::loadSymbol<hipMemcpy_t>(hipHandle, "hipMemcpy");
            hipGetErrorName = details::loadSymbol<hipGetErrorName_t>(hipHandle, "hipGetErrorName");
            hipGetErrorString = details::loadSymbol<hipGetErrorString_t>(hipHandle, "hipGetErrorString");
            hipDeviceSynchronize = details::loadSymbol<hipDeviceSynchronize_t>(hipHandle, "hipDeviceSynchronize");
            hipPointerGetAttributes = details::loadSymbol<hipPointerGetAttributes_t>(hipHandle, "hipPointerGetAttributes");
            hipFree = details::loadSymbol<hipFree_t>(hipHandle, "hipFree");
            hipRuntimeGetVersion = details::loadSymbol<hipRuntimeGetVersion_t>(hipHandle, "hipRuntimeGetVersion");
            hipDriverGetVersion = details::loadSymbol<hipDriverGetVersion_t>(hipHandle, "hipDriverGetVersion");
            hipStreamCreate = details::loadSymbol<hipStreamCreate_t>(hipHandle, "hipStreamCreate");
            hipStreamCreateWithFlags = details::loadSymbol<hipStreamCreateWithFlags_t>(hipHandle, "hipStreamCreateWithFlags");
            hipStreamDestroy = details::loadSymbol<hipStreamDestroy_t>(hipHandle, "hipStreamDestroy");
            hipStreamSynchronize = details::loadSymbol<hipStreamSynchronize_t>(hipHandle, "hipStreamSynchronize");
            hipHostMalloc = details::loadSymbol<hipHostMalloc_t>(hipHandle, "hipHostMalloc");
            hipHostFree = details::loadSymbol<hipHostFree_t>(hipHandle, "hipHostFree");
            hipHostGetDevicePointer = details::loadSymbol<hipHostGetDevicePointer_t>(hipHandle, "hipHostGetDevicePointer");
            hipMemcpyAsync = details::loadSymbol<hipMemcpyAsync_t>(hipHandle, "hipMemcpyAsync");
            hipMemsetAsync = details::loadSymbol<hipMemsetAsync_t>(hipHandle, "hipMemsetAsync");
            hipMallocAsync = details::loadSymbol<hipMallocAsync_t>(hipHandle, "hipMallocAsync");
            hipFreeAsync = details::loadSymbol<hipFreeAsync_t>(hipHandle, "hipFreeAsync");
            hipEventCreate = details::loadSymbol<hipEventCreate_t>(hipHandle, "hipEventCreate");
            hipEventCreateWithFlags = details::loadSymbol<hipEventCreateWithFlags_t>(hipHandle, "hipEventCreateWithFlags");
            hipEventDestroy = details::loadSymbol<hipEventDestroy_t>(hipHandle, "hipEventDestroy");
            hipEventRecord = details::loadSymbol<hipEventRecord_t>(hipHandle, "hipEventRecord");
            hipEventSynchronize = details::loadSymbol<hipEventSynchronize_t>(hipHandle, "hipEventSynchronize");
            hipEventQuery = details::loadSymbol<hipEventQuery_t>(hipHandle, "hipEventQuery");
            hipEventElapsedTime = details::loadSymbol<hipEventElapsedTime_t>(hipHandle, "hipEventElapsedTime");
            hipStreamWaitEvent = details::loadSymbol<hipStreamWaitEvent_t>(hipHandle, "hipStreamWaitEvent");

            hipInit = details::loadSymbol<hipInit_t>(hipHandle, "hipInit");
            hipDeviceGet = details::loadSymbol<hipDeviceGet_t>(hipHandle, "hipDeviceGet");
            hipCtxCreate = details::loadSymbol<hipCtxCreate_t>(hipHandle, "hipCtxCreate");
            hipCtxDestroy = details::loadSymbol<hipCtxDestroy_t>(hipHandle, "hipCtxDestroy");
            hipDevicePrimaryCtxRetain = details::loadSymbol<hipDevicePrimaryCtxRetain_t>(hipHandle, "hipDevicePrimaryCtxRetain");
            hipDevicePrimaryCtxRelease = details::loadSymbol<hipDevicePrimaryCtxRelease_t>(hipHandle, "hipDevicePrimaryCtxRelease");
            hipCtxGetCurrent = details::loadSymbol<hipCtxGetCurrent_t>(hipHandle, "hipCtxGetCurrent");
            hipCtxSetCurrent = details::loadSymbol<hipCtxSetCurrent_t>(hipHandle, "hipCtxSetCurrent");
            hipCtxPushCurrent = details::loadSymbol<hipCtxPushCurrent_t>(hipHandle, "hipCtxPushCurrent");
            hipCtxPopCurrent = details::loadSymbol<hipCtxPopCurrent_t>(hipHandle, "hipCtxPopCurrent");
            hipCtxGetDevice = details::loadSymbol<hipCtxGetDevice_t>(hipHandle, "hipCtxGetDevice");
            hipCtxSynchronize = details::loadSymbol<hipCtxSynchronize_t>(hipHandle, "hipCtxSynchronize");
            hipModuleLoadData = details::loadSymbol<hipModuleLoadData_t>(hipHandle, "hipModuleLoadData");
            hipModuleLoadDataEx = details::loadSymbol<hipModuleLoadDataEx_t>(hipHandle, "hipModuleLoadDataEx");
            hipModuleUnload = details::loadSymbol<hipModuleUnload_t>(hipHandle, "hipModuleUnload");
            hipModuleGetFunction = details::loadSymbol<hipModuleGetFunction_t>(hipHandle, "hipModuleGetFunction");
            hipFuncGetAttribute = details::loadSymbol<hipFuncGetAttribute_t>(hipHandle, "hipFuncGetAttribute");
            hipFuncSetAttribute = details::loadSymbol<hipFuncSetAttribute_t>(hipHandle, "hipFuncSetAttribute");
            hipDeviceGetAttribute = details::loadSymbol<hipDeviceGetAttribute_t>(hipHandle, "hipDeviceGetAttribute");
            hipDeviceGetName = details::loadSymbol<hipDeviceGetName_t>(hipHandle, "hipDeviceGetName");
            hipDeviceTotalMem = details::loadSymbol<hipDeviceTotalMem_t>(hipHandle, "hipDeviceTotalMem");
            hipPointerGetAttribute = details::loadSymbol<hipPointerGetAttribute_t>(hipHandle, "hipPointerGetAttribute");
            hipMemsetD8Async = details::loadSymbol<hipMemsetD8Async_t>(hipHandle, "hipMemsetD8Async");
            hipMemsetD16Async = details::loadSymbol<hipMemsetD16Async_t>(hipHandle, "hipMemsetD16Async");
            hipMemsetD32Async = details::loadSymbol<hipMemsetD32Async_t>(hipHandle, "hipMemsetD32Async");
            hipModuleLaunchKernel = details::loadSymbol<hipModuleLaunchKernel_t>(hipHandle, "hipModuleLaunchKernel");
        }
    }

    ~HIPRuntime() {
#if defined(__linux__) || defined(__APPLE__)
        if (hipHandle) {
            dlclose(hipHandle);
        }
#elif defined(_WIN32)
        if (hipHandle && ownedHandle) {
            FreeLibrary(static_cast<HMODULE>(hipHandle));
        }
#else
#error "Platform not supported"
#endif
    }

    // Prevent copying
    HIPRuntime(const HIPRuntime&) = delete;
    HIPRuntime& operator=(const HIPRuntime&) = delete;

    void* hipHandle = nullptr;
    bool ownedHandle = false;
};

/*
This class allows us to dynamically load HIPRTC and reference the functions
contained within the libhiprtc.so library (see HIPRTC API:
https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_rtc.html).
*/
class HIPRTC {

  public:
    static HIPRTC& instance() {
        static HIPRTC instance;
        return instance;
    }

    static bool loaded() { return instance().hiprtcHandle != nullptr; }

    using hiprtcCreateProgram_t = hiprtcResult (*)(hiprtcProgram*, const char*, const char*, int, const char* const*, const char* const*);
    using hiprtcCompileProgram_t = hiprtcResult (*)(hiprtcProgram, int, const char* const*);
    using hiprtcGetCode_t = hiprtcResult (*)(hiprtcProgram, char*);
    using hiprtcGetCodeSize_t = hiprtcResult (*)(hiprtcProgram, size_t*);
    using hiprtcGetBitcode_t = hiprtcResult (*)(hiprtcProgram, char*);
    using hiprtcGetBitcodeSize_t = hiprtcResult (*)(hiprtcProgram, size_t*);
    using hiprtcGetProgramLog_t = hiprtcResult (*)(hiprtcProgram, char*);
    using hiprtcGetProgramLogSize_t = hiprtcResult (*)(hiprtcProgram, size_t*);
    using hiprtcAddNameExpression_t = hiprtcResult (*)(hiprtcProgram, const char*);
    using hiprtcGetLoweredName_t = hiprtcResult (*)(hiprtcProgram, const char*, const char**);
    using hiprtcDestroyProgram_t = hiprtcResult (*)(hiprtcProgram*);
    using hiprtcGetErrorString_t = const char* (*)(hiprtcResult);
    using hiprtcVersion_t = hiprtcResult (*)(int*, int*);

    hiprtcCreateProgram_t hiprtcCreateProgram;
    hiprtcCompileProgram_t hiprtcCompileProgram;
    hiprtcGetCode_t hiprtcGetCode;
    hiprtcGetCodeSize_t hiprtcGetCodeSize;
    hiprtcGetBitcode_t hiprtcGetBitcode;
    hiprtcGetBitcodeSize_t hiprtcGetBitcodeSize;
    hiprtcGetProgramLog_t hiprtcGetProgramLog;
    hiprtcGetProgramLogSize_t hiprtcGetProgramLogSize;
    hiprtcGetLoweredName_t hiprtcGetLoweredName;
    hiprtcAddNameExpression_t hiprtcAddNameExpression;
    hiprtcDestroyProgram_t hiprtcDestroyProgram;
    hiprtcGetErrorString_t hiprtcGetErrorString;
    hiprtcVersion_t hiprtcVersion;

    HIPRTC() {
#if defined(__linux__) || defined(__APPLE__)
        ownedHandle = true;
        // check if libhiprtc is already loaded to avoid loading multiple
        // versions in the same process. The trailing dot prevents matching
        // `libhiprtc-builtins.so`.
        hiprtcHandle = details::findLoadedLibrary("libhiprtc.");
        if (!hiprtcHandle) {
            static const char* CANDIDATES[] = {
                "libhiprtc.so",
                "libhiprtc.so.7",
                "libhiprtc.so.6",
                "libhiprtc.so.8",
            };
            for (auto* candidate: CANDIDATES) {
                hiprtcHandle = dlopen(candidate, RTLD_NOW);
                if (hiprtcHandle != nullptr) {
                    break;
                }
            }
        }

#elif defined(_WIN32)
        // check if hiprtc is already loaded to avoid loading multiple versions.
        // The DLL is named hiprtcMMmm.dll, and the `0` in the prefix prevents
        // matching `hiprtc-builtinsMMmm.dll`.
        hiprtcHandle = details::findLoadedLibrary(L"hiprtc0");
        if (hiprtcHandle) {
            ownedHandle = false;
        } else {
            // otherwise, look for the most recent version
            auto candidates = std::vector<std::wstring>();
            for (int major = 8; major >= 6; major--) {
                for (int minor = 9; minor >= 0; minor--) {
                    candidates.push_back(
                        L"hiprtc0" + std::to_wstring(major) +
                        L"0" + std::to_wstring(minor) + L".dll"
                    );
                }
            }
            candidates.push_back(L"hiprtc.dll");

            hiprtcHandle = details::loadHipDll(candidates);
            ownedHandle = true;
        }
#else
#error "Platform not supported"
#endif

        if (hiprtcHandle) {
            // Load HIPRTC function pointers using template
            hiprtcCreateProgram = details::loadSymbol<hiprtcCreateProgram_t>(hiprtcHandle, "hiprtcCreateProgram");
            hiprtcCompileProgram = details::loadSymbol<hiprtcCompileProgram_t>(hiprtcHandle, "hiprtcCompileProgram");
            hiprtcGetCode = details::loadSymbol<hiprtcGetCode_t>(hiprtcHandle, "hiprtcGetCode");
            hiprtcGetCodeSize = details::loadSymbol<hiprtcGetCodeSize_t>(hiprtcHandle, "hiprtcGetCodeSize");
            hiprtcGetBitcode = details::loadSymbol<hiprtcGetBitcode_t>(hiprtcHandle, "hiprtcGetBitcode");
            hiprtcGetBitcodeSize = details::loadSymbol<hiprtcGetBitcodeSize_t>(hiprtcHandle, "hiprtcGetBitcodeSize");
            hiprtcGetProgramLog = details::loadSymbol<hiprtcGetProgramLog_t>(hiprtcHandle, "hiprtcGetProgramLog");
            hiprtcGetProgramLogSize = details::loadSymbol<hiprtcGetProgramLogSize_t>(hiprtcHandle, "hiprtcGetProgramLogSize");
            hiprtcGetLoweredName = details::loadSymbol<hiprtcGetLoweredName_t>(hiprtcHandle, "hiprtcGetLoweredName");
            hiprtcAddNameExpression = details::loadSymbol<hiprtcAddNameExpression_t>(hiprtcHandle, "hiprtcAddNameExpression");
            hiprtcDestroyProgram = details::loadSymbol<hiprtcDestroyProgram_t>(hiprtcHandle, "hiprtcDestroyProgram");
            hiprtcGetErrorString = details::loadSymbol<hiprtcGetErrorString_t>(hiprtcHandle, "hiprtcGetErrorString");
            hiprtcVersion = details::loadSymbol<hiprtcVersion_t>(hiprtcHandle, "hiprtcVersion");
        }
    }

    ~HIPRTC() {
#if defined(__linux__) || defined(__APPLE__)
        if (hiprtcHandle) {
            dlclose(hiprtcHandle);
        }
#elif defined(_WIN32)
        if (hiprtcHandle && ownedHandle) {
            FreeLibrary(static_cast<HMODULE>(hiprtcHandle));
        }
#else
#error "Platform not supported"
#endif
    }

    // Prevent copying
    HIPRTC(const HIPRTC&) = delete;
    HIPRTC& operator=(const HIPRTC&) = delete;

    void* hiprtcHandle = nullptr;
    bool ownedHandle = false;
};

namespace details {

inline const char* hipErrorString(hipError_t error) {
    if (HIPRuntime::instance().loaded()) {
        const char* errorStr = HIPRuntime::instance().hipGetErrorString(error);
        return errorStr ? errorStr : "Unknown HIP error";
    }
    return "HIP runtime library not loaded";
}

inline const char* hiprtcErrorString(hiprtcResult error) {
    if (HIPRTC::instance().loaded()) {
        const char* errorStr = HIPRTC::instance().hiprtcGetErrorString(error);
        return errorStr ? errorStr : "Unknown HIPRTC error";
    }
    return "HIPRTC library not loaded";
}

} // namespace details


#define GPULITE_HIP_CALL(func) \
    gpulite::details::checkCall(gpulite::HIPRuntime::instance().func, hipSuccess, gpulite::details::hipErrorString, __FILE__, __LINE__, #func)

#define GPULITE_HIPRTC_CALL(func) \
    gpulite::details::checkCall(gpulite::HIPRTC::instance().func, HIPRTC_SUCCESS, gpulite::details::hiprtcErrorString, __FILE__, __LINE__, #func)

} // namespace gpulite

#endif // GPULITE_HIP_LIB_HPP
