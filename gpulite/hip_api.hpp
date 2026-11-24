// gpulite: re-declaration of the parts of the official HIP/HIPRTC headers
// (hip_runtime_api.h, driver_types.h, linker_types.h, hiprtc.h) used by
// gpulite, so the code can be compiled without the ROCm/HIP SDK.
//
// The values here correspond to ROCm 6 and later.
#ifndef GPULITE_HIP_API_HPP
#define GPULITE_HIP_API_HPP

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif


// ========================================================================== //
//         HIP Runtime API (from hip_runtime_api.h and driver_types.h)        //
// ========================================================================== //
typedef int hipDevice_t;
typedef struct ihipCtx_t* hipCtx_t;
typedef struct ihipModule_t* hipModule_t;
typedef struct ihipModuleSymbol_t* hipFunction_t;
typedef struct ihipStream_t* hipStream_t;
typedef struct ihipEvent_t* hipEvent_t;
typedef void* hipDeviceptr_t;

typedef enum hipError_t {
    hipSuccess = 0,
    hipErrorInvalidValue = 1,
    hipErrorOutOfMemory = 2,
    hipErrorMemoryAllocation = 2,
    hipErrorNotInitialized = 3,
    hipErrorInitializationError = 3,
    hipErrorDeinitialized = 4,
    // [...]
    hipErrorInvalidDevice = 101,
    // [...]
    hipErrorInvalidContext = 201,
    // [...]
    hipErrorNotReady = 600,
    // [...]
    hipErrorNotSupported = 801,
    // [...]
    hipErrorUnknown = 999,
    // [...]
} hipError_t;

typedef enum hipDeviceAttribute_t {
    // [...]
    hipDeviceAttributeClockRate = 5,
    // [...]
    hipDeviceAttributeL2CacheSize = 19,
    // [...]
    hipDeviceAttributeComputeCapabilityMajor = 23,
    // [...]
    hipDeviceAttributeMaxBlockDimX = 26,
    hipDeviceAttributeMaxBlockDimY = 27,
    hipDeviceAttributeMaxBlockDimZ = 28,
    hipDeviceAttributeMaxGridDimX = 29,
    hipDeviceAttributeMaxGridDimY = 30,
    hipDeviceAttributeMaxGridDimZ = 31,
    // [...]
    hipDeviceAttributeMaxThreadsPerBlock = 56,
    hipDeviceAttributeMaxThreadsPerMultiProcessor = 57,
    hipDeviceAttributeMaxPitch = 58,
    // [...]
    hipDeviceAttributeComputeCapabilityMinor = 61,
    // [...]
    hipDeviceAttributeMultiprocessorCount = 63,
    // [...]
    hipDeviceAttributeMaxRegistersPerBlock = 71,
    // [...]
    hipDeviceAttributeReservedSharedMemPerBlock = 73,
    hipDeviceAttributeMaxSharedMemoryPerBlock = 74,
    hipDeviceAttributeSharedMemPerBlockOptin = 75,
    hipDeviceAttributeSharedMemPerMultiprocessor = 76,
    // [...]
    hipDeviceAttributeTotalConstantMemory = 83,
    hipDeviceAttributeTotalGlobalMem = 84,
    // [...]
    hipDeviceAttributeWarpSize = 87,
    // [...]
    hipDeviceAttributeMaxSharedMemoryPerMultiprocessor = 10002,
    // [...]
} hipDeviceAttribute_t;

typedef enum hipFunction_attribute {
    HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,
    HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1,
    HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2,
    HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3,
    HIP_FUNC_ATTRIBUTE_NUM_REGS = 4,
    HIP_FUNC_ATTRIBUTE_PTX_VERSION = 5,
    HIP_FUNC_ATTRIBUTE_BINARY_VERSION = 6,
    HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7,
    HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8,
    HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9,
    HIP_FUNC_ATTRIBUTE_MAX
} hipFunction_attribute;

// used with hipFuncSetAttribute
typedef enum hipFuncAttribute {
    hipFuncAttributeMaxDynamicSharedMemorySize = 8,
    hipFuncAttributePreferredSharedMemoryCarveout = 9,
    hipFuncAttributeMax
} hipFuncAttribute;

typedef enum hipPointer_attribute {
    HIP_POINTER_ATTRIBUTE_CONTEXT = 1,
    HIP_POINTER_ATTRIBUTE_MEMORY_TYPE = 2,
    HIP_POINTER_ATTRIBUTE_DEVICE_POINTER = 3,
    HIP_POINTER_ATTRIBUTE_HOST_POINTER = 4,
    HIP_POINTER_ATTRIBUTE_P2P_TOKENS = 5,
    HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS = 6,
    HIP_POINTER_ATTRIBUTE_BUFFER_ID = 7,
    HIP_POINTER_ATTRIBUTE_IS_MANAGED = 8,
    HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9,
    HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE = 10,
    HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR = 11,
    HIP_POINTER_ATTRIBUTE_RANGE_SIZE = 12,
    HIP_POINTER_ATTRIBUTE_MAPPED = 13,
    HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES = 14,
    HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE = 15,
    HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS = 16,
    HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE = 17
} hipPointer_attribute;

// Note: these values changed in ROCm 6 to match CUDA
typedef enum hipMemoryType {
    hipMemoryTypeUnregistered = 0,
    hipMemoryTypeHost = 1,
    hipMemoryTypeDevice = 2,
    hipMemoryTypeManaged = 3,
    hipMemoryTypeArray = 10,
    hipMemoryTypeUnified = 11
} hipMemoryType;

typedef struct hipPointerAttribute_t {
    enum hipMemoryType type;
    int device;
    void* devicePointer;
    void* hostPointer;
    int isManaged;
    unsigned allocationFlags;
} hipPointerAttribute_t;

typedef enum hipMemcpyKind {
    hipMemcpyHostToHost = 0,
    hipMemcpyHostToDevice = 1,
    hipMemcpyDeviceToHost = 2,
    hipMemcpyDeviceToDevice = 3,
    hipMemcpyDefault = 4,
    hipMemcpyDeviceToDeviceNoCU = 1024
} hipMemcpyKind;

// from linker_types.h
typedef enum hipJitOption {
    hipJitOptionMaxRegisters = 0,
    hipJitOptionThreadsPerBlock = 1,
    hipJitOptionWallTime = 2,
    hipJitOptionInfoLogBuffer = 3,
    hipJitOptionInfoLogBufferSizeBytes = 4,
    hipJitOptionErrorLogBuffer = 5,
    hipJitOptionErrorLogBufferSizeBytes = 6,
    hipJitOptionOptimizationLevel = 7,
    hipJitOptionTargetFromContext = 8,
    hipJitOptionTarget = 9,
    hipJitOptionFallbackStrategy = 10,
    hipJitOptionGenerateDebugInfo = 11,
    hipJitOptionLogVerbose = 12,
    hipJitOptionGenerateLineInfo = 13,
    hipJitOptionCacheMode = 14,
    hipJitOptionSm3xOpt = 15,
    hipJitOptionFastCompile = 16,
    hipJitOptionGlobalSymbolNames = 17,
    hipJitOptionGlobalSymbolAddresses = 18,
    hipJitOptionGlobalSymbolCount = 19,
    hipJitOptionLto = 20,
    hipJitOptionFtz = 21,
    hipJitOptionPrecDiv = 22,
    hipJitOptionPrecSqrt = 23,
    hipJitOptionFma = 24,
    hipJitOptionPositionIndependentCode = 25,
    hipJitOptionMinCTAPerSM = 26,
    hipJitOptionMaxThreadsPerBlock = 27,
    hipJitOptionOverrideDirectiveValues = 28,
    hipJitOptionNumOptions = 29,
    hipJitOptionIRtoISAOptExt = 10000,
    hipJitOptionIRtoISAOptCountExt = 10001,
} hipJitOption;

#ifndef GPULITE_DIM3_DEFINED
#define GPULITE_DIM3_DEFINED
struct dim3 {
    unsigned int x, y, z;
    dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
};
#endif

#define hipHostMallocDefault                0x00
#define hipHostMallocPortable               0x01
#define hipHostMallocMapped                 0x02
#define hipHostMallocWriteCombined          0x04

#define hipEventDefault                     0x00
#define hipEventBlockingSync                0x01
#define hipEventDisableTiming               0x02
#define hipEventInterprocess                0x04

#define hipEventWaitDefault                 0x00
#define hipEventWaitExternal                0x01

// allow including `.hip` files in the CPU code
#if !defined(__HIPCC__) && !defined(__CUDACC__) && !defined(__global__)
#define __global__
#endif

// ========================================================================== //
//                         HIPRTC API (from hiprtc.h)                         //
// ========================================================================== //

typedef enum hiprtcResult {
    HIPRTC_SUCCESS = 0,
    HIPRTC_ERROR_OUT_OF_MEMORY = 1,
    HIPRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,
    HIPRTC_ERROR_INVALID_INPUT = 3,
    HIPRTC_ERROR_INVALID_PROGRAM = 4,
    HIPRTC_ERROR_INVALID_OPTION = 5,
    HIPRTC_ERROR_COMPILATION = 6,
    HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7,
    HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8,
    HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9,
    HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10,
    HIPRTC_ERROR_INTERNAL_ERROR = 11,
    HIPRTC_ERROR_LINKING = 100
} hiprtcResult;

typedef struct _hiprtcProgram* hiprtcProgram;


#ifdef __cplusplus
}
#endif

#endif // GPULITE_HIP_API_HPP
