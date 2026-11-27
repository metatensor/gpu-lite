// gpulite: re-declaration of the parts of the official CUDA/NVRTC headers
// (cuda.h, cuda_runtime_api.h, nvrtc.h) used by gpulite, so the code can be
// compiled without the CUDA SDK.
#ifndef GPULITE_CUDA_API_HPP
#define GPULITE_CUDA_API_HPP

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif


// ========================================================================== //
//                       CUDA Driver API (from cuda.h)                        //
// ========================================================================== //
typedef int CUdevice;
typedef struct CUctx_st* CUcontext;
typedef struct CUmod_st* CUmodule;
typedef struct CUfunc_st* CUfunction;
typedef struct CUstream_st* CUstream;
typedef struct CUevent_st* CUevent;
typedef size_t CUdeviceptr;

typedef enum cudaError_enum {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE = 1,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_NOT_INITIALIZED = 3,
    // [...]
    CUDA_ERROR_UNKNOWN = 999
} CUresult;


typedef enum CUdevice_attribute_enum {
    // [...]
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK = 8,
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
    CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
    CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK = 12,
    // [...]
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    // [...]
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
    // [...]
    CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK = 111,
    // [...]
} CUdevice_attribute;

typedef enum CUfunction_attribute_enum {
    CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,
    CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1,
    CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2,
    CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3,
    CU_FUNC_ATTRIBUTE_NUM_REGS = 4,
    CU_FUNC_ATTRIBUTE_PTX_VERSION = 5,
    CU_FUNC_ATTRIBUTE_BINARY_VERSION = 6,
    CU_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7,
    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8,
    CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9,
    CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET = 10,
    CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH = 11,
    CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT = 12,
    CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH = 13,
    CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED = 14,
    CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE = 15,
} CUfunction_attribute;

typedef enum CUpointer_attribute_enum {
    CU_POINTER_ATTRIBUTE_CONTEXT = 1,
    CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2,
    CU_POINTER_ATTRIBUTE_DEVICE_POINTER = 3,
    CU_POINTER_ATTRIBUTE_HOST_POINTER = 4,
    CU_POINTER_ATTRIBUTE_P2P_TOKENS = 5,
    CU_POINTER_ATTRIBUTE_SYNC_MEMOPS = 6,
    CU_POINTER_ATTRIBUTE_BUFFER_ID = 7,
    CU_POINTER_ATTRIBUTE_IS_MANAGED = 8,
    CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9,
    CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE = 10,
    CU_POINTER_ATTRIBUTE_RANGE_START_ADDR = 11,
    CU_POINTER_ATTRIBUTE_RANGE_SIZE = 12,
    CU_POINTER_ATTRIBUTE_MAPPED = 13,
    CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES = 14,
    CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE = 15,
    CU_POINTER_ATTRIBUTE_ACCESS_FLAGS = 16,
    CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE = 17,
    CU_POINTER_ATTRIBUTE_MAPPING_SIZE = 18,
    CU_POINTER_ATTRIBUTE_MAPPING_BASE_ADDR = 19,
    CU_POINTER_ATTRIBUTE_MEMORY_BLOCK_ID = 20,
    CU_POINTER_ATTRIBUTE_IS_HW_DECOMPRESS_CAPABLE = 21
} CUpointer_attribute;

typedef enum CUmemorytype_enum {
    CU_MEMORYTYPE_HOST    = 0x01,
    CU_MEMORYTYPE_DEVICE  = 0x02,
    CU_MEMORYTYPE_ARRAY   = 0x03,
    CU_MEMORYTYPE_UNIFIED = 0x04
} CUmemorytype;

typedef enum CUjit_option_enum {
    CU_JIT_MAX_REGISTERS = 0,
    CU_JIT_THREADS_PER_BLOCK = 1,
    CU_JIT_WALL_TIME = 2,
    CU_JIT_INFO_LOG_BUFFER = 3,
    CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4,
    CU_JIT_ERROR_LOG_BUFFER = 5,
    CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6,
    CU_JIT_OPTIMIZATION_LEVEL = 7,
    CU_JIT_TARGET_FROM_CUCONTEXT = 8,
    CU_JIT_TARGET = 9,
    CU_JIT_FALLBACK_STRATEGY = 10,
    CU_JIT_GENERATE_DEBUG_INFO = 11,
    CU_JIT_LOG_VERBOSE = 12,
    CU_JIT_GENERATE_LINE_INFO = 13,
    CU_JIT_CACHE_MODE = 14,
    CU_JIT_NEW_SM3X_OPT = 15,
    CU_JIT_FAST_COMPILE = 16,
    CU_JIT_GLOBAL_SYMBOL_NAMES = 17,
    CU_JIT_GLOBAL_SYMBOL_ADDRESSES = 18,
    CU_JIT_GLOBAL_SYMBOL_COUNT = 19,
    CU_JIT_LTO = 20,
    CU_JIT_FTZ = 21,
    CU_JIT_PREC_DIV = 22,
    CU_JIT_PREC_SQRT = 23,
    CU_JIT_FMA = 24,
    CU_JIT_REFERENCED_KERNEL_NAMES = 25,
    CU_JIT_REFERENCED_KERNEL_COUNT = 26,
    CU_JIT_REFERENCED_VARIABLE_NAMES = 27,
    CU_JIT_REFERENCED_VARIABLE_COUNT = 28,
    CU_JIT_OPTIMIZE_UNUSED_DEVICE_VARIABLES = 29,
    CU_JIT_POSITION_INDEPENDENT_CODE = 30,
    CU_JIT_MIN_CTA_PER_SM = 31,
    CU_JIT_MAX_THREADS_PER_BLOCK = 32,
    CU_JIT_OVERRIDE_DIRECTIVE_VALUES = 33,
    CU_JIT_SPLIT_COMPILE = 34,
    CU_JIT_NUM_OPTIONS
} CUjit_option;

// ========================================================================== //
//                 CUDA Runtime API (from cuda_runtime_api.h)                 //
// ========================================================================== //

struct dim3 {
    unsigned int x, y, z;
    dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
};

enum cudaError {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInitializationError = 3,
    // [...]
    cudaErrorNotReady = 600,
    // [...]
    cudaErrorUnknown = 999,
    cudaErrorApiFailureBase = 10000
};

typedef enum cudaError cudaError_t;
typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;

enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
};

enum cudaMemoryType {
      cudaMemoryTypeUnregistered = 0,
      cudaMemoryTypeHost = 1,
      cudaMemoryTypeDevice = 2,
      cudaMemoryTypeManaged = 3
};

struct cudaPointerAttributes {
    enum cudaMemoryType type;
    int device;
    void* devicePointer;
    void* hostPointer;
    long reserved[8];
};

#define cudaHostAllocDefault                0x00
#define cudaHostAllocPortable               0x01
#define cudaHostAllocMapped                 0x02
#define cudaHostAllocWriteCombined          0x04

#define cudaEventDefault                    0x00
#define cudaEventBlockingSync               0x01
#define cudaEventDisableTiming              0x02
#define cudaEventInterprocess               0x04

#define cudaEventWaitDefault                0x00
#define cudaEventWaitExternal               0x01

enum cudaDeviceAttr {
    cudaDevAttrMaxThreadsPerBlock             = 1,
    cudaDevAttrMaxBlockDimX                   = 2,
    cudaDevAttrMaxBlockDimY                   = 3,
    cudaDevAttrMaxBlockDimZ                   = 4,
    cudaDevAttrMaxGridDimX                    = 5,
    cudaDevAttrMaxGridDimY                    = 6,
    cudaDevAttrMaxGridDimZ                    = 7,
    cudaDevAttrMaxSharedMemoryPerBlock        = 8,
    cudaDevAttrTotalConstantMemory            = 9,
    cudaDevAttrWarpSize                       = 10,
    cudaDevAttrMaxPitch                       = 11,
    cudaDevAttrMaxRegistersPerBlock           = 12,
    cudaDevAttrClockRate                      = 13,
    cudaDevAttrTextureAlignment               = 14,
    cudaDevAttrGpuOverlap                     = 15,
    cudaDevAttrMultiProcessorCount            = 16,
    cudaDevAttrKernelExecTimeout              = 17,
    cudaDevAttrIntegrated                     = 18,
    cudaDevAttrCanMapHostMemory               = 19,
    cudaDevAttrComputeMode                    = 20,
    // [...]
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81,
    cudaDevAttrMaxRegistersPerMultiprocessor  = 82,
    cudaDevAttrManagedMemory                  = 83,
    // [...]
    cudaDevAttrCooperativeLaunch              = 95,
    // [...]
};

// ========================================================================== //
//                          NVRTC API (from nvrtc.h)                          //
// ========================================================================== //

typedef enum {
    NVRTC_SUCCESS = 0,
    NVRTC_ERROR_OUT_OF_MEMORY = 1,
    NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,
    NVRTC_ERROR_INVALID_INPUT = 3,
    NVRTC_ERROR_INVALID_PROGRAM = 4,
    NVRTC_ERROR_INVALID_OPTION = 5,
    NVRTC_ERROR_COMPILATION = 6,
    NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7,
    NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8,
    NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9,
    NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10,
    NVRTC_ERROR_INTERNAL_ERROR = 11,
    NVRTC_ERROR_TIME_FILE_WRITE_FAILED = 12,
    NVRTC_ERROR_NO_PCH_CREATE_ATTEMPTED = 13,
    NVRTC_ERROR_PCH_CREATE_HEAP_EXHAUSTED = 14,
    NVRTC_ERROR_PCH_CREATE = 15,
    NVRTC_ERROR_CANCELLED = 16,
    NVRTC_ERROR_TIME_TRACE_FILE_WRITE_FAILED = 17
} nvrtcResult;

typedef struct _nvrtcProgram* nvrtcProgram;


#ifdef __cplusplus
}
#endif

#endif // GPULITE_CUDA_API_HPP
