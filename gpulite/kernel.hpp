// gpulite: runtime compilation, caching and launching of CUDA and HIP kernels
#ifndef GPULITE_KERNEL_HPP
#define GPULITE_KERNEL_HPP

#include <cassert>
#include <cstddef>
#include <cstdlib>

#include <algorithm>
#include <array>
#include <atomic>
#include <fstream>
#include <initializer_list>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

#if defined(__linux__) || defined(__APPLE__)
    #include <unistd.h>  // for getcwd
#elif defined(_WIN32)
    #include <direct.h>  // for _getcwd
    #define getcwd _getcwd
#endif

#if defined(__GNUC__) || defined(__clang__)
#include <cxxabi.h>
#endif

#include "./cuda_api.hpp"
#include "./cuda_lib.hpp"
#include "./hip_api.hpp"
#include "./hip_lib.hpp"

namespace gpulite {

namespace details {

// Helper function to demangle the type name if necessary
inline std::string demangleTypeName(const std::string& name) {
#if defined(__GNUC__) || defined(__clang__)
    int status = 0;
    std::unique_ptr<char, void (*)(void*)> demangled_name(
        abi::__cxa_demangle(name.c_str(), 0, 0, &status), std::free
    );
    return (status == 0) ? demangled_name.get() : name;
#else
    // not ideal, but better than nothing on other compilers
    return name;
#endif
}

// Function to get type name of a single type
template <typename T> std::string typeName() {
    return demangleTypeName(typeid(T).name());
}

// Variadic template function to build type list
template <typename T, typename... Ts> void buildTemplateTypes(std::string& base) {
    base += typeName<T>(); // Add the first type
    // If there are more types, add a comma and recursively call for the remaining types
    if constexpr (sizeof...(Ts) > 0) {
        base += ", ";
        buildTemplateTypes<Ts...>(base); // Recursively call for the rest of the types
    }
}

// Helper function to start building the types
template <typename T, typename... Ts> std::string buildTemplateTypes() {
    std::string result;
    buildTemplateTypes<T, Ts...>(result); // Use recursive variadic template
    return result;
}

template <typename FuncType> struct function_traits;

template <typename Ret, typename... Args>
struct function_traits<Ret(Args...)> {
    using args_tuple = std::tuple<Args...>;
    using return_type = Ret;
};

template <typename From, typename To>
struct arg_compatible : std::is_same<From, To> {};

// Allow T* -> const T*
template <typename T>
struct arg_compatible<T*, const T*> : std::true_type {};

template <typename... Expected, typename... Actual>
struct arg_compatible<
    std::tuple<Expected...>,
    std::tuple<Actual...>>
    : std::bool_constant<
          sizeof...(Expected) == sizeof...(Actual) &&
          (arg_compatible<Actual, Expected>::value && ...)
      >
{};

template <typename Expected, typename Actual>
inline constexpr bool arg_compatible_v = arg_compatible<Expected, Actual>::value;

template <std::size_t ArgIndex, typename Expected, typename Actual>
struct argument_mismatch;

template <std::size_t ArgIndex, typename Expected, typename Actual>
constexpr void check_arg() {
    static_assert(
        arg_compatible<Actual, Expected>::value,
        "argument type mismatch (check first template parameter for argument index)"
    );
}

template <typename ExpectedTuple, typename ActualTuple, std::size_t... ArgIndices>
constexpr void check_tuple_impl(std::index_sequence<ArgIndices...>) {
    (check_arg<
        ArgIndices,
        std::tuple_element_t<ArgIndices, ExpectedTuple>,
        std::tuple_element_t<ArgIndices, ActualTuple>
    >(), ...);
}

template <typename FuncType, typename ...Actual>
constexpr void check_tuple_arguments() {
    using ExpectedTuple = typename details::function_traits<FuncType>::args_tuple;
    using ActualTuple = std::tuple<std::decay_t<Actual>...>;

    static_assert(std::tuple_size_v<ExpectedTuple> == std::tuple_size_v<ActualTuple>, "different number of arguments");

    check_tuple_impl<ExpectedTuple, ActualTuple>(std::make_index_sequence<std::tuple_size_v<ExpectedTuple>>{});
}

} // namespace details

// Base case: No template arguments, return function name without any type information
inline std::string getTemplateKernelName(const std::string& fn_name) {
    return fn_name;
}

/*
Function to get the kernel name with the list of templated types if any:
*/
template <typename T, typename... Ts> std::string getTemplateKernelName(const std::string& fn_name) {
    std::string type_list = details::buildTemplateTypes<T, Ts...>(); // Build type list
    return fn_name + "<" + type_list + ">"; // Return function name with type list in angle brackets
}

// =============================================================================
// Kernel Cache Manager - Runtime compilation and caching system
// =============================================================================

/// GPU API used to compile and launch kernels
enum class Api {
    /// NVIDIA CUDA, using NVRTC and the CUDA driver API
    CUDA,
    /// AMD HIP, using HIPRTC and the HIP module API
    HIP,
};

struct LaunchConfig {
    dim3 gridDim = dim3(1);
    dim3 blockDim = dim3(1);
    size_t dynamicSmemBytes = 0;
    /// Stream to launch the kernel on, either a `cudaStream_t`/`CUstream` or
    /// a `hipStream_t` depending on the API of the kernel.
    void* stream = nullptr;
};

/// Container class for the cached kernels. Provides functionality for launching
/// compiled kernels as well as automatically resizing dynamic shared memory
/// allocations, when needed. Kernels are compiled on first launch.
class CachedKernelBase {
public:
    CachedKernelBase(
        Api api,
        const std::string& kernel_name,
        const std::string& code,
        const std::string& path,
        std::vector<std::string> options
    ): CachedKernelBase(api, kernel_name, code.c_str(), path.c_str(), std::move(options)) {}

    CachedKernelBase(
        Api api,
        const std::string& kernel_name,
        const char* code,
        const char* path,
        std::vector<std::string> options
    ): api_(api), kernel_name_(kernel_name), options_(std::move(options)) {
        if (api_ == Api::CUDA) {
            if (!NVRTC::loaded() || !CUDADriver::loaded()) {
                throw std::runtime_error(
                    "could not load the NVRTC and CUDA driver libraries"
                );
            }
        } else {
            assert(api_ == Api::HIP);
            if (!HIPRTC::loaded() || !HIPRuntime::loaded()) {
                throw std::runtime_error(
                    "could not load the HIPRTC and HIP runtime libraries"
                );
            }
        }

        // Check if debug option is enabled
        this->debug_ = std::any_of(
            this->options_.cbegin(), this->options_.cend(),
            [&](const std::string& opt) {
                if (api_ == Api::CUDA) {
                    return opt == "-G" || opt == "--device-debug";
                } else {
                    assert(api_ == Api::HIP);
                    return opt == "-g";
                }
            }
        );

        // When debugging, write source to a real file so cuda-gdb/rocgdb can find it
        std::string effective_source_name;
        if (path) {
            effective_source_name = path;
        } else if (api_ == Api::CUDA) {
            effective_source_name = this->kernel_name_ + ".cu";
        } else {
            assert(api_ == Api::HIP);
            effective_source_name = this->kernel_name_ + ".hip";
        }

        if (this->debug_) {
            // Create a debug source file in the current working directory
            // Use absolute path so the debugger can reliably find it
            char cwd[4096];
            if (getcwd(cwd, sizeof(cwd)) != nullptr) {
                effective_source_name = std::string(cwd) + "/" + effective_source_name;
            }

            std::ofstream debug_source_file(effective_source_name);
            if (debug_source_file.is_open()) {
                debug_source_file << code;
                debug_source_file.close();
            } else {
                throw std::runtime_error(
                    "Failed to write debug source file: " + effective_source_name
                );
            }
        }

        if (api_ == Api::CUDA) {
            nvrtcProgram program = nullptr;
            GPULITE_NVRTC_CALL(nvrtcCreateProgram(
                &program, code, effective_source_name.c_str(), 0, nullptr, nullptr
            ));
            this->program_ = program;
            GPULITE_NVRTC_CALL(nvrtcAddNameExpression(program, this->kernel_name_.c_str()));
        } else {
            assert(api_ == Api::HIP);
            hiprtcProgram program = nullptr;
            GPULITE_HIPRTC_CALL(hiprtcCreateProgram(
                &program, code, effective_source_name.c_str(), 0, nullptr, nullptr
            ));
            this->program_ = program;
            GPULITE_HIPRTC_CALL(hiprtcAddNameExpression(program, this->kernel_name_.c_str()));
        }
    }

    virtual ~CachedKernelBase() {
        if (this->program_) {
            if (api_ == Api::CUDA) {
                auto program = static_cast<nvrtcProgram>(this->program_);
                NVRTC::instance().nvrtcDestroyProgram(&program);
            } else {
                assert(api_ == Api::HIP);
                auto program = static_cast<hiprtcProgram>(this->program_);
                HIPRTC::instance().hiprtcDestroyProgram(&program);
            }
        }
    }

    CachedKernelBase(const CachedKernelBase&) = delete;
    CachedKernelBase& operator=(const CachedKernelBase&) = delete;
    CachedKernelBase(CachedKernelBase&&) = delete;
    CachedKernelBase& operator=(CachedKernelBase&&) = delete;

    /// Get the API used by this kernel
    Api api() const {
        return api_;
    }

    /// Set a CUDA function attribute. Only available for CUDA kernels.
    void setFuncAttribute(CUfunction_attribute attribute, int value) const {
        this->checkApi(Api::CUDA, "setFuncAttribute(CUfunction_attribute)");
        GPULITE_CUDA_DRIVER_CALL(cuFuncSetAttribute(
            static_cast<CUfunction>(function_), attribute, value
        ));
    }

    /// Get a CUDA function attribute. Only available for CUDA kernels.
    int getFuncAttribute(CUfunction_attribute attribute) const {
        this->checkApi(Api::CUDA, "getFuncAttribute(CUfunction_attribute)");
        int value;
        GPULITE_CUDA_DRIVER_CALL(cuFuncGetAttribute(
            &value, attribute, static_cast<CUfunction>(function_)
        ));
        return value;
    }

    /// Get a HIP function attribute. Only available for HIP kernels.
    int getFuncAttribute(hipFunction_attribute attribute) const {
        this->checkApi(Api::HIP, "getFuncAttribute(hipFunction_attribute)");
        int value;
        GPULITE_HIP_CALL(hipFuncGetAttribute(
            &value, attribute, static_cast<hipFunction_t>(function_)
        ));
        return value;
    }

protected:
    /// Internal launch with void* args. Used by CachedKernel<FuncType>::launch.
    /// `args` points to `n_args` kernel argument pointers, owned by the caller
    /// (typically a stack array) for the duration of this call.
    void launchRaw(const LaunchConfig& config, void* const* args, size_t n_args) {

        if (!compiled_.load(std::memory_order_acquire)) {
            std::lock_guard<std::mutex> lock(compile_mutex_);
            if (!compiled_.load(std::memory_order_relaxed)) {
                if (api_ == Api::CUDA) {
                    this->compileKernelCuda(args, n_args);
                } else {
                    assert(api_ == Api::HIP);
                    this->compileKernelHip(args, n_args);
                }
            }
        }

        if (api_ == Api::CUDA) {
            CUcontext currentContext = nullptr;
            CUresult result = CUDADriver::instance().cuCtxGetCurrent(&currentContext);

            if (result != CUDA_SUCCESS || !currentContext) {
                throw std::runtime_error("CachedKernelBase::launch error getting current context.");
            }

            if (currentContext != context_) {
                GPULITE_CUDA_DRIVER_CALL(cuCtxSetCurrent(context_));
            }

            this->checkAndAdjustSharedMem(config.dynamicSmemBytes);

            GPULITE_CUDA_DRIVER_CALL(cuLaunchKernel(
                static_cast<CUfunction>(function_),
                config.gridDim.x,
                config.gridDim.y,
                config.gridDim.z,
                config.blockDim.x,
                config.blockDim.y,
                config.blockDim.z,
                config.dynamicSmemBytes,
                static_cast<CUstream>(config.stream),
                const_cast<void**>(args),
                nullptr
            ));

            if (currentContext != context_) {
                GPULITE_CUDA_DRIVER_CALL(cuCtxSetCurrent(currentContext));
            }
        } else {
            assert(api_ == Api::HIP);
            int currentDevice = -1;
            GPULITE_HIP_CALL(hipGetDevice(&currentDevice));

            if (currentDevice != device_) {
                GPULITE_HIP_CALL(hipSetDevice(device_));
            }

            this->checkAndAdjustSharedMem(config.dynamicSmemBytes);

            GPULITE_HIP_CALL(hipModuleLaunchKernel(
                static_cast<hipFunction_t>(function_),
                config.gridDim.x,
                config.gridDim.y,
                config.gridDim.z,
                config.blockDim.x,
                config.blockDim.y,
                config.blockDim.z,
                static_cast<unsigned int>(config.dynamicSmemBytes),
                static_cast<hipStream_t>(config.stream),
                const_cast<void**>(args),
                nullptr
            ));

            if (currentDevice != device_) {
                GPULITE_HIP_CALL(hipSetDevice(currentDevice));
            }
        }
    }

private:
    void checkApi(Api expected, const char* function) const {
        if (api_ != expected) {
            std::string api_name;
            if (expected == Api::CUDA) {
                api_name = "CUDA";
            } else {
                assert(expected == Api::HIP);
                api_name = "HIP";
            }
            throw std::runtime_error(
                std::string("CachedKernelBase::") + function + " is only available for " +
                api_name + " kernels"
            );
        }
    }

    /// The default shared memory space on most recent NVIDIA cards is 49152
    /// bytes. This method attempts to adjust the shared memory to fit the
    /// requested configuration if the kernel launch parameters exceeds the
    /// default 49152 bytes. AMD GPUs do not need to opt-in to use more shared
    /// memory, so we only check that the request fits on the device.
    void checkAndAdjustSharedMem(int query_shared_mem_size) {
        if (current_smem_size_ == 0) {
            if (api_ == Api::CUDA) {
                CUdevice cuDevice;
                GPULITE_CUDA_DRIVER_CALL(cuCtxGetDevice(&cuDevice));

                GPULITE_CUDA_DRIVER_CALL(cuDeviceGetAttribute(
                    &max_smem_size_optin_, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, cuDevice
                ));

                int reserved_smem_per_block = 0;

                GPULITE_CUDA_DRIVER_CALL(cuDeviceGetAttribute(
                    &reserved_smem_per_block, CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK, cuDevice
                ));

                int curr_max_smem_per_block = 0;

                GPULITE_CUDA_DRIVER_CALL(cuDeviceGetAttribute(
                    &curr_max_smem_per_block, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, cuDevice
                ));

                current_smem_size_ = (curr_max_smem_per_block - reserved_smem_per_block);
            } else {
                assert(api_ == Api::HIP);
                GPULITE_HIP_CALL(hipDeviceGetAttribute(
                    &current_smem_size_, hipDeviceAttributeMaxSharedMemoryPerBlock, device_
                ));
                max_smem_size_optin_ = current_smem_size_;
            }
        }

        if (query_shared_mem_size > current_smem_size_) {

            if (query_shared_mem_size > max_smem_size_optin_) {
                throw std::runtime_error(
                    "CachedKernelBase::launch requested more shared memory (smem) than available on card."
                );
            } else {
                // only reachable for CUDA, since for HIP `max_smem_size_optin_
                // == current_smem_size_`
                assert(api_ == Api::CUDA);
                GPULITE_CUDA_DRIVER_CALL(cuFuncSetAttribute(
                    static_cast<CUfunction>(function_),
                    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    query_shared_mem_size
                ));
                current_smem_size_ = query_shared_mem_size;
            }
        }
    }

    /// Compiles the kernel "kernel_name" located in source file "kernel_code",
    /// which additional parameters "options" passed to the NVRTC instance. Will
    /// auto-detect the compute capability of the available card. args for the
    /// launch need to be queried as we need to grab the CUcontext in which
    /// these ptrs exist.
    void compileKernelCuda(void* const* kernel_args, size_t n_args) {
        this->initCudaDriver();

        CUcontext currentContext = nullptr;

        for (size_t ptr_id = 0; ptr_id < n_args; ptr_id++) {
            unsigned int memtype = 0;
            CUdeviceptr device_ptr = *reinterpret_cast<CUdeviceptr*>(kernel_args[ptr_id]);

            CUresult res = CUDADriver::instance().cuPointerGetAttribute(
                &memtype, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, device_ptr
            );

            if (res == CUDA_SUCCESS && memtype == CU_MEMORYTYPE_DEVICE) {
                GPULITE_CUDA_DRIVER_CALL(cuPointerGetAttribute(
                    &currentContext, CU_POINTER_ATTRIBUTE_CONTEXT, device_ptr
                ));

                if (currentContext) {
                    break;
                }
            }
        }

        CUcontext query = nullptr;
        GPULITE_CUDA_DRIVER_CALL(cuCtxGetCurrent(&query));

        if (currentContext == nullptr) {
            // None of the arguments is tied to a context. This happens when
            // they are all scalars, or when the device memory comes from the
            // stream-ordered allocator (`cudaMallocAsync`) or from the virtual
            // memory management API, which are not tied to a specific context.
            // In this case, we use the current context.
            if (query == nullptr) {
                throw std::runtime_error(
                    "CachedKernelBase::compileKernel: could not find a CUDA context "
                    "from the kernel arguments, and there is no current CUDA context"
                );
            }
            currentContext = query;
        } else if (query != currentContext) {
            GPULITE_CUDA_DRIVER_CALL(cuCtxSetCurrent(currentContext));
        }

        CUdevice cuDevice;
        GPULITE_CUDA_DRIVER_CALL(cuCtxGetDevice(&cuDevice));

        std::vector<const char*> c_options;
        c_options.reserve(this->options_.size() + 1);
        for (const auto& option : this->options_) {
            c_options.push_back(option.c_str());
        }

        int major = 0;
        int minor = 0;
        GPULITE_CUDA_DRIVER_CALL(cuDeviceGetAttribute(
            &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice
        ));
        GPULITE_CUDA_DRIVER_CALL(cuDeviceGetAttribute(
            &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice
        ));
        int arch = major * 10 + minor;
        std::string smbuf = "--gpu-architecture=sm_" + std::to_string(arch);

        c_options.push_back(smbuf.c_str());

        auto program = static_cast<nvrtcProgram>(this->program_);
        nvrtcResult compileResult = NVRTC::instance().nvrtcCompileProgram(
            program, static_cast<int>(c_options.size()), c_options.data()
        );
        if (compileResult != NVRTC_SUCCESS) {
            size_t logSize;
            GPULITE_NVRTC_CALL(nvrtcGetProgramLogSize(program, &logSize));
            std::string log(logSize, '\0');
            GPULITE_NVRTC_CALL(nvrtcGetProgramLog(program, &log[0]));
            throw std::runtime_error(
                "CachedKernelBase::compileKernel: Failed to compile CUDA program:\n" + log
            );
        }

        // fetch CUBIN
        size_t cubinSize = 0;
        GPULITE_NVRTC_CALL(nvrtcGetCUBINSize(program, &cubinSize));
        std::vector<char> cubin(cubinSize);
        GPULITE_NVRTC_CALL(nvrtcGetCUBIN(program, cubin.data()));

        // load the module from cubin
        CUmodule module = nullptr;
        CUresult cuResult;

        if (this->debug_) {
            // Load with JIT debug info
            CUjit_option opts[1];
            opts[0] = CU_JIT_GENERATE_DEBUG_INFO;
            void* vals[1];
            vals[0] = (void*)(size_t)1;
            cuResult = CUDADriver::instance().cuModuleLoadDataEx(
                &module, cubin.data(), 1, opts, vals
            );
        } else {
            // Load without JIT options
            cuResult = CUDADriver::instance().cuModuleLoadDataEx(
                &module, cubin.data(), 0, 0, 0
            );
        }

        if (cuResult != CUDA_SUCCESS) {
            throw std::runtime_error(
                "CachedKernelBase::compileKernel: Failed to load CUBIN code into CUDA "
                "module (error code: " + std::to_string(cuResult) + ")"
            );
        }

        const char* lowered_name;
        GPULITE_NVRTC_CALL(nvrtcGetLoweredName(program, this->kernel_name_.c_str(), &lowered_name));
        CUfunction kernel;
        GPULITE_CUDA_DRIVER_CALL(cuModuleGetFunction(&kernel, module, lowered_name));

        this->module_ = module;
        this->function_ = kernel;
        this->context_ = currentContext;
        this->compiled_.store(true, std::memory_order_release);

        GPULITE_NVRTC_CALL(nvrtcDestroyProgram(&program));
        this->program_ = nullptr;
    }

    /// Compiles the kernel with HIPRTC for the device on which the kernel
    /// arguments live (or the current device if none of the arguments is a
    /// device pointer), and loads it as a module on this device.
    void compileKernelHip(void* const* kernel_args, size_t n_args) {
        GPULITE_HIP_CALL(hipInit(0));

        int previousDevice = -1;
        GPULITE_HIP_CALL(hipGetDevice(&previousDevice));

        int device = -1;
        for (size_t ptr_id = 0; ptr_id < n_args; ptr_id++) {
            unsigned int memtype = 0;
            hipDeviceptr_t device_ptr = *reinterpret_cast<hipDeviceptr_t*>(kernel_args[ptr_id]);

            hipError_t res = HIPRuntime::instance().hipPointerGetAttribute(
                &memtype, HIP_POINTER_ATTRIBUTE_MEMORY_TYPE, device_ptr
            );

            if (res == hipSuccess && memtype == static_cast<unsigned int>(hipMemoryTypeDevice)) {
                GPULITE_HIP_CALL(hipPointerGetAttribute(
                    &device, HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL, device_ptr
                ));

                if (device >= 0) {
                    break;
                }
            }
        }

        if (device < 0) {
            // None of the arguments is a device pointer, use the current device
            device = previousDevice;
        }

        if (device != previousDevice) {
            GPULITE_HIP_CALL(hipSetDevice(device));
        }

        std::vector<const char*> c_options;
        c_options.reserve(this->options_.size());
        for (const auto& option : this->options_) {
            c_options.push_back(option.c_str());
        }

        // Without an explicit `--offload-arch`, HIPRTC compiles for the
        // current device.
        auto program = static_cast<hiprtcProgram>(this->program_);
        hiprtcResult compileResult = HIPRTC::instance().hiprtcCompileProgram(
            program, static_cast<int>(c_options.size()), c_options.data()
        );
        if (compileResult != HIPRTC_SUCCESS) {
            size_t logSize = 0;
            GPULITE_HIPRTC_CALL(hiprtcGetProgramLogSize(program, &logSize));
            std::string log(logSize, '\0');
            GPULITE_HIPRTC_CALL(hiprtcGetProgramLog(program, &log[0]));
            throw std::runtime_error(
                "CachedKernelBase::compileKernel: Failed to compile HIP program:\n" + log
            );
        }

        // fetch the code object
        size_t codeSize = 0;
        GPULITE_HIPRTC_CALL(hiprtcGetCodeSize(program, &codeSize));
        std::vector<char> code(codeSize);
        GPULITE_HIPRTC_CALL(hiprtcGetCode(program, code.data()));

        hipModule_t module = nullptr;
        GPULITE_HIP_CALL(hipModuleLoadData(&module, code.data()));

        const char* lowered_name;
        GPULITE_HIPRTC_CALL(hiprtcGetLoweredName(program, this->kernel_name_.c_str(), &lowered_name));
        hipFunction_t kernel;
        GPULITE_HIP_CALL(hipModuleGetFunction(&kernel, module, lowered_name));

        this->module_ = module;
        this->function_ = kernel;
        this->device_ = device;
        this->compiled_.store(true, std::memory_order_release);

        GPULITE_HIPRTC_CALL(hiprtcDestroyProgram(&program));
        this->program_ = nullptr;

        if (device != previousDevice) {
            GPULITE_HIP_CALL(hipSetDevice(previousDevice));
        }
    }

    void initCudaDriver() {
        int deviceCount = 0;
        // Check if CUDA has already been initialized
        CUresult res = CUDADriver::instance().cuDeviceGetCount(&deviceCount);
        if (res == CUDA_ERROR_NOT_INITIALIZED) {
            // CUDA hasn't been initialized, so we initialize it now
            res = CUDADriver::instance().cuInit(0);
            if (res != CUDA_SUCCESS) {
                throw std::runtime_error(
                    "CachedKernelBase::initCudaDriver: Failed to initialize the CUDA driver."
                );
            }
        }
    }

    Api api_;
    std::string kernel_name_;

    int current_smem_size_ = 0;
    int max_smem_size_optin_ = 0;
    /// `CUmodule` or `hipModule_t`
    void* module_ = nullptr;
    /// `CUfunction` or `hipFunction_t`
    void* function_ = nullptr;
    /// `nvrtcProgram` or `hiprtcProgram`
    void* program_ = nullptr;
    /// CUDA context in which the kernel was loaded (CUDA only)
    CUcontext context_ = nullptr;
    /// Device on which the kernel was loaded (HIP only)
    int device_ = -1;
    std::atomic<bool> compiled_{false};
    bool debug_ = false;

    std::mutex compile_mutex_;
    std::vector<std::string> options_;
};


// =============================================================================
// Type-safe CachedKernel wrapper
// =============================================================================

/// Type-safe kernel wrapper. The template parameter FuncType is the function
/// signature of the kernel (e.g. void(float*, float*, int)). Provides a typed
/// launch() method that checks argument count at compile time.
template <typename FuncType>
class CachedKernel : public CachedKernelBase {
    static_assert(std::is_function_v<FuncType>,
                  "CachedKernel requires a function type (e.g. void(float*, int))");

public:
    using CachedKernelBase::CachedKernelBase;

    /// Launch the kernel with typed arguments. Options like grid/block/stream
    /// are passed in a LaunchConfig struct as the first argument.
    template <typename... Ts>
    void launch(const LaunchConfig& config, Ts... args) {
        details::check_tuple_arguments<FuncType, Ts...>();

        // Stack-allocated (size known at compile time from the parameter
        // pack), avoiding a heap allocation on every kernel launch.
        std::array<void*, sizeof...(Ts)> kernel_args = { static_cast<void*>(&args)... };
        launchRaw(config, kernel_args.data(), kernel_args.size());
    }
};


/// Factory class to create and store compiled CUDA or HIP kernels for caching
/// as a simple name-based map. Allows both compiling from a source file, or for
/// compiling from a variable containing CUDA/HIP code.
///
/// The pointers returned by `create()` and `getKernel()` stay valid for the
/// whole lifetime of the program, so callers that launch the same kernel many
/// times can look it up once and keep the pointer around.
class KernelFactory {
  public:
    KernelFactory(const KernelFactory&) = delete;
    KernelFactory& operator=(const KernelFactory&) = delete;
    KernelFactory(KernelFactory&&) = delete;
    KernelFactory& operator=(KernelFactory&&) = delete;

    /// Get the singleton instance of the KernelFactory for a given API and
    /// device. This ensures that each device has its own kernel cache. Kernels
    /// created by this factory are compiled and launched with the given `api`.
    static KernelFactory& instance(Api api, int device) {
        static std::mutex INSTANCES_MUTEX;
        static std::vector<std::unique_ptr<KernelFactory>> CUDA_INSTANCES;
        static std::vector<std::unique_ptr<KernelFactory>> HIP_INSTANCES;

        // Different threads can ask for different devices at the same time.
        // The factories are allocated separately and never removed, so the
        // references stay valid when the vector grows, and the lock is only
        // needed while looking up/adding elements.
        if (device < 0) {
            throw std::runtime_error("invalid device index: " + std::to_string(device));
        }

        std::lock_guard<std::mutex> lock(INSTANCES_MUTEX);
        std::vector<std::unique_ptr<KernelFactory>>* instances = nullptr;
        if (api == Api::CUDA) {
            instances = &CUDA_INSTANCES;
        } else {
            assert(api == Api::HIP);
            instances = &HIP_INSTANCES;
        }

        auto index = static_cast<size_t>(device);
        while (instances->size() < index + 1) {
            instances->push_back(std::unique_ptr<KernelFactory>(new KernelFactory(api)));
        }

        return *(*instances)[index];
    }

    /// Get the API used by the kernels in this factory
    Api api() const {
        return api_;
    }

    /// Check if a kernel with the given name exists in the cache.
    bool hasKernel(std::string_view kernel_name) {
        std::lock_guard<std::mutex> kernel_cache_lock(kernel_cache_mutex_);
        return kernel_cache_.find(kernel_name) != kernel_cache_.end();
    }

    /// Retrieve a previously cached kernel. FuncType must match the type used
    /// when the kernel was created.
    template <typename FuncType>
    CachedKernel<FuncType>* getKernel(std::string_view kernel_name) {
        static_assert(std::is_function_v<FuncType>,
                      "FuncType must be a function type (e.g. void(float*, int))");
        std::lock_guard<std::mutex> kernel_cache_lock(kernel_cache_mutex_);
        auto* kernel = this->findKernel<FuncType>(kernel_name);
        if (kernel == nullptr) {
            throw std::runtime_error("Kernel not found in cache: " + std::string(kernel_name));
        }
        return kernel;
    }

    /// Create or retrieve a kernel from inline source. If the kernel already
    /// exists in the cache, returns the cached instance. FuncType is the
    /// function signature of the kernel (e.g. void(float*, float*, int)).
    ///
    /// This overload is used when the options are given as a list of string
    /// literals (e.g. `{"-std=c++17"}`), and does not allocate any memory when
    /// the kernel is already in the cache.
    template <typename FuncType>
    CachedKernel<FuncType>* create(
        std::string_view kernel_name,
        const char* code,
        const char* path,
        std::initializer_list<const char*> options
    ) {
        return this->createImpl<FuncType>(kernel_name, code, path, [&]() {
            return std::vector<std::string>(options.begin(), options.end());
        });
    }

    /// Create or retrieve a kernel from inline source. If the kernel already
    /// exists in the cache, returns the cached instance. FuncType is the
    /// function signature of the kernel (e.g. void(float*, float*, int)).
    template <typename FuncType>
    CachedKernel<FuncType>* create(
        std::string_view kernel_name,
        const char* code,
        const char* path,
        std::vector<std::string> options
    ) {
        return this->createImpl<FuncType>(kernel_name, code, path, [&]() {
            return std::move(options);
        });
    }

    template <typename FuncType>
    CachedKernel<FuncType>* create(
        std::string_view kernel_name,
        const std::string& code,
        const std::string& path,
        std::vector<std::string> options
    ) {
        return this->create<FuncType>(kernel_name, code.c_str(), path.c_str(), std::move(options));
    }

  private:
    explicit KernelFactory(Api api): api_(api) {}

    /// Find the kernel with the given name in the cache, returning `nullptr` if
    /// it does not exist yet. `kernel_cache_mutex_` must be held by the caller.
    template <typename FuncType>
    CachedKernel<FuncType>* findKernel(std::string_view kernel_name) {
        auto it = kernel_cache_.find(kernel_name);
        if (it == kernel_cache_.end()) {
            return nullptr;
        }

        auto* ptr = dynamic_cast<CachedKernel<FuncType>*>(it->second.get());
        if (!ptr) {
            throw std::runtime_error(
                "Kernel type mismatch for '" + std::string(kernel_name) + "': requested "
                "type does not match the type used at creation"
            );
        }
        return ptr;
    }

    /// Implementation of `create()`. The options are only built (by calling
    /// `make_options()`) when the kernel is not in the cache yet.
    template <typename FuncType, typename MakeOptions>
    CachedKernel<FuncType>* createImpl(
        std::string_view kernel_name,
        const char* code,
        const char* path,
        MakeOptions&& make_options
    ) {
        static_assert(std::is_function_v<FuncType>, "FuncType must be a function type (e.g. void(float*, int))");
        std::lock_guard<std::mutex> kernel_cache_lock(kernel_cache_mutex_);
        auto* existing = this->findKernel<FuncType>(kernel_name);
        if (existing != nullptr) {
            return existing;
        }

        auto name = std::string(kernel_name);
        auto kernel = std::make_unique<CachedKernel<FuncType>>(
            api_, name, code, path, make_options()
        );
        auto* ptr = kernel.get();
        kernel_cache_.emplace(std::move(name), std::move(kernel));
        return ptr;
    }

    /// The kernels created on this device. This uses `std::less<>` to allow
    /// looking up kernels by `std::string_view` without creating a
    /// `std::string`.
    std::map<std::string, std::unique_ptr<CachedKernelBase>, std::less<>> kernel_cache_;
    /// The API used to compile and launch kernels from this factory
    Api api_;
    /// Protects `kernel_cache_`. Each device has its own mutex, so threads
    /// working with different devices do not wait for each other.
    std::mutex kernel_cache_mutex_;
};

} // namespace gpulite

#endif // GPULITE_KERNEL_HPP
