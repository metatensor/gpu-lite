// gpulite: runtime compilation, caching and launching of CUDA kernels
#ifndef GPULITE_KERNEL_HPP
#define GPULITE_KERNEL_HPP

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
// CUDA Kernel Cache Manager - Runtime compilation and caching system
// =============================================================================

struct LaunchConfig {
    dim3 gridDim = dim3(1);
    dim3 blockDim = dim3(1);
    size_t dynamicSmemBytes = 0;
    cudaStream_t stream = nullptr;
};

/// Container class for the cached kernels. Provides functionality for launching
/// compiled kernels as well as automatically resizing dynamic shared memory
/// allocations, when needed. Kernels are compiled on first launch.
class CachedKernelBase {
public:
    CachedKernelBase(
        const std::string& kernel_name,
        const std::string& code,
        const std::string& path,
        std::vector<std::string> options
    ): CachedKernelBase(kernel_name, code.c_str(), path.c_str(), std::move(options)) {}

    CachedKernelBase(
        const std::string& kernel_name,
        const char* code,
        const char* path,
        std::vector<std::string> options
    ): kernel_name_(kernel_name), options_(std::move(options)) {
        // Check if debug option is enabled
        this->debug_ = std::any_of(
            this->options_.cbegin(), this->options_.cend(),
            [](const std::string& opt) {
                return opt == "-G" || opt == "--device-debug";
            }
        );

        // When debugging, write source to a real file so cuda-gdb can find it
        std::string effective_source_name = path ? path : (this->kernel_name_ + std::string(".cu"));
        if (this->debug_) {
            // Create a debug source file in the current working directory
            // Use absolute path so cuda-gdb can reliably find it
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

        GPULITE_NVRTC_CALL(nvrtcCreateProgram(
            &this->program_, code, effective_source_name.c_str(), 0, nullptr, nullptr
        ));

        GPULITE_NVRTC_CALL(nvrtcAddNameExpression(this->program_, this->kernel_name_.c_str()));
    }

    virtual ~CachedKernelBase() {
        if (this->program_) {
            NVRTC::instance().nvrtcDestroyProgram(&this->program_);
        }
    }

    CachedKernelBase(const CachedKernelBase&) = delete;
    CachedKernelBase& operator=(const CachedKernelBase&) = delete;
    CachedKernelBase(CachedKernelBase&&) = delete;
    CachedKernelBase& operator=(CachedKernelBase&&) = delete;

    inline void setFuncAttribute(CUfunction_attribute attribute, int value) const {
        GPULITE_CUDA_DRIVER_CALL(cuFuncSetAttribute(function_, attribute, value));
    }

    int getFuncAttribute(CUfunction_attribute attribute) const {
        int value;
        GPULITE_CUDA_DRIVER_CALL(cuFuncGetAttribute(&value, attribute, function_));
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
                this->compileKernel(args, n_args);
            }
        }

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
            function_,
            config.gridDim.x,
            config.gridDim.y,
            config.gridDim.z,
            config.blockDim.x,
            config.blockDim.y,
            config.blockDim.z,
            config.dynamicSmemBytes,
            config.stream,
            const_cast<void**>(args),
            nullptr
        ));

        if (currentContext != context_) {
            GPULITE_CUDA_DRIVER_CALL(cuCtxSetCurrent(currentContext));
        }
    }

private:
    /// The default shared memory space on most recent NVIDIA cards is 49152
    /// bytes. This method attempts to adjust the shared memory to fit the
    /// requested configuration if the kernel launch parameters exceeds the
    /// default 49152 bytes.
    void checkAndAdjustSharedMem(int query_shared_mem_size) {
        if (current_smem_size_ == 0) {
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
        }

        if (query_shared_mem_size > current_smem_size_) {

            if (query_shared_mem_size > max_smem_size_optin_) {
                throw std::runtime_error(
                    "CachedKernelBase::launch requested more smem than available on card."
                );
            } else {
                GPULITE_CUDA_DRIVER_CALL(cuFuncSetAttribute(
                    function_, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, query_shared_mem_size
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
    void compileKernel(void* const* kernel_args, size_t n_args) {
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
        c_options.reserve(this->options_.size());
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

        nvrtcResult compileResult = NVRTC::instance().nvrtcCompileProgram(this->program_, c_options.size(), c_options.data());
        if (compileResult != NVRTC_SUCCESS) {
            size_t logSize;
            GPULITE_NVRTC_CALL(nvrtcGetProgramLogSize(this->program_, &logSize));
            std::string log(logSize, '\0');
            GPULITE_NVRTC_CALL(nvrtcGetProgramLog(this->program_, &log[0]));
            throw std::runtime_error(
                "KernelFactory::compileAndCacheKernel: Failed to compile CUDA program:\n" + log
            );
        }

        // fetch CUBIN
        size_t cubinSize = 0;
        GPULITE_NVRTC_CALL(nvrtcGetCUBINSize(this->program_, &cubinSize));
        std::vector<char> cubin(cubinSize);
        GPULITE_NVRTC_CALL(nvrtcGetCUBIN(this->program_, cubin.data()));

        // load the module from cubin
        CUmodule module = nullptr;
        CUresult cuResult;

        if (this->debug_) {
            // Load with JIT debug info
            CUjit_option opts[1];
            opts[0] = CU_JIT_GENERATE_DEBUG_INFO;
            void** vals = new void*[1];
            vals[0] = (void*)(size_t)1;
            cuResult = CUDADriver::instance().cuModuleLoadDataEx(
                &module, cubin.data(), 1, opts, vals
            );
            delete[] vals;
        } else {
            // Load without JIT options
            cuResult = CUDADriver::instance().cuModuleLoadDataEx(
                &module, cubin.data(), 0, 0, 0
            );
        }

        if (cuResult != CUDA_SUCCESS) {
            throw std::runtime_error(
                "KernelFactory::compileAndCacheKernel: Failed to load PTX code into CUDA "
                "module "
                "(error code: " +
                std::to_string(cuResult) + ")"
            );
        }

        const char* lowered_name;
        GPULITE_NVRTC_CALL(nvrtcGetLoweredName(this->program_, this->kernel_name_.c_str(), &lowered_name));
        CUfunction kernel;
        GPULITE_CUDA_DRIVER_CALL(cuModuleGetFunction(&kernel, module, lowered_name));

        this->module_ = module;
        this->function_ = kernel;
        this->context_ = currentContext;
        this->compiled_.store(true, std::memory_order_release);

        GPULITE_NVRTC_CALL(nvrtcDestroyProgram(&this->program_));
        this->program_ = nullptr;
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
                    "KernelFactory::initCudaDriver: Failed to initialize CUDA CUDA_DRIVER_INSTANCE."
                );
                return;
            }
        }
    }

    std::string kernel_name_;

    int current_smem_size_ = 0;
    int max_smem_size_optin_ = 0;
    CUmodule module_ = nullptr;
    CUfunction function_ = nullptr;
    CUcontext context_ = nullptr;
    nvrtcProgram program_ = nullptr;
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

    /// Launch the kernel with typed arguments. Options like grid/block/sync
    /// are passed in a cudaLaunchConfig_t struct as the first argument.
    template <typename... Ts>
    void launch(const LaunchConfig& config, Ts... args) {
        details::check_tuple_arguments<FuncType, Ts...>();

        // Stack-allocated (size known at compile time from the parameter
        // pack), avoiding a heap allocation on every kernel launch.
        std::array<void*, sizeof...(Ts)> kernel_args = { static_cast<void*>(&args)... };
        launchRaw(config, kernel_args.data(), kernel_args.size());
    }
};


/// Factory class to create and store compiled cuda kernels for caching as a
/// simple name-based map. Allows both compiling from a source file, or for
/// compiling from a variable containing CUDA code.
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

    /// Get the singleton instance of the KernelFactory for a given CUDA device.
    /// This ensures that each CUDA device has its own kernel cache.
    static KernelFactory& instance(CUdevice device) {
        static std::mutex INSTANCES_MUTEX;
        static std::vector<std::unique_ptr<KernelFactory>> INSTANCES;

        // Different threads can ask for different devices at the same time.
        // The factories are allocated separately and never removed, so the
        // references stay valid when the vector grows, and the lock is only
        // needed while looking up/adding elements.
        if (device < 0) {
            throw std::runtime_error("invalid CUDA device index: " + std::to_string(device));
        }

        std::lock_guard<std::mutex> lock(INSTANCES_MUTEX);
        auto index = static_cast<size_t>(device);
        while (INSTANCES.size() < index + 1) {
            INSTANCES.push_back(std::unique_ptr<KernelFactory>(new KernelFactory()));
        }

        return *INSTANCES[index];
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
    KernelFactory() {}

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
            name, code, path, make_options()
        );
        auto* ptr = kernel.get();
        kernel_cache_.emplace(std::move(name), std::move(kernel));
        return ptr;
    }

    /// The kernels created on this device. This uses `std::less<>` to allow
    /// looking up kernels by `std::string_view` without creating a
    /// `std::string`.
    std::map<std::string, std::unique_ptr<CachedKernelBase>, std::less<>> kernel_cache_;
    /// Protects `kernel_cache_`. Each device has its own mutex, so threads
    /// working with different devices do not wait for each other.
    std::mutex kernel_cache_mutex_;
};

} // namespace gpulite

#endif // GPULITE_KERNEL_HPP
