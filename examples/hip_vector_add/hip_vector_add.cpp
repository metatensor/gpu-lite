#include <gpulite/gpulite.hpp>

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

// Exit code used to tell CTest that this example was skipped
constexpr int SKIP_RETURN_CODE = 77;

int main() {
    try {
        // Check if HIP is available
        if (!gpulite::HIPRuntime::loaded() || !gpulite::HIPRTC::loaded()) {
            std::cout << "HIP runtime libraries not available. Please install ROCm and the AMD GPU drivers." << std::endl;
            return SKIP_RETURN_CODE;
        }

        int deviceCount = 0;
        auto status = gpulite::HIPRuntime::instance().hipGetDeviceCount(&deviceCount);
        if (status != hipSuccess || deviceCount == 0) {
            std::cout << "No HIP device available." << std::endl;
            return SKIP_RETURN_CODE;
        }
        GPULITE_HIP_CALL(hipSetDevice(0));

        int major = 0, minor = 0;
        GPULITE_HIPRTC_CALL(hiprtcVersion(&major, &minor));
        std::cout << "HIP Vector Addition Example (HIPRTC " << major << "." << minor << ")" << std::endl;

        // Vector size
        const int N = 1024 * 1024;
        const size_t size = N * sizeof(float);
        std::cout << "Vector size: " << N << " elements" << std::endl;

        // Initialize host vectors with random data
        std::vector<float> h_a(N), h_b(N), h_c(N);
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
        for (int i = 0; i < N; i++) {
            h_a[i] = dist(gen);
            h_b[i] = dist(gen);
        }

        // HIP kernel source code. HIPRTC automatically provides the HIP
        // built-ins (blockIdx, threadIdx, ...), so no include is needed.
        const char* kernel_source = R"(
extern "C" __global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
)";

        // Create and cache the kernel. Compilation with HIPRTC happens on the
        // first launch, for the device holding the kernel arguments.
        auto& factory = gpulite::KernelFactory::instance(gpulite::Api::HIP, 0);

        // Kernel declaration matching the source passed to HIPRTC
        __global__ void vector_add(const float* a, const float* b, float* c, int n);

        auto* kernel = factory.create<decltype(vector_add)>(
            "vector_add",           // kernel name
            kernel_source,          // kernel source code
            "vector_add.hip",       // virtual source filename
            {"-std=c++17"}          // compilation options
        );

        // Allocate device memory and copy data to device
        float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
        GPULITE_HIP_CALL(hipMalloc(reinterpret_cast<void**>(&d_a), size));
        GPULITE_HIP_CALL(hipMalloc(reinterpret_cast<void**>(&d_b), size));
        GPULITE_HIP_CALL(hipMalloc(reinterpret_cast<void**>(&d_c), size));

        GPULITE_HIP_CALL(hipMemcpy(d_a, h_a.data(), size, hipMemcpyHostToDevice));
        GPULITE_HIP_CALL(hipMemcpy(d_b, h_b.data(), size, hipMemcpyHostToDevice));

        // Launch the kernel
        unsigned int threadsPerBlock = 256;
        unsigned int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        std::cout << "Launching kernel with " << blocksPerGrid << " blocks of " << threadsPerBlock << " threads" << std::endl;

        auto config = gpulite::LaunchConfig();
        config.gridDim = dim3(blocksPerGrid);
        config.blockDim = dim3(threadsPerBlock);
        kernel->launch(config, d_a, d_b, d_c, N);
        GPULITE_HIP_CALL(hipDeviceSynchronize());

        // Copy result back to host and verify
        GPULITE_HIP_CALL(hipMemcpy(h_c.data(), d_c, size, hipMemcpyDeviceToHost));

        std::cout << "Verifying results..." << std::endl;
        bool success = true;
        for (int i = 0; i < N; i++) {
            float expected = h_a[i] + h_b[i];
            if (std::abs(h_c[i] - expected) > 1e-5) {
                std::cout << "Error at index " << i << ": expected " << expected << ", got " << h_c[i] << std::endl;
                success = false;
                break;
            }
        }

        if (success) {
            std::cout << "SUCCESS: Vector addition completed correctly!" << std::endl;
        } else {
            std::cout << "FAILURE: Results do not match expected values." << std::endl;
        }

        // Clean up
        GPULITE_HIP_CALL(hipFree(d_a));
        GPULITE_HIP_CALL(hipFree(d_b));
        GPULITE_HIP_CALL(hipFree(d_c));

        return success ? 0 : 1;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
