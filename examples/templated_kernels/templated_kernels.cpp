#include <gpulite/gpulite.hpp>

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <type_traits>
#include <iomanip>
#include <thread>
#include <algorithm>
#include <numeric>

template<typename T>
void run_templated_example() {
    const int N = 1024 * 256;
    const size_t size = N * sizeof(T);

    // Initialize host data
    std::vector<T> h_input(N), h_output(N);

    // Fill with test data
    if constexpr (std::is_integral_v<T>) {
        for (int i = 0; i < N; i++) {
            h_input[i] = static_cast<T>(i % 100);
        }
    } else {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dist(1.0, 10.0);
        for (int i = 0; i < N; i++) {
            h_input[i] = dist(gen);
        }
    }

    // CUDA kernel template source - uses C++ template syntax
    std::string kernel_source = R"(
template<typename T>
__device__ T square(T x) {
    return x * x;
}

template<typename T>
__device__ T cube(T x) {
    return x * x * x;
}

template<typename T>
__global__ void process_array(T* input, T* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        auto val = input[idx];
        // Apply some mathematical operations
        output[idx] = square(val) + cube(val) / (val + 1);
    }
}
)";

    try {
        // Allocate device memory
        T *d_input, *d_output;
        GPULITE_CUDART_CALL(cudaMalloc(reinterpret_cast<void**>(&d_input), size));
        GPULITE_CUDART_CALL(cudaMalloc(reinterpret_cast<void**>(&d_output), size));

        // Copy input data to device
        GPULITE_CUDART_CALL(cudaMemcpy(d_input, h_input.data(), size, cudaMemcpyHostToDevice));

        // Create templated kernel name using the helper function
        std::string kernel_name = gpulite::getTemplateKernelName<T>("process_array");

        // Create and cache kernel
        auto& factory = gpulite::KernelFactory::instance(CUdevice(0));
        std::cout << "Compiling kernel: " << kernel_name << std::endl;

        auto compile_start = std::chrono::high_resolution_clock::now();
        auto* kernel = factory.create(
            kernel_name,                           // templated kernel name
            kernel_source,                         // kernel source code
            "templated_kernel.cu",                 // virtual source filename
            {"-std=c++17", "--use_fast_math"}      // compilation options
        );
        auto compile_end = std::chrono::high_resolution_clock::now();

        auto compile_time = std::chrono::duration_cast<std::chrono::milliseconds>(compile_end - compile_start);
        std::cout << "Kernel compiled in: " << compile_time.count() << " ms" << std::endl;

        // Launch configuration
        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

        // Prepare kernel arguments
        std::vector<void*> args = {&d_input, &d_output, const_cast<void*>(static_cast<const void*>(&N))};

        // Warmup runs
        std::cout << "Performing warmup runs..." << std::endl;
        for (int i = 0; i < 5; i++) {
            kernel->launch(dim3(blocksPerGrid), dim3(threadsPerBlock), 0, nullptr, args, true);
        }

        // Cooldown period
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Benchmark runs
        std::cout << "Running performance benchmark..." << std::endl;
        const int num_runs = 20;
        std::vector<double> execution_times;
        execution_times.reserve(num_runs);

        for (int run = 0; run < num_runs; run++) {
            auto kernel_start = std::chrono::high_resolution_clock::now();
            kernel->launch(
                dim3(blocksPerGrid),
                dim3(threadsPerBlock),
                0,
                nullptr,
                args,
                true
            );
            auto kernel_end = std::chrono::high_resolution_clock::now();

            auto kernel_time = std::chrono::duration_cast<std::chrono::microseconds>(kernel_end - kernel_start);
            execution_times.push_back(kernel_time.count());

            // Small cooldown between runs
            if (run < num_runs - 1) {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }

        // Calculate statistics
        std::sort(execution_times.begin(), execution_times.end());
        double min_time = execution_times.front();
        double max_time = execution_times.back();
        double median_time = execution_times[num_runs / 2];
        double avg_time = std::accumulate(execution_times.begin(), execution_times.end(), 0.0) / num_runs;

        std::cout << "Execution time statistics (μs):" << std::endl;
        std::cout << "  Min: " << std::fixed << std::setprecision(2) << min_time << " μs" << std::endl;
        std::cout << "  Max: " << max_time << " μs" << std::endl;
        std::cout << "  Median: " << median_time << " μs" << std::endl;
        std::cout << "  Average: " << avg_time << " μs" << std::endl;

        // Copy result back to host
        GPULITE_CUDART_CALL(cudaMemcpy(h_output.data(), d_output, size, cudaMemcpyDeviceToHost));

        // Verify results (spot check)
        bool success = true;
        for (int i = 0; i < std::min(10, N); i++) {
            T expected;
            if constexpr (std::is_integral_v<T>) {
                expected = h_input[i] * h_input[i] + (h_input[i] * h_input[i] * h_input[i]) / (h_input[i] + 1);
            } else {
                expected = h_input[i] * h_input[i] + (h_input[i] * h_input[i] * h_input[i]) / (h_input[i] + static_cast<T>(1.0));
            }

            T tolerance = std::is_integral_v<T> ? T(0) : static_cast<T>(1e-5);
            if (std::abs(h_output[i] - expected) > tolerance) {
                std::cout << "Error at index " << i << ": expected " << expected
                         << ", got " << h_output[i] << std::endl;
                success = false;
                break;
            }
        }

        if (success) {
            std::cout << "SUCCESS: templated kernel executed correctly!" << std::endl;

            // Performance metrics using different time estimates
            double bandwidth_avg = (2.0 * N * sizeof(T)) / (avg_time * 1e-6) / 1e9;
            double bandwidth_peak = (2.0 * N * sizeof(T)) / (min_time * 1e-6) / 1e9;
            std::cout << "Memory bandwidth (average): " << std::fixed << std::setprecision(2)
                      << bandwidth_avg << " GB/s" << std::endl;
            std::cout << "Memory bandwidth (peak): " << bandwidth_peak << " GB/s" << std::endl;
        } else {
            std::cout << "FAILURE: templated kernel produced incorrect results." << std::endl;
        }

        // Clean up
        GPULITE_CUDART_CALL(cudaFree(d_input));
        GPULITE_CUDART_CALL(cudaFree(d_output));

    } catch (const std::exception& e) {
        std::cerr << "Error in example: " << e.what() << std::endl;
    }
}

int main() {
    try {
        // Check if CUDA is available
        if (!gpulite::CUDADriver::loaded() || !gpulite::NVRTC::loaded() || !gpulite::CUDART::loaded()) {
            std::cout << "CUDA runtime libraries not available. Please install NVIDIA drivers." << std::endl;
            return 1;
        }

        std::cout << "Templated Kernels Example" << std::endl;
        std::cout << "This example demonstrates how to create type-specific kernels" << std::endl;
        std::cout << "for different data types using runtime compilation." << std::endl;

        // Run examples with different data types
        run_templated_example<float>();
        run_templated_example<double>();
        run_templated_example<int>();

        std::cout << "\n=== Advanced Template Example ===" << std::endl;

        // Example showing how to use the getKernelName helper for true C++ templates
        const char* template_kernel = R"(
template<typename T, int BLOCK_SIZE>
__global__ void reduction_sum(T* input, T* output, int n) {
    extern __shared__ T sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    // Load data into shared memory
    sdata[tid] = (i < n) ? input[i] : 0;
    __syncthreads();

    // Perform reduction
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s && i + s < n) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Explicit instantiation for float with BLOCK_SIZE=256
template __global__ void reduction_sum<float, 256>(float*, float*, int);

extern "C" __global__ void reduction_sum_float_256(float* input, float* output, int n) {
    reduction_sum<float, 256>(input, output, n);
}
)";

        // This demonstrates how you would handle complex templated kernels
        // by using explicit instantiation and wrapper functions
        std::cout << "Complex template kernels require explicit instantiation" << std::endl;
        std::cout << "and wrapper functions for runtime compilation." << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
