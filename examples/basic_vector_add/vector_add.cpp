#include <gpulite/gpulite.hpp>

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <thread>
#include <algorithm>
#include <numeric>

int main() {
    try {
        // Check if CUDA is available
        if (!gpulite::CUDADriver::loaded() || !gpulite::NVRTC::loaded() || !gpulite::CUDART::loaded()) {
            std::cout << "CUDA runtime libraries not available. Please install the CUDA SDK and drivers." << std::endl;
            return 1;
        }

        // Vector size
        const int N = 1024 * 1024;
        const size_t size = N * sizeof(float);

        std::cout << "Vector Addition Example" << std::endl;
        std::cout << "Vector size: " << N << " elements" << std::endl;

        // Initialize host vectors
        std::vector<float> h_a(N), h_b(N), h_c(N);

        // Fill vectors with random data
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

        for (int i = 0; i < N; i++) {
            h_a[i] = dist(gen);
            h_b[i] = dist(gen);
        }

        // CUDA kernel source code
        const char* kernel_source = R"(
extern "C" __global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
)";

        // Allocate device memory using the cleaner GPULITE_CUDART_CALL macro
        float *d_a, *d_b, *d_c;
        GPULITE_CUDART_CALL(cudaMalloc(reinterpret_cast<void**>(&d_a), size));
        GPULITE_CUDART_CALL(cudaMalloc(reinterpret_cast<void**>(&d_b), size));
        GPULITE_CUDART_CALL(cudaMalloc(reinterpret_cast<void**>(&d_c), size));

        // Copy data to device
        GPULITE_CUDART_CALL(cudaMemcpy(d_a, h_a.data(), size, cudaMemcpyHostToDevice));
        GPULITE_CUDART_CALL(cudaMemcpy(d_b, h_b.data(), size, cudaMemcpyHostToDevice));

        // Create and cache kernel
        auto& factory = gpulite::KernelFactory::instance(CUdevice(0));
        std::cout << "Compiling kernel..." << std::endl;

        auto start = std::chrono::high_resolution_clock::now();
        auto* kernel = factory.create(
            "vector_add",           // kernel name
            kernel_source,          // kernel source code
            "vector_add.cu",        // virtual source filename
            {"-std=c++17"}          // compilation options
        );
        auto end = std::chrono::high_resolution_clock::now();

        auto compile_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Kernel compiled in: " << compile_time.count() << " ms" << std::endl;

        // Launch configuration
        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

        std::cout << "Launching kernel with " << blocksPerGrid << " blocks of " << threadsPerBlock << " threads" << std::endl;

        // Prepare kernel arguments
        std::vector<void*> args = {&d_a, &d_b, &d_c, const_cast<void*>(static_cast<const void*>(&N))};

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
            start = std::chrono::high_resolution_clock::now();
            kernel->launch(
                dim3(blocksPerGrid),    // grid size
                dim3(threadsPerBlock),  // block size
                0,                      // shared memory size
                nullptr,                // stream (default)
                args,                   // kernel arguments
                true                    // synchronize after launch
            );
            end = std::chrono::high_resolution_clock::now();

            auto kernel_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
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
        GPULITE_CUDART_CALL(cudaMemcpy(h_c.data(), d_c, size, cudaMemcpyDeviceToHost));

        // Verify results
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

            // Calculate performance metrics using median time for best estimate
            double bandwidth_avg = (3.0 * N * sizeof(float)) / (avg_time * 1e-6) / 1e9;
            double bandwidth_peak = (3.0 * N * sizeof(float)) / (min_time * 1e-6) / 1e9;
            std::cout << "Memory bandwidth (average): " << std::fixed << std::setprecision(2) << bandwidth_avg << " GB/s" << std::endl;
            std::cout << "Memory bandwidth (peak): " << bandwidth_peak << " GB/s" << std::endl;
        } else {
            std::cout << "FAILURE: Results do not match expected values." << std::endl;
        }

        // Clean up device memory
        GPULITE_CUDART_CALL(cudaFree(d_a));
        GPULITE_CUDART_CALL(cudaFree(d_b));
        GPULITE_CUDART_CALL(cudaFree(d_c));

        return success ? 0 : 1;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
