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
            std::cout << "CUDA runtime libraries not available. Please install NVIDIA drivers." << std::endl;
            return 1;
        }

        // Matrix dimensions (N x N matrices)
        const int N = 1024;
        const size_t size = N * N * sizeof(float);

        std::cout << "Matrix Multiplication Example" << std::endl;
        std::cout << "Matrix size: " << N << "x" << N << std::endl;

        // Initialize host matrices
        std::vector<float> h_a(N * N), h_b(N * N), h_c(N * N), h_c_ref(N * N);

        // Fill matrices with random data
        std::random_device rd;
        std::mt19937 gen(42); // Fixed seed for reproducible results
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        for (int i = 0; i < N * N; i++) {
            h_a[i] = dist(gen);
            h_b[i] = dist(gen);
            h_c[i] = 0.0f;
        }

        // CUDA kernel source code - optimized tiled matrix multiplication
        const char* kernel_source = R"(
#define TILE_SIZE 16

extern "C" __global__ void matrix_multiply(float* A, float* B, float* C, int N) {
    // Shared memory for tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Calculate the row and column of C that this thread will compute
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tile into shared memory
        int a_row = row;
        int a_col = t * TILE_SIZE + tx;
        int b_row = t * TILE_SIZE + ty;
        int b_col = col;

        if (a_row < N && a_col < N) {
            As[ty][tx] = A[a_row * N + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }

        if (b_row < N && b_col < N) {
            Bs[ty][tx] = B[b_row * N + b_col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial sum for this tile
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // Write result
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}
)";

        // CPU reference implementation for verification
        std::cout << "Computing CPU reference..." << std::endl;
        auto cpu_start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < N; k++) {
                    sum += h_a[i * N + k] * h_b[k * N + j];
                }
                h_c_ref[i * N + j] = sum;
            }
        }

        auto cpu_end = std::chrono::high_resolution_clock::now();
        auto cpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start);
        std::cout << "CPU computation time: " << cpu_time.count() << " ms" << std::endl;

        // Allocate device memory
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

        auto compile_start = std::chrono::high_resolution_clock::now();
        auto* kernel = factory.create(
            "matrix_multiply",                      // kernel name
            kernel_source,                          // kernel source code
            "matrix_multiply.cu",                   // virtual source filename
            {"-std=c++17", "--use_fast_math"}       // compilation options
        );
        auto compile_end = std::chrono::high_resolution_clock::now();

        auto compile_time = std::chrono::duration_cast<std::chrono::milliseconds>(compile_end - compile_start);
        std::cout << "Kernel compiled in: " << compile_time.count() << " ms" << std::endl;

        // Launch configuration - using 16x16 thread blocks to match TILE_SIZE
        const int TILE_SIZE = 16;
        dim3 blockSize(TILE_SIZE, TILE_SIZE);
        dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

        std::cout << "Launching kernel with " << gridSize.x << "x" << gridSize.y
                  << " blocks of " << blockSize.x << "x" << blockSize.y << " threads" << std::endl;

        // Prepare kernel arguments
        std::vector<void*> args = {&d_a, &d_b, &d_c, const_cast<void*>(static_cast<const void*>(&N))};

        // Warmup runs
        std::cout << "Performing warmup runs..." << std::endl;
        for (int i = 0; i < 5; i++) {
            kernel->launch(gridSize, blockSize, 0, nullptr, args, true);
        }

        // Cooldown period
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Benchmark runs
        std::cout << "Running performance benchmark..." << std::endl;
        const int num_runs = 20;
        std::vector<double> execution_times;
        execution_times.reserve(num_runs);

        for (int run = 0; run < num_runs; run++) {
            auto gpu_start = std::chrono::high_resolution_clock::now();
            kernel->launch(gridSize, blockSize, 0, nullptr, args, true);
            auto gpu_end = std::chrono::high_resolution_clock::now();

            auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start);
            execution_times.push_back(gpu_time.count());

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
        float max_error = 0.0f;

        for (int i = 0; i < N * N; i++) {
            float error = std::abs(h_c[i] - h_c_ref[i]);
            max_error = std::max(max_error, error);
            if (error > 1e-3) {  // Relaxed tolerance for floating-point arithmetic
                std::cout << "Error at index " << i << ": expected " << h_c_ref[i] << ", got " << h_c[i]
                         << " (error: " << error << ")" << std::endl;
                success = false;
                break;
            }
        }

        if (success) {
            std::cout << "SUCCESS: Matrix multiplication completed correctly!" << std::endl;
            std::cout << "Maximum error: " << std::scientific << std::setprecision(2) << max_error << std::endl;

            // Calculate performance metrics using different time estimates
            double gflops_avg = (2.0 * N * N * N) / (avg_time * 1e-6) / 1e9;
            double gflops_peak = (2.0 * N * N * N) / (min_time * 1e-6) / 1e9;
            double speedup_avg = (double)cpu_time.count() / (avg_time * 1e-3);
            double speedup_peak = (double)cpu_time.count() / (min_time * 1e-3);

            std::cout << std::fixed << std::setprecision(2);
            std::cout << "GPU Performance (average): " << gflops_avg << " GFLOPS" << std::endl;
            std::cout << "GPU Performance (peak): " << gflops_peak << " GFLOPS" << std::endl;
            std::cout << "Speedup over CPU (average): " << speedup_avg << "x" << std::endl;
            std::cout << "Speedup over CPU (peak): " << speedup_peak << "x" << std::endl;
        } else {
            std::cout << "FAILURE: Results do not match expected values." << std::endl;
            std::cout << "Maximum error: " << std::scientific << std::setprecision(2) << max_error << std::endl;
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
