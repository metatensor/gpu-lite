#include <gpulite/gpulite.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

const char* kernel_source = R"(
extern "C" __global__ void scale_array(float* data, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Do some work to make the kernel non-trivial
        float val = data[idx];
        for (int i = 0; i < 100; i++) {
            val = val * scale + 0.001f;
        }
        data[idx] = val;
    }
}
)";

int main() {
    std::cout << "Async Operations Example" << std::endl;
    std::cout << "========================" << std::endl;
    std::cout << "Demonstrates overlapping memory transfers with kernel execution\n" << std::endl;

    try {
        // Check CUDA availability
        if (!gpulite::CUDADriver::loaded() || !gpulite::NVRTC::loaded() || !gpulite::CUDART::loaded()) {
            std::cerr << "Error: CUDA runtime libraries not available" << std::endl;
            return 1;
        }

        const int N = 1024 * 1024;  // 1M elements per chunk
        const int NUM_CHUNKS = 4;
        const int TOTAL_SIZE = N * NUM_CHUNKS;
        const size_t chunk_bytes = N * sizeof(float);
        const float scale = 1.0001f;

        std::cout << "Total elements: " << TOTAL_SIZE << std::endl;
        std::cout << "Chunks: " << NUM_CHUNKS << " x " << N << " elements" << std::endl;

        // Allocate pinned host memory for async transfers
        float* h_data;
        GPULITE_CUDART_CALL(cudaHostAlloc(
            reinterpret_cast<void**>(&h_data),
            TOTAL_SIZE * sizeof(float),
            cudaHostAllocDefault
        ));

        // Initialize host data
        for (int i = 0; i < TOTAL_SIZE; i++) {
            h_data[i] = static_cast<float>(i % 1000) * 0.001f;
        }

        // Allocate device memory (double buffer for overlapping)
        float *d_data[2];
        GPULITE_CUDART_CALL(cudaMalloc(reinterpret_cast<void**>(&d_data[0]), chunk_bytes));
        GPULITE_CUDART_CALL(cudaMalloc(reinterpret_cast<void**>(&d_data[1]), chunk_bytes));

        // Create streams for async operations
        cudaStream_t streams[2];
        GPULITE_CUDART_CALL(cudaStreamCreate(&streams[0]));
        GPULITE_CUDART_CALL(cudaStreamCreate(&streams[1]));

        // Compile kernel
        auto& factory = gpulite::KernelFactory::instance(CUdevice(0));
        std::cout << "\nCompiling kernel..." << std::endl;
        auto* kernel = factory.create(
            "scale_array",
            kernel_source,
            "async_kernel.cu",
            {"-std=c++17"}
        );
        std::cout << "Kernel compiled successfully" << std::endl;

        // Launch configuration
        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

        // ========================================
        // Synchronous execution (baseline)
        // ========================================
        std::cout << "\n--- Synchronous Execution (baseline) ---" << std::endl;

        auto sync_start = std::chrono::high_resolution_clock::now();

        for (int chunk = 0; chunk < NUM_CHUNKS; chunk++) {
            float* h_chunk = h_data + chunk * N;

            // Copy to device (blocking)
            GPULITE_CUDART_CALL(cudaMemcpy(d_data[0], h_chunk, chunk_bytes, cudaMemcpyHostToDevice));

            // Execute kernel
            std::vector<void*> args = {&d_data[0], const_cast<float*>(&scale), const_cast<int*>(&N)};
            kernel->launch(dim3(blocksPerGrid), dim3(threadsPerBlock), 0, nullptr, args, true);

            // Copy back (blocking)
            GPULITE_CUDART_CALL(cudaMemcpy(h_chunk, d_data[0], chunk_bytes, cudaMemcpyDeviceToHost));
        }

        auto sync_end = std::chrono::high_resolution_clock::now();
        auto sync_time = std::chrono::duration_cast<std::chrono::microseconds>(sync_end - sync_start).count();
        std::cout << "Synchronous time: " << sync_time << " us" << std::endl;

        // Reset data for async test
        for (int i = 0; i < TOTAL_SIZE; i++) {
            h_data[i] = static_cast<float>(i % 1000) * 0.001f;
        }

        // ========================================
        // Asynchronous execution with overlap
        // ========================================
        std::cout << "\n--- Asynchronous Execution (overlapped) ---" << std::endl;

        auto async_start = std::chrono::high_resolution_clock::now();

        for (int chunk = 0; chunk < NUM_CHUNKS; chunk++) {
            int buf = chunk % 2;
            cudaStream_t stream = streams[buf];
            float* h_chunk = h_data + chunk * N;

            // Async copy to device
            GPULITE_CUDART_CALL(cudaMemcpyAsync(
                d_data[buf], h_chunk, chunk_bytes,
                cudaMemcpyHostToDevice, stream
            ));

            // Execute kernel on stream
            std::vector<void*> args = {&d_data[buf], const_cast<float*>(&scale), const_cast<int*>(&N)};
            kernel->launch(
                dim3(blocksPerGrid), dim3(threadsPerBlock), 0,
                reinterpret_cast<void*>(stream), args, false
            );

            // Async copy back
            GPULITE_CUDART_CALL(cudaMemcpyAsync(
                h_chunk, d_data[buf], chunk_bytes,
                cudaMemcpyDeviceToHost, stream
            ));
        }

        // Wait for all operations to complete
        GPULITE_CUDART_CALL(cudaStreamSynchronize(streams[0]));
        GPULITE_CUDART_CALL(cudaStreamSynchronize(streams[1]));

        auto async_end = std::chrono::high_resolution_clock::now();
        auto async_time = std::chrono::duration_cast<std::chrono::microseconds>(async_end - async_start).count();
        std::cout << "Asynchronous time: " << async_time << " us" << std::endl;

        // Results
        std::cout << "\n--- Results ---" << std::endl;
        double speedup = static_cast<double>(sync_time) / async_time;
        std::cout << "Speedup: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;

        if (speedup > 1.0) {
            std::cout << "SUCCESS: Async operations provided performance improvement!" << std::endl;
        } else {
            std::cout << "Note: No speedup observed (kernel may be too short or memory-bound)" << std::endl;
        }

        // Cleanup
        GPULITE_CUDART_CALL(cudaStreamDestroy(streams[0]));
        GPULITE_CUDART_CALL(cudaStreamDestroy(streams[1]));
        GPULITE_CUDART_CALL(cudaFree(d_data[0]));
        GPULITE_CUDART_CALL(cudaFree(d_data[1]));
        GPULITE_CUDART_CALL(cudaFreeHost(h_data));

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
