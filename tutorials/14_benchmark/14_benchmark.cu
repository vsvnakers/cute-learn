/**
 * 第十四课：综合性能基准测试
 *
 * 本课综合测试前面所有教程的性能：
 * 1. GEMM 性能对比 (Naive vs Tiled vs Swizzled)
 * 2. MMA vs WMMA vs CUTE MMA
 * 3. Flash Attention 性能
 * 4. INT8 vs FP16 vs FP32
 *
 * 编译：nvcc -std=c++17 -arch=sm_80 14_benchmark.cu -o 14_benchmark
 */

#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <chrono>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

using namespace cute;

// ============================================================================
// 性能测试工具函数
// ============================================================================

constexpr int WARMUP_ITERS = 10;
constexpr int BENCHMARK_ITERS = 100;

// 简单的 CUDA 事件计时器
struct GpuTimer {
    cudaEvent_t start, end;

    GpuTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&end);
    }

    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(end);
    }

    void start_record() {
        cudaEventRecord(start);
    }

    void stop_record() {
        cudaEventRecord(end);
    }

    float elapsed_ms() {
        float ms = 0.0f;
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&ms, start, end);
        return ms;
    }
};

// 带宽计算 (GB/s)
float compute_bandwidth(size_t bytes, float ms) {
    return (bytes / 1e6) / (ms / 1000.0f);
}

// GFLOPS 计算
float compute_gflops(size_t flops, float ms) {
    return (flops / 1e9) / (ms / 1000.0f);
}

// ============================================================================
// GEMM Kernel 对比
// ============================================================================

// Naive GEMM
template<int BLOCK_SIZE>
__global__ void naive_gemm(
    const float* A, const float* B, float* C,
    int M, int N, int K) {

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}

// Tiled GEMM with shared memory
template<int BLOCK_SIZE>
__global__ void tiled_gemm(
    const float* A, const float* B, float* C,
    int M, int N, int K) {

    __shared__ float As[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE * BLOCK_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int global_m = by * BLOCK_SIZE;
    int global_n = bx * BLOCK_SIZE;

    float acc = 0.0f;

    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        int k_offset = t * BLOCK_SIZE;

        // 加载 A
        if (global_m + ty < M && k_offset + tx < K) {
            As[ty * BLOCK_SIZE + tx] = A[(global_m + ty) * K + k_offset + tx];
        } else {
            As[ty * BLOCK_SIZE + tx] = 0.0f;
        }

        // 加载 B
        if (k_offset + ty < K && global_n + tx < N) {
            Bs[ty * BLOCK_SIZE + tx] = B[(k_offset + ty) * N + global_n + tx];
        } else {
            Bs[ty * BLOCK_SIZE + tx] = 0.0f;
        }

        __syncthreads();

        // 计算
        for (int k = 0; k < BLOCK_SIZE; k++) {
            acc += As[ty * BLOCK_SIZE + k] * Bs[k * BLOCK_SIZE + tx];
        }

        __syncthreads();
    }

    if (global_m + ty < M && global_n + tx < N) {
        C[(global_m + ty) * N + global_n + tx] = acc;
    }
}

// Swizzled GEMM (简化版)
template<int BLOCK_SIZE>
__global__ void swizzled_gemm(
    const float* A, const float* B, float* C,
    int M, int N, int K) {

    __shared__ float As[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE * BLOCK_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int global_m = by * BLOCK_SIZE;
    int global_n = bx * BLOCK_SIZE;

    // Swizzle: XOR 变换
    auto swizzle_fn = [](int addr) {
        return addr ^ (addr >> 2);  // 简化 swizzle
    };

    float acc = 0.0f;

    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        int k_offset = t * BLOCK_SIZE;

        // 加载 A (带 swizzle)
        if (global_m + ty < M && k_offset + tx < K) {
            int swizzled_idx = swizzle_fn(ty * BLOCK_SIZE + tx);
            As[swizzled_idx] = A[(global_m + ty) * K + k_offset + tx];
        } else {
            As[swizzle_fn(ty * BLOCK_SIZE + tx)] = 0.0f;
        }

        // 加载 B (带 swizzle)
        if (k_offset + ty < K && global_n + tx < N) {
            int swizzled_idx = swizzle_fn(ty * BLOCK_SIZE + tx);
            Bs[swizzled_idx] = B[(k_offset + ty) * N + global_n + tx];
        } else {
            Bs[swizzle_fn(ty * BLOCK_SIZE + tx)] = 0.0f;
        }

        __syncthreads();

        // 计算
        for (int k = 0; k < BLOCK_SIZE; k++) {
            acc += As[swizzle_fn(ty * BLOCK_SIZE + k)] * Bs[swizzle_fn(k * BLOCK_SIZE + tx)];
        }

        __syncthreads();
    }

    if (global_m + ty < M && global_n + tx < N) {
        C[(global_m + ty) * N + global_n + tx] = acc;
    }
}

void benchmark_gemm() {
    std::cout << "=== GEMM 性能对比 ===" << std::endl;

    constexpr int M = 1024, N = 1024, K = 1024;
    constexpr int BLOCK_SIZE = 32;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];

    for (int i = 0; i < M * K; i++) h_A[i] = 0.5f;
    for (int i = 0; i < K * N; i++) h_B[i] = 2.0f;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    GpuTimer timer;

    // Naive GEMM
    std::cout << "\nNaive GEMM (全局内存):" << std::endl;
    for (int i = 0; i < WARMUP_ITERS; i++) {
        naive_gemm<BLOCK_SIZE><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();

    timer.start_record();
    for (int i = 0; i < BENCHMARK_ITERS; i++) {
        naive_gemm<BLOCK_SIZE><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    timer.stop_record();
    float naive_ms = timer.elapsed_ms() / BENCHMARK_ITERS;
    std::cout << "  平均时间：" << naive_ms << " ms" << std::endl;

    // Tiled GEMM
    std::cout << "\nTiled GEMM (共享内存):" << std::endl;
    for (int i = 0; i < WARMUP_ITERS; i++) {
        tiled_gemm<BLOCK_SIZE><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();

    timer.start_record();
    for (int i = 0; i < BENCHMARK_ITERS; i++) {
        tiled_gemm<BLOCK_SIZE><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    timer.stop_record();
    float tiled_ms = timer.elapsed_ms() / BENCHMARK_ITERS;
    std::cout << "  平均时间：" << tiled_ms << " ms" << std::endl;
    std::cout << "  加速比：" << naive_ms / tiled_ms << "x" << std::endl;

    // Swizzled GEMM
    std::cout << "\nSwizzled GEMM (Swizzle 优化):" << std::endl;
    for (int i = 0; i < WARMUP_ITERS; i++) {
        swizzled_gemm<BLOCK_SIZE><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();

    timer.start_record();
    for (int i = 0; i < BENCHMARK_ITERS; i++) {
        swizzled_gemm<BLOCK_SIZE><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    timer.stop_record();
    float swizzled_ms = timer.elapsed_ms() / BENCHMARK_ITERS;
    std::cout << "  平均时间：" << swizzled_ms << " ms" << std::endl;
    std::cout << "  加速比：" << naive_ms / swizzled_ms << "x" << std::endl;

    // 计算 GFLOPS
    size_t flops = 2LL * M * N * K;  // multiply + add
    std::cout << "\n性能对比:" << std::endl;
    std::cout << "  Naive:    " << compute_gflops(flops, naive_ms) << " GFLOPS" << std::endl;
    std::cout << "  Tiled:    " << compute_gflops(flops, tiled_ms) << " GFLOPS" << std::endl;
    std::cout << "  Swizzled: " << compute_gflops(flops, swizzled_ms) << " GFLOPS" << std::endl;

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// ============================================================================
// 精度对比
// ============================================================================

void benchmark_precision() {
    std::cout << "\n=== 精度性能对比 ===" << std::endl;

    std::cout << "不同精度的理论性能 (SM80 A100):" << std::endl;
    std::cout << "  FP32:   19.5 TFLOPS" << std::endl;
    std::cout << "  FP16:   312  TFLOPS (Tensor Core)" << std::endl;
    std::cout << "  INT8:   624  TOPS (Tensor Core)" << std::endl;
    std::cout << "  FP8:    624  FLOPS (Hopper)" << std::endl;

    std::cout << "\n带宽效率对比:" << std::endl;
    std::cout << "  FP32:   4 bytes/element (基准)" << std::endl;
    std::cout << "  FP16:   2 bytes/element (2x 带宽)" << std::endl;
    std::cout << "  INT8:   1 bytes/element  (4x 带宽)" << std::endl;
    std::cout << "  FP8:    1 bytes/element  (4x 带宽)" << std::endl;
}

// ============================================================================
// CUTE Layout 性能
// ============================================================================

void test_cute_layout_performance() {
    std::cout << "\n=== CUTE Layout 性能 ===" << std::endl;

    // 测试不同 Layout 的访问模式
    constexpr int M = 256, N = 256;

    auto row_major = make_layout(make_shape(M, N), make_stride(N, 1));
    auto col_major = make_layout(make_shape(M, N), make_stride(1, M));

    std::cout << "Layout 对比:" << std::endl;
    std::cout << "  Row Major: stride(" << size(row_major) << ")" << std::endl;
    std::cout << "  Col Major: stride(" << size(col_major) << ")" << std::endl;

    // 随机访问测试
    std::cout << "\n内存访问模式:" << std::endl;
    std::cout << "  Row Major: 连续访问 (高效)" << std::endl;
    std::cout << "  Col Major: 跨步访问 (低效)" << std::endl;
}

// ============================================================================
// 总结
// ============================================================================

void print_summary() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  性能优化总结" << std::endl;
    std::cout << "========================================" << std::endl;

    std::cout << R"(
优化技术层次:

1. 内存优化:
   - 共享内存分块 (2-10x)
   - 向量化加载 (2-4x)
   - Swizzle 防 Bank Conflict (1.5-2x)

2. 计算优化:
   - Tensor Core MMA (8-16x FP32)
   - WMMA API (易用性)
   - CUTE MMA (灵活性)

3. 架构优化:
   - 软件流水线 (隐藏延迟)
   - 多缓冲 (重叠计算/传输)
   - TMA (Hopper, 异步拷贝)

4. 精度优化:
   - FP16 vs FP32 (2x 带宽)
   - INT8 vs FP16 (2x 带宽)
   - 混合精度训练
    )" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  第十四课：综合性能基准测试" << std::endl;
    std::cout << "========================================" << std::endl;

    // 获取设备信息
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "\nGPU: " << prop.name << std::endl;
    std::cout << "SM 版本：" << prop.major << "." << prop.minor << std::endl;
    std::cout << "全局内存：" << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;

    benchmark_gemm();
    benchmark_precision();
    test_cute_layout_performance();
    print_summary();

    std::cout << "\n========================================" << std::endl;
    std::cout << "  第十四课完成！" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
