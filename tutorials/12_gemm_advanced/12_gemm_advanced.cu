/**
 * 第十二课：高级 GEMM 优化技术
 *
 * 本课讲解 CUTLASS 风格的高级 GEMM 优化：
 * 1. 多级分块
 * 2. 流水线隐藏延迟
 * 3. 向量化内存访问
 * 4. Warp 调度优化
 *
 * 编译：nvcc -std=c++17 -arch=sm_80 12_gemm_advanced.cu -o 12_gemm_advanced
 */

#include <iostream>
#include <cuda_runtime.h>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>

using namespace cute;

// ============================================================================
// GEMM 优化技术概览
// ============================================================================

void test_gemm_optimization_overview() {
    std::cout << "=== GEMM 优化技术概览 ===" << std::endl;

    std::cout << R"(
GEMM 优化层次:

Level 1 - 基础优化:
  - 共享内存分块
  - 合并内存访问
  - 避免 Bank Conflict

Level 2 - 中级优化:
  - Tensor Core (MMA)
  - 向量化加载 (ldmatrix)
  - 双缓冲

Level 3 - 高级优化:
  - 软件流水线
  - Warp 级分块
  - 指令调度

Level 4 - 极致优化:
  - 异步拷贝 (TMA)
  - Cluster 同步
  - 混合精度
    )" << std::endl;
}

// ============================================================================
// 多级分块策略
// ============================================================================

void test_multi_level_tiling() {
    std::cout << "\n=== 多级分块策略 ===" << std::endl;

    // 以 4096x4096 GEMM 为例
    constexpr int M = 4096, N = 4096, K = 4096;

    std::cout << "问题规模：" << M << "x" << N << "x" << K << std::endl;

    // 1. Block 级分块 (CTA Level)
    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 128;
    constexpr int BLOCK_K = 32;

    std::cout << "\nBlock 级分块:" << std::endl;
    std::cout << "  Tile 大小：" << BLOCK_M << "x" << BLOCK_N << std::endl;
    std::cout << "  Block 数量：" << (M/BLOCK_M) << "x" << (N/BLOCK_N) << std::endl;

    // 2. Warp 级分块
    constexpr int WARP_M = 64;
    constexpr int WARP_N = 64;

    std::cout << "\nWarp 级分块:" << std::endl;
    std::cout << "  每 Block Warps: " << (BLOCK_M/WARP_M) << "x" << (BLOCK_N/WARP_N) << std::endl;
    std::cout << "  每 Warp Tile: " << WARP_M << "x" << WARP_N << std::endl;

    // 3. Thread 级分块
    constexpr int THREAD_M = 16;
    constexpr int THREAD_N = 16;

    std::cout << "\nThread 级分块:" << std::endl;
    std::cout << "  每 Warp Threads: 32" << std::endl;
    std::cout << "  每 Thread 计算：" << THREAD_M << "x" << THREAD_N << std::endl;

    // 4. MMA 级分块
    std::cout << "\nMMA 级分块:" << std::endl;
    std::cout << "  Tensor Core: 16x8x8" << std::endl;
    std::cout << "  每 Thread MMA 次数：" << (THREAD_M/16) << "x" << (THREAD_N/8) << std::endl;
}

// ============================================================================
// 软件流水线
// ============================================================================

template<int STAGES, int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void pipelined_gemm_kernel(
    const half* A, const half* B, float* C,
    int M, int N, int K) {

    // 共享内存 - 多缓冲
    constexpr int SMEM_SIZE = BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N;
    __shared__ float smem[SMEM_SIZE * STAGES];

    int tid = threadIdx.x;
    int block_m = blockIdx.y;
    int block_n = blockIdx.x;

    // 计算全局偏移
    int global_m = block_m * BLOCK_M;
    int global_n = block_n * BLOCK_N;

    float acc[BLOCK_M * BLOCK_N / 32] = {0};  // 每线程累加器

    // 流水线
    int k_tile = (K + BLOCK_K - 1) / BLOCK_K;

    for (int k = 0; k < k_tile; k++) {
        int stage = k % STAGES;

        // 1. 异步加载 A 和 B 到共享内存
        // (简化为同步加载)
        int k_offset = k * BLOCK_K;
        for (int i = tid; i < BLOCK_M * BLOCK_K; i += BLOCK_M * BLOCK_N) {
            int m = i / BLOCK_K;
            int kk = i % BLOCK_K;
            if (global_m + m < M && k_offset + kk < K) {
                smem[stage * SMEM_SIZE + m * BLOCK_K + kk] =
                    __half2float(A[(global_m + m) * K + k_offset + kk]);
            }
        }
        for (int i = tid; i < BLOCK_K * BLOCK_N; i += BLOCK_M * BLOCK_N) {
            int kk = i / BLOCK_N;
            int n = i % BLOCK_N;
            if (k_offset + kk < K && global_n + n < N) {
                smem[stage * SMEM_SIZE + BLOCK_M * BLOCK_K + kk * BLOCK_N + n] =
                    __half2float(B[(k_offset + kk) * N + global_n + n]);
            }
        }

        __syncthreads();

        // 2. 计算
        for (int kk = 0; kk < BLOCK_K; kk++) {
            for (int m = 0; m < BLOCK_M; m++) {
                for (int n = 0; n < BLOCK_N; n++) {
                    float a = smem[stage * SMEM_SIZE + m * BLOCK_K + kk];
                    float b = smem[stage * SMEM_SIZE + BLOCK_M * BLOCK_K + kk * BLOCK_N + n];
                    acc[m * BLOCK_N + n] += a * b;
                }
            }
        }

        __syncthreads();
    }

    // 3. 写回结果
    for (int i = tid; i < BLOCK_M * BLOCK_N; i += BLOCK_M * BLOCK_N) {
        int m = i / BLOCK_N;
        int n = i % BLOCK_N;
        if (global_m + m < M && global_n + n < N) {
            C[(global_m + m) * N + global_n + n] = acc[i];
        }
    }
}

void test_pipelined_gemm() {
    std::cout << "\n=== 软件流水线 GEMM ===" << std::endl;

    constexpr int M = 256, N = 256, K = 256;
    constexpr int BLOCK_M = 64, BLOCK_N = 64, BLOCK_K = 32;
    constexpr int STAGES = 3;

    std::cout << "问题规模：" << M << "x" << N << "x" << K << std::endl;
    std::cout << "Block 大小：" << BLOCK_M << "x" << BLOCK_N << std::endl;
    std::cout << "流水线级数：" << STAGES << std::endl;

    // 分配内存
    size_t size_A = M * K * sizeof(half);
    size_t size_B = K * N * sizeof(half);
    size_t size_C = M * N * sizeof(float);

    half *d_A, *d_B;
    float *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // 初始化
    half* h_A = new half[M * K];
    half* h_B = new half[K * N];
    for (int i = 0; i < M * K; i++) h_A[i] = __float2half(0.5f);
    for (int i = 0; i < K * N; i++) h_B[i] = __float2half(2.0f);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // 启动 Kernel
    dim3 block(128);
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);

    pipelined_gemm_kernel<STAGES, BLOCK_M, BLOCK_N, BLOCK_K>
        <<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    // 验证
    float* h_C = new float[M * N];
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    std::cout << "结果验证：C[0] = " << h_C[0] << " (期望：" << K * 0.5 * 2.0 << ")" << std::endl;
    std::cout << "验证：" << (fabs(h_C[0] - K) < 0.1f ? "PASS" : "FAIL") << std::endl;

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// ============================================================================
// 向量化内存访问
// ============================================================================

void test_vectorized_memory() {
    std::cout << "\n=== 向量化内存访问 ===" << std::endl;

    std::cout << "向量化类型:" << std::endl;
    std::cout << "  float4:  128-bit 加载 (4x float)" << std::endl;
    std::cout << "  float2:   64-bit 加载 (2x float)" << std::endl;
    std::cout << "  uint128: 128-bit 整数加载" << std::endl;

    std::cout << "\n优势:" << std::endl;
    std::cout << "  - 减少指令数量" << std::endl;
    std::cout << "  - 提高内存吞吐量" << std::endl;
    std::cout << "  - 更好的指令级并行" << std::endl;

    // CUTE 向量化
    std::cout << "\nCUTE 向量化 API:" << std::endl;
    std::cout << "  - segmented_copy: 分段拷贝" << std::endl;
    std::cout << "  - tiled_copy: 分块拷贝" << std::endl;
}

// ============================================================================
// 性能调优技巧
// ============================================================================

void test_performance_tuning() {
    std::cout << "\n=== 性能调优技巧 ===" << std::endl;

    std::cout << "1. Block 大小选择:" << std::endl;
    std::cout << "   128x128: 适合小矩阵" << std::endl;
    std::cout << "   256x128: 适合大矩阵" << std::endl;
    std::cout << "   128x256: 适合 M 小 N 大" << std::endl;

    std::cout << "\n2. 流水线深度:" << std::endl;
    std::cout << "   2-stage: 寄存器压力小" << std::endl;
    std::cout << "   3-stage: 平衡点" << std::endl;
    std::cout << "   4+ stage: 需要更多调优" << std::endl;

    std::cout << "\n3. Warp 布局:" << std::endl;
    std::cout << "   4x4 Warps: 标准配置" << std::endl;
    std::cout << "   2x8 Warps: M 方向小" << std::endl;
    std::cout << "   8x2 Warps: N 方向小" << std::endl;

    std::cout << "\n4. 性能分析工具:" << std::endl;
    std::cout << "   - Nsight Compute: 内核分析" << std::endl;
    std::cout << "   - nvprof: 快速分析" << std::endl;
    std::cout << "   - cuda-gdb: 调试" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  第十二课：高级 GEMM 优化技术" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    test_gemm_optimization_overview();
    test_multi_level_tiling();
    test_pipelined_gemm();
    test_vectorized_memory();
    test_performance_tuning();

    std::cout << "\n========================================" << std::endl;
    std::cout << "  第十二课完成！" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
