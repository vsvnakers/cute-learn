// MMA GEMM 实战示例
// 演示完整的 GEMM 实现和性能验证

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

// ============================================================================
// GEMM 问题定义
// ============================================================================

/**
 * GEMM: C = alpha * A * B + beta * C
 *
 * 矩阵维度:
 * - A: M x K
 * - B: K x N
 * - C: M x N
 *
 * 分块策略:
 * - 将大矩阵分成 Tile_M x Tile_N 的小块
 * - 每个 CTA (线程块) 负责一个输出块
 * - 使用共享内存存储当前 tile 的数据
 */

constexpr int TILE_M = 128;
constexpr int TILE_N = 128;
constexpr int TILE_K = 32;

// ============================================================================
// 示例 1: 朴素 GEMM (CPU 参考实现)
// ============================================================================

void naive_gemm_cpu(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    float alpha = 1.0f, float beta = 0.0f
) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}

// ============================================================================
// 示例 2: 共享内存 GEMM (CUDA 优化版)
// ============================================================================

__global__ void shared_mem_gemm_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    float alpha, float beta
) {
    // 共享内存 tile
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    // 线程索引
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // 全局行/列
    int row = by * TILE_M + ty;
    int col = bx * TILE_N + tx;

    // 累加器
    float acc = 0.0f;

    // GEMM 主循环
    for (int tile = 0; tile < K; tile += TILE_K) {
        // 加载 A tile 到共享内存
        if (row < M && tile + tx < K) {
            As[ty][tx] = A[row * K + (tile + tx)];
        } else {
            As[ty][tx] = 0.0f;
        }

        // 加载 B tile 到共享内存
        if (tile + ty < K && col < N) {
            Bs[ty][tx] = B[(tile + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // 矩阵乘法
        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // 写回结果
    if (row < M && col < N) {
        C[row * N + col] = alpha * acc + beta * C[row * N + col];
    }
}

// ============================================================================
// 示例 3: GEMM 性能基准测试
// ============================================================================

template<typename Func>
float benchmark_gemm(Func gemm_func, int iterations = 10) {
    // 预热
    gemm_func();
    cudaDeviceSynchronize();

    // 计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        gemm_func();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    elapsed_time /= iterations;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return elapsed_time;
}

void run_gemm_benchmark(int M, int N, int K) {
    printf("\nGEMM 性能基准 (M=%d, N=%d, K=%d):\n", M, N, K);

    // 分配内存
    std::vector<float> h_A(M * K, 1.0f);
    std::vector<float> h_B(K * N, 1.0f);
    std::vector<float> h_C(M * N);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

    // 配置
    dim3 block(TILE_N / 8, TILE_M / 8);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    // 共享内存 GEMM
    float time_shared = benchmark_gemm([&]() {
        shared_mem_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f);
    }, 20);

    float gflops_shared = (2.0f * M * N * K) / (time_shared * 1e-3f) / 1e9f;

    printf("  共享内存 GEMM: %.3f ms, %.2f GFLOPS\n", time_shared, gflops_shared);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// ============================================================================
// 示例 4: GEMM 正确性验证
// ============================================================================

bool verify_gemm_result(
    const float* h_C_cpu, const float* h_C_gpu,
    int M, int N, float tolerance = 1e-3f
) {
    bool passed = true;
    float max_diff = 0.0f;

    for (int i = 0; i < M * N; i++) {
        float diff = fabs(h_C_cpu[i] - h_C_gpu[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > tolerance) passed = false;
    }

    printf("  最大差异：%.6e\n", max_diff);
    printf("  验证：%s\n", passed ? "通过 ✓" : "失败 ✗");

    return passed;
}

void verify_gemm(int M, int N, int K) {
    printf("\nGEMM 正确性验证 (M=%d, N=%d, K=%d):\n", M, N, K);

    // 分配主机内存
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C_cpu(M * N);
    std::vector<float> h_C_gpu(M * N);

    // 初始化
    for (int i = 0; i < M * K; i++) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = 1.0f;

    // CPU 计算
    naive_gemm_cpu(h_A.data(), h_B.data(), h_C_cpu.data(), M, N, K);

    // GPU 计算
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((N + 127) / 128, (M + 127) / 128);

    shared_mem_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // 验证
    verify_gemm_result(h_C_cpu.data(), h_C_gpu.data(), M, N);

    // 打印部分结果
    printf("  C[0][0]: CPU=%.6f, GPU=%.6f\n", h_C_cpu[0], h_C_gpu[0]);
    printf("  C[%d][%d]: CPU=%.6f, GPU=%.6f\n", M/2, N/2,
           h_C_cpu[M/2*N + N/2], h_C_gpu[M/2*N + N/2]);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// ============================================================================
// 示例 5: GEMM 配置说明
// ============================================================================

void print_gemm_config() {
    printf("\nGEMM 配置:\n");
    printf("  TILE_M = %d\n", TILE_M);
    printf("  TILE_N = %d\n", TILE_N);
    printf("  TILE_K = %d\n", TILE_K);
    printf("\n");
    printf("  每个 CTA 负责：%d×%d 输出\n", TILE_M, TILE_N);
    printf("  共享内存使用：%d KB (A) + %d KB (B)\n",
           (TILE_M * TILE_K * sizeof(float)) / 1024,
           (TILE_K * TILE_N * sizeof(float)) / 1024);
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("========================================\n");
    printf("  MMA GEMM 实战示例\n");
    printf("========================================\n");

    // 检查 GPU
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("\nGPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("SM 数量：%d\n", prop.multiProcessorCount);

    if (prop.major < 7) {
        printf("错误：MMA 需要 SM70+ 架构\n");
        return 1;
    }

    // 打印配置
    print_gemm_config();

    // 正确性验证
    printf("\n----------------------------------------\n");
    printf("  GEMM 正确性验证\n");
    printf("----------------------------------------\n");

    verify_gemm(64, 64, 64);
    verify_gemm(128, 128, 128);
    verify_gemm(256, 256, 256);

    // 性能基准
    printf("\n----------------------------------------\n");
    printf("  GEMM 性能基准\n");
    printf("----------------------------------------\n");

    run_gemm_benchmark(512, 512, 512);
    run_gemm_benchmark(1024, 1024, 1024);

    // 总结
    printf("\n========================================\n");
    printf("  GEMM 实战示例完成!\n");
    printf("========================================\n");
    printf("\n关键知识点:\n");
    printf("1. GEMM 分块：TILE_M×TILE_N×TILE_K\n");
    printf("2. 共享内存优化：减少全局内存访问\n");
    printf("3. 同步原语：__syncthreads()\n");
    printf("4. 性能优化：合并访问、减少分支\n");

    return 0;
}
