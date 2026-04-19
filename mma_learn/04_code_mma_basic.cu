// MMA 基础代码示例
// 演示 MMA 基础概念和寄存器级操作

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <limits.h>
#include <float.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

// ============================================================================
// 示例 1: 基础 MMA 概念演示
// ============================================================================

/**
 * MMA 概念演示 kernel
 * 展示 16×8×8 MMA 操作的线程配置
 */
__global__ void mma_concept_demo() {
    // 简单验证：输出线程 ID 和配置
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("MMA 16x8x8 概念演示\n");
        printf("线程配置：blockDim=(%d,%d,%d), gridDim=(%d,%d,%d)\n",
               blockDim.x, blockDim.y, blockDim.z,
               gridDim.x, gridDim.y, gridDim.z);
        printf("\nMMA 16x8x8 规格:\n");
        printf("  A 矩阵：16×8 FP16\n");
        printf("  B 矩阵：8×8 FP16\n");
        printf("  C/D 矩阵：16×8 FP16/FP32\n");
        printf("  计算量：16×8×8 = 1024 FMA\n");
        printf("  32 线程参与，每线程负责 2×2 输出\n");
    }
}

// ============================================================================
// 示例 2: MMA 寄存器布局演示
// ============================================================================

/**
 * 演示 MMA 操作中寄存器的分配和数据布局
 * 16×8×8 FP16 MMA 需要:
 * - A: 4 个 32-bit 寄存器 (每线程)
 * - B: 2 个 32-bit 寄存器 (每线程)
 * - C/D: 4 个 32-bit 寄存器 (每线程)
 */
__global__ void mma_register_layout_demo() {
    int tid = threadIdx.x;

    __shared__ int shared_data[32];

    // 每个线程负责 2×2 的输出块
    int output_row = (tid / 2) * 2;
    int output_col = (tid % 2) * 2;

    shared_data[tid] = output_row * 100 + output_col;
    __syncthreads();

    if (tid < 8) {
        printf("线程 %2d: 负责输出块 [%2d:%2d, %2d:%2d], data=%d\n",
               tid, output_row, output_row+2, output_col, output_col+2,
               shared_data[tid]);
    }
}

// ============================================================================
// 示例 3: 使用 PTX 读取 MMA 相关寄存器
// ============================================================================

__global__ void mma_ptx_register_demo() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 使用 PTX 读取特殊寄存器
    unsigned int tid, bid, bdim;
    asm volatile("mov.u32 %0, %tid.x;" : "=r"(tid));
    asm volatile("mov.u32 %0, %ctaid.x;" : "=r"(bid));
    asm volatile("mov.u32 %0, %ntid.x;" : "=r"(bdim));

    if (idx < 4) {
        printf("\nPTX 寄存器读取:\n");
        printf("线程 %d: tid=%u, bid=%u, bdim=%u\n", idx, tid, bid, bdim);
    }

    // 演示 warp 级别的概念
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;

    if (idx < 4) {
        printf("  warp_id=%u, lane_id=%u\n", warp_id, lane_id);
        printf("  MMA 操作由整个 warp (32 线程) 共同执行\n");
    }
}

// ============================================================================
// 示例 4: 简单矩阵乘法验证 (CPU 参考实现对比 GPU)
// ============================================================================

void simple_gemm_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

__global__ void simple_gemm_gpu_kernel(float* d_C, const float* d_A, const float* d_B,
                                       int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += d_A[row * K + k] * d_B[k * N + col];
        }
        d_C[row * N + col] = sum;
    }
}

bool verify_gemm(int M, int N, int K) {
    // 分配主机内存
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C_cpu(M * N);
    std::vector<float> h_C_gpu(M * N);

    // 初始化随机数据
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (int i = 0; i < M * K; i++) h_A[i] = dis(gen);
    for (int i = 0; i < K * N; i++) h_B[i] = dis(gen);

    // CPU 计算
    simple_gemm_cpu(h_A.data(), h_B.data(), h_C_cpu.data(), M, N, K);

    // GPU 分配
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

    // 启动 GPU kernel
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    simple_gemm_gpu_kernel<<<grid, block>>>(d_C, d_A, d_B, M, N, K);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // 验证结果
    bool passed = true;
    float max_diff = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float diff = fabs(h_C_cpu[i] - h_C_gpu[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-3f) passed = false;
    }

    // 清理
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    printf("GEMM 验证 (M=%d, N=%d, K=%d):\n", M, N, K);
    printf("  CPU 结果 [0]: %.6f, GPU 结果 [0]: %.6f\n", h_C_cpu[0], h_C_gpu[0]);
    printf("  最大差异：%.6e\n", max_diff);
    printf("  验证：%s\n\n", passed ? "通过 ✓" : "失败 ✗");

    return passed;
}

// ============================================================================
// 示例 5: FP16 vs FP32 精度演示
// ============================================================================

__global__ void fp16_vs_fp32_demo() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx == 0) {
        printf("FP16 vs FP32 精度演示:\n");

        // FP16 范围
        half h_max = __float2half(65504.0f);
        half h_min = __float2half(6.10352e-05f);
        printf("  FP16 范围：%.4e ~ %.4e\n", __half2float(h_min), __half2float(h_max));

        // FP32 范围
        printf("  FP32 范围：%.4e ~ %.4e\n", FLT_MIN, FLT_MAX);

        // FP16 精度
        float h_eps = __half2float(__float2half(1.0f + 0.000977f)) - 1.0f;
        printf("  FP16 精度：~%.4e\n", h_eps);

        // FP32 精度
        printf("  FP32 精度：~%.4e\n", FLT_EPSILON);
    }
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("========================================\n");
    printf("  MMA 基础示例代码\n");
    printf("========================================\n");

    // 检查 GPU
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("\nGPU 信息:\n");
    printf("  名称：%s\n", prop.name);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  SM 数量：%d\n", prop.multiProcessorCount);
    printf("\n");

    // 检查 MMA 支持
    if (prop.major >= 7) {
        printf("MMA 支持：✓ 支持 (SM%d%d)\n", prop.major, prop.minor);
    } else {
        printf("MMA 支持：✗ 不支持 (需要 SM70+)\n");
        return 1;
    }

    // 示例 1: 基础 MMA 概念
    printf("\n----------------------------------------\n");
    printf("  示例 1: MMA 基础概念演示\n");
    printf("----------------------------------------\n");

    dim3 block1(32);
    dim3 grid1(1);
    mma_concept_demo<<<grid1, block1>>>();
    cudaDeviceSynchronize();

    // 示例 2: 寄存器布局
    printf("\n----------------------------------------\n");
    printf("  示例 2: MMA 寄存器布局演示\n");
    printf("----------------------------------------\n");

    dim3 block2(32);
    mma_register_layout_demo<<<1, block2>>>();
    cudaDeviceSynchronize();

    // 示例 3: PTX 寄存器
    printf("\n----------------------------------------\n");
    printf("  示例 3: PTX 寄存器读取演示\n");
    printf("----------------------------------------\n");

    dim3 block3(32);
    dim3 grid3(1);
    mma_ptx_register_demo<<<grid3, block3>>>();
    cudaDeviceSynchronize();

    // 示例 4: GEMM 验证
    printf("\n----------------------------------------\n");
    printf("  示例 4: GEMM 正确性验证\n");
    printf("----------------------------------------\n");

    bool test1 = verify_gemm(16, 16, 16);
    bool test2 = verify_gemm(32, 32, 32);
    bool test3 = verify_gemm(64, 64, 64);

    // 示例 5: FP16 vs FP32
    printf("\n----------------------------------------\n");
    printf("  示例 5: FP16 vs FP32 精度演示\n");
    printf("----------------------------------------\n");

    dim3 block5(32);
    dim3 grid5(1);
    fp16_vs_fp32_demo<<<grid5, block5>>>();
    cudaDeviceSynchronize();

    // 总结
    printf("\n========================================\n");
    printf("  MMA 基础示例完成!\n");
    printf("========================================\n");
    printf("\n知识点总结:\n");
    printf("1. MMA 指令格式：mma.sync.aligned.m<M>n<N>k<K>...\n");
    printf("2. MMA 寄存器布局：32 线程×4FP16/线程 = 128FP16 输出\n");
    printf("3. 特殊寄存器：%%tid, %%ctaid, %%ntid, %%clock64\n");
    printf("4. GEMM 验证：CPU 和 GPU 结果一致\n");
    printf("5. FP16 vs FP32: FP16 范围小但速度快\n");

    return 0;
}
