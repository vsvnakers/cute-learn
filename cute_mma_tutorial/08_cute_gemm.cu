/**
 * ============================================================================
 * CuTe + MMA 教程 08: 完整 GEMM 实现
 * ============================================================================
 *
 * 本文件实现一个完整的 GEMM (General Matrix Multiplication):
 *   C = A * B + C
 *
 * 使用 CuTe 的 TiledMMA 实现，沿 K 维度循环:
 *   1. 创建全局内存 Tensor
 *   2. partition_fragment_A/B: 创建寄存器 fragment
 *   3. K-loop: 沿 K 维度迭代，每次处理 MMA_K 个 K 元素
 *   4. MMA: 使用 Tensor Core 计算矩阵乘法
 *
 * 编译：make 08_cute_gemm
 * 运行：./08_cute_gemm
 * NCU:  make ncu_08_cute_gemm
 */

#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/layout.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm80.hpp>

using namespace cute;

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)          \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

// ============================================================================
// GEMM 配置
// ============================================================================

using MMA_OpType = SM80_16x8x8_F32F16F16F32_TN;
using MMA = MMA_Atom<MMA_OpType>;

constexpr int MMA_M = size<0>(MMA::Shape_MNK{});
constexpr int MMA_N = size<1>(MMA::Shape_MNK{});
constexpr int MMA_K = size<2>(MMA::Shape_MNK{});
constexpr int NUM_THREADS = 32;

// ============================================================================
// GEMM Kernel
// ============================================================================
//
// TN 格式: A (M, K), B (N, K), C (M, N)
//
// K-loop 实现:
//   - 创建完整的 A, B, C Tensor (编译期 M, N 形状，运行时 K)
//   - 使用 partition_A/B 获取带 k_tiles 维度的分区视图
//   - 循环中通过 (_, _, _, k_tile) 索引每个 K-tile
//   - 使用指针偏移创建 K-slice 的 Tensor

__global__ void cute_gemm_kernel(
    const half* __restrict__ A_ptr,  // [M x K] Row-Major
    const half* __restrict__ B_ptr,  // [N x K] Row-Major (TN格式)
    float* __restrict__ C_ptr,       // [M x N] Row-Major, 累加器也是输出
    int K)                           // K维度大小 (运行时可变, K循环上限)
{
    auto tiled_mma = make_tiled_mma(MMA{});
    int tid = threadIdx.x;
    auto thr_mma = tiled_mma.get_slice(tid);  // 获取当前线程的MMA视图

    // ---- C: (MMA_M, MMA_N) 编译期形状 ----
    auto C = make_tensor(make_gmem_ptr(C_ptr),
                         make_shape(Int<MMA_M>{}, Int<MMA_N>{}),
                         make_stride(Int<MMA_N>{}, Int<1>{}));
    auto part_C = thr_mma.partition_C(C);     // 将C按线程分区
    auto frag_C = tiled_mma.make_fragment_C(part_C);  // 创建寄存器累加器
    clear(frag_C);

    // ---- A/B fragment: 用A_k/B_k的形状推导寄存器fragment布局 ----
    // stride=(K, 1): Row-Major, 跨行stride=K, 行内stride=1
    auto A_k = make_tensor(make_gmem_ptr(A_ptr),
                           make_shape(Int<MMA_M>{}, Int<MMA_K>{}),
                           make_stride(K, Int<1>{}));
    auto B_k = make_tensor(make_gmem_ptr(B_ptr),
                           make_shape(Int<MMA_N>{}, Int<MMA_K>{}),
                           make_stride(K, Int<1>{}));

    // 只用模板tensor的形状推导fragment布局, 实际数据在循环中加载
    auto frag_A = thr_mma.partition_fragment_A(A_k);
    auto frag_B = thr_mma.partition_fragment_B(B_k);

    // ---- K 循环: 沿K维度滑动, 每次步进MMA_K ----
    for (int k = 0; k < K; k += MMA_K) {
        // 指针偏移A_ptr+k创建当前K-slice (第k列开始的MMA_K列)
        auto A_k = make_tensor(make_gmem_ptr(A_ptr + k),
                               make_shape(Int<MMA_M>{}, Int<MMA_K>{}),
                               make_stride(K, Int<1>{}));
        auto B_k = make_tensor(make_gmem_ptr(B_ptr + k),
                               make_shape(Int<MMA_N>{}, Int<MMA_K>{}),
                               make_stride(K, Int<1>{}));

        // partition → copy → gemm
        cute::copy(thr_mma.partition_A(A_k), frag_A);
        cute::copy(thr_mma.partition_B(B_k), frag_B);
        cute::gemm(tiled_mma, frag_A, frag_B, frag_C);
    }

    // ---- 写回结果 ----
    cute::copy(frag_C, part_C);
}

// ============================================================================
// 验证函数
// ============================================================================

void verify_gemm(const float* C, const half* A, const half* B,
                 int M, int N, int K) {
    float max_err = 0.0f;
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float expected = 0.0f;
            for (int k = 0; k < K; k++) {
                expected += __half2float(A[m * K + k]) * __half2float(B[n * K + k]);
            }
            float actual = C[m * N + n];
            float err = fabsf(actual - expected);
            max_err = fmaxf(max_err, err);
        }
    }
    std::cout << "  最大误差: " << max_err << std::endl;
    if (max_err < 1e-3) {
        std::cout << "  验证通过!" << std::endl;
    } else {
        std::cout << "  验证失败!" << std::endl;
    }
}

// ============================================================================
// 测试用例
// ============================================================================

void test_gemm_small() {
    std::cout << "=== 测试 1: 小规模 GEMM (16x8x8) ===" << std::endl;
    std::cout << std::endl;

    const int M = 16, N = 8, K = 8;

    half h_A[16 * 8], h_B[8 * 8];
    float h_C[16 * 8];

    for (int i = 0; i < M * K; i++) h_A[i] = __float2half(1.0f);
    for (int i = 0; i < N * K; i++) h_B[i] = __float2half(1.0f);
    for (int i = 0; i < M * N; i++) h_C[i] = 0.0f;

    half *d_A, *d_B;
    float *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, N * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, N * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice));

    cute_gemm_kernel<<<1, NUM_THREADS>>>(d_A, d_B, d_C, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "  结果 (A 全 1, B 全 1, K=8):" << std::endl;
    for (int m = 0; m < M; m++) {
        std::cout << "    ";
        for (int n = 0; n < N; n++) {
            std::cout << std::setw(6) << std::fixed << std::setprecision(1) << h_C[m * N + n];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    verify_gemm(h_C, h_A, h_B, M, N, K);
    std::cout << std::endl;

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

void test_gemm_k32() {
    std::cout << "=== 测试 2: K-loop GEMM (16x8x32) ===" << std::endl;
    std::cout << std::endl;

    const int M = 16, N = 8, K = 32;

    half* h_A = new half[M * K];
    half* h_B = new half[N * K];
    float* h_C = new float[M * N];

    srand(42);
    for (int i = 0; i < M * K; i++) h_A[i] = __float2half((float)(rand() % 10) / 10.0f);
    for (int i = 0; i < N * K; i++) h_B[i] = __float2half((float)(rand() % 10) / 10.0f);
    for (int i = 0; i < M * N; i++) h_C[i] = 0.0f;

    half *d_A, *d_B;
    float *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, N * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, N * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice));

    cute_gemm_kernel<<<1, NUM_THREADS>>>(d_A, d_B, d_C, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    verify_gemm(h_C, h_A, h_B, M, N, K);
    std::cout << std::endl;

    std::cout << "  结果 (部分):" << std::endl;
    for (int m = 0; m < 4; m++) {
        std::cout << "    ";
        for (int n = 0; n < N; n++) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(3) << h_C[m * N + n];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

// ============================================================================
// GEMM 架构说明
// ============================================================================

void print_gemm_architecture() {
    std::cout << "=== CuTe GEMM 架构说明 ===" << std::endl;
    std::cout << std::endl;

    std::cout << "  MMA 配置:" << std::endl;
    std::cout << "    MMA_M = " << MMA_M << ", MMA_N = " << MMA_N << ", MMA_K = " << MMA_K << std::endl;
    std::cout << std::endl;

    std::cout << "  K-loop 实现:" << std::endl;
    std::cout << "    1. 创建 C 的 fragment (编译期形状)" << std::endl;
    std::cout << "    2. 创建 A_k, B_k 的 fragment 模板" << std::endl;
    std::cout << "    3. K-loop: 每次用指针偏移创建 K-slice" << std::endl;
    std::cout << "    4. partition + copy + gemm" << std::endl;
    std::cout << std::endl;

    std::cout << "  关键技巧:" << std::endl;
    std::cout << "    - 指针偏移: make_gmem_ptr(A_ptr + k)" << std::endl;
    std::cout << "    - 编译期形状: make_shape(Int<MMA_M>{}, Int<MMA_K>{})" << std::endl;
    std::cout << "    - 运行时 stride: make_stride(K, Int<1>{})" << std::endl;
    std::cout << std::endl;
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  CuTe + MMA 教程 08: 完整 GEMM" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    print_gemm_architecture();
    test_gemm_small();
    test_gemm_k32();

    std::cout << "========================================" << std::endl;
    std::cout << "  教程 08 完成！" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
