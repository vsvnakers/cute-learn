/**
 * 第八课：WMMA 和 MMA 深入详解
 *
 * 本课深入讲解 Tensor Core 的两种 API：
 * 1. CUDA WMMA API - C 风格接口
 * 2. CUTE MMA Atom - C++ 模板接口
 * 3. PTX MMA 指令 - 底层汇编接口
 *
 * 编译：nvcc -std=c++17 -arch=sm_80 08_wmma_mma.cu -o 08_wmma_mma
 */

#include <iostream>
#include <cuda_runtime.h>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>

using namespace cute;

// ============================================================================
// 第一部分：Tensor Core 基础回顾
// ============================================================================

void test_tensor_core_basics() {
    std::cout << "=== 测试 1: Tensor Core 基础 ===" << std::endl;

    // Tensor Core 发展历程
    std::cout << "Tensor Core 架构演进:" << std::endl;
    std::cout << "  Volta (SM70):   4x4x4 FP16,  8 TFLOPS" << std::endl;
    std::cout << "  Turing (SM75):  8x8x4 FP16, 16 TFLOPS" << std::endl;
    std::cout << "  Ampere (SM80): 16x8x8 FP16, 64 TFLOPS" << std::endl;
    std::cout << "  Hopper (SM90): 16x8x16 FP16, 197 TFLOPS" << std::endl;

    // SM80 (Ampere) 支持的 MMA 指令
    std::cout << "\nSM80 支持的 MMA 指令:" << std::endl;
    std::cout << "  mma.sync.aligned.m16n8k8.f32.f16.f16.f32  (FP16)" << std::endl;
    std::cout << "  mma.sync.aligned.m16n8k16.f32.f16.f16.f32 (FP16)" << std::endl;
    std::cout << "  mma.sync.aligned.m16n8k8.f32.bf16.bf16.f32 (BF16)" << std::endl;
    std::cout << "  mma.sync.aligned.m8n8k16.s32.s8.s8.s32    (INT8)" << std::endl;

    std::cout << std::endl;
}

// ============================================================================
// 第二部分：CUDA WMMA API 理论讲解
// ============================================================================

void test_wmma_api() {
    std::cout << "=== 测试 2: CUDA WMMA API (理论) ===" << std::endl;

    // WMMA API 的关键函数：
    std::cout << "WMMA API 关键函数 (nvcuda::wmma):" << std::endl;
    std::cout << "  1. fragment<T> - 声明寄存器片段" << std::endl;
    std::cout << "  2. load_matrix_sync() - 加载矩阵到寄存器" << std::endl;
    std::cout << "  3. mma_sync() - 执行矩阵乘加" << std::endl;
    std::cout << "  4. store_matrix_sync() - 存储结果" << std::endl;
    std::cout << "  5. fill_fragment() - 初始化 fragment" << std::endl;

    // WMMA 的数据布局
    std::cout << "\nWMMA 数据布局选项:" << std::endl;
    std::cout << "  - mem_rowMajor: 行优先" << std::endl;
    std::cout << "  - mem_colMajor: 列优先" << std::endl;

    // WMMA 的注意事项
    std::cout << "\nWMMA 使用注意:" << std::endl;
    std::cout << "  1. 必须在 Warp 级别调用 (32 线程)" << std::endl;
    std::cout << "  2. 所有线程必须同步执行 load/mma/store" << std::endl;
    std::cout << "  3. 需要 __syncwarp() 同步" << std::endl;

    // WMMA 配置示例（仅理论）
    std::cout << "\nWMMA 配置示例:" << std::endl;
    std::cout << "  using namespace nvcuda::wmma;" << std::endl;
    std::cout << "  fragment<accumulator> c_frag;" << std::endl;
    std::cout << "  fragment<row_major, half> a_frag;" << std::endl;
    std::cout << "  fragment<col_major, half> b_frag;" << std::endl;
    std::cout << "  load_matrix_sync(a_frag, A_ptr, lda);" << std::endl;
    std::cout << "  mma_sync(c_frag, a_frag, b_frag, c_frag);" << std::endl;
    std::cout << "  store_matrix_sync(C_ptr, c_frag, ldc, mem_rowMajor);" << std::endl;

    std::cout << std::endl;
}

// ============================================================================
// 第三部分：CUTE MMA Atom API
// ============================================================================

void test_cute_mma_atom() {
    std::cout << "=== 测试 3: CUTE MMA Atom API ===" << std::endl;

    // CUTE 提供了更高级的 MMA 抽象

    // 1. 定义 MMA 配置
    using MMA_SM80_FP16 = MMA_Atom<SM80_16x8x8_F32F16F16F32_TN>;

    std::cout << "CUTE MMA Atom 配置:" << std::endl;
    std::cout << "  名称：SM80_16x8x8_F32F16F16F32_TN" << std::endl;
    std::cout << "  含义：SM80, 16x8x8, FP32=FP16*FP16+FP32, Transpose A, No Transpose B" << std::endl;

    // 2. 获取 MMA 形状
    auto mma_shape = MMA_SM80_FP16::Shape_MNK{};
    std::cout << "\nMMA 形状 (M,N,K):" << std::endl;
    std::cout << "  M = " << get<0>(mma_shape) << std::endl;
    std::cout << "  N = " << get<1>(mma_shape) << std::endl;
    std::cout << "  K = " << get<2>(mma_shape) << std::endl;

    // 3. 寄存器布局说明 (理论)
    // 注意：实际 CUTE API 中，寄存器布局需要通过其他方式获取
    // 以下是概念说明：
    std::cout << "\n寄存器布局 (概念说明):" << std::endl;
    std::cout << "  A 矩阵：16x8 FP16 -> 8 个寄存器 (打包)" << std::endl;
    std::cout << "  B 矩阵：8x8 FP16  -> 4 个寄存器 (打包)" << std::endl;
    std::cout << "  C/D矩阵：16x8 FP32 -> 16 个寄存器 (结果)" << std::endl;

    // 4. CUTE 实际 API 使用 (简化演示)
    std::cout << "\nCUTE 实际使用:" << std::endl;
    std::cout << "  // 创建 MMA descriptor" << std::endl;
    std::cout << "  auto mma_atom = MMA_Atom<SM80_16x8x8_F32F16F16F32_TN>{};" << std::endl;
    std::cout << "  // 使用 mma_atom 进行计算..." << std::endl;

    std::cout << std::endl;
}

// ============================================================================
// 第四部分：PTX MMA 指令详解
// ============================================================================

void test_ptx_mma_instruction() {
    std::cout << "=== 测试 4: PTX MMA 指令详解 ===" << std::endl;

    // PTX MMA 指令格式
    std::cout << "PTX MMA 指令格式:" << std::endl;
    std::cout << R"(
mma.sync.aligned.m16n8k8.f32.f16.f16.f32
    {d0, d1, ..., d7},
    {a0, a1},
    {b0, b1},
    {c0, c1, ..., c7};

参数说明:
- m16n8k8: MMA 尺寸 (M=16, N=8, K=8)
- .f32.f16.f16.f32: 数据类型 (D=C=A*B+C)
- .aligned: 地址对齐要求
- .sync: 线程同步
    )" << std::endl;

    // 寄存器分配
    std::cout << "寄存器分配:" << std::endl;
    std::cout << "  A: 2 个 uint32 (打包 8 个 FP16)" << std::endl;
    std::cout << "  B: 2 个 uint32 (打包 8 个 FP16)" << std::endl;
    std::cout << "  C/D: 8 个 uint32 (8 个 FP32 结果)" << std::endl;

    std::cout << std::endl;
}

// ============================================================================
// 第五部分：实际的 MMA Kernel
// ============================================================================

/**
 * 使用 CUTE MMA Atom 的简单 GEMM
 */
template<int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void mma_gemm_kernel(
    const half* A, const half* B, float* C,
    int M, int N, int K) {

    using MMA_Atom = MMA_Atom<SM80_16x8x8_F32F16F16F32_TN>;

    // 共享内存
    __shared__ half As[BLOCK_M * BLOCK_K];
    __shared__ half Bs[BLOCK_K * BLOCK_N];

    // 线程索引
    int tid = threadIdx.x;
    int row = blockIdx.y * BLOCK_M + tid / BLOCK_N;
    int col = blockIdx.x * BLOCK_N + tid % BLOCK_N;

    // 加载数据到共享内存
    if (row < M && tid < BLOCK_M * BLOCK_K) {
        As[tid] = A[row * K + (tid % BLOCK_K)];
    }
    if (col < N && tid < BLOCK_K * BLOCK_N) {
        Bs[tid] = B[(tid / BLOCK_N) * N + col];
    }
    __syncthreads();

    // 简化的计算（实际应使用 MMA 指令）
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < BLOCK_K; k++) {
            acc += __half2float(As[row * BLOCK_K + k]) *
                   __half2float(Bs[k * BLOCK_N + (tid % BLOCK_N)]);
        }
        C[row * N + col] = acc;
    }
}

void test_mma_kernel() {
    std::cout << "=== 测试 5: MMA Kernel 演示 ===" << std::endl;

    // 参数
    constexpr int M = 64;
    constexpr int N = 64;
    constexpr int K = 64;

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
    for (int i = 0; i < M * K; i++) h_A[i] = __float2half(1.0f);
    for (int i = 0; i < K * N; i++) h_B[i] = __float2half(2.0f);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // 启动 Kernel
    dim3 block(128);
    dim3 grid((N + 63) / 64, (M + 63) / 64);

    mma_gemm_kernel<64, 64, 64><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    // 验证
    float* h_C = new float[M * N];
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    std::cout << "MMA GEMM 结果验证:" << std::endl;
    std::cout << "  C[0] = " << h_C[0] << " (期望：" << K * 1.0 * 2.0 << ")" << std::endl;
    std::cout << "  结果：" << (fabs(h_C[0] - K * 2.0) < 0.01 ? "PASS" : "FAIL") << std::endl;

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    std::cout << std::endl;
}

// ============================================================================
// 第六部分：MMA 流水线优化
// ============================================================================

void test_mma_pipeline() {
    std::cout << "=== 测试 6: MMA 流水线优化 ===" << std::endl;

    // 高性能 GEMM 使用流水线隐藏内存延迟
    std::cout << "MMA 流水线阶段:" << std::endl;
    std::cout << "  阶段 1: Global -> Shared (CP_ASYNC)" << std::endl;
    std::cout << "  阶段 2: Shared -> Register (LDMATRIX)" << std::endl;
    std::cout << "  阶段 3: Register -> MMA (mma.sync)" << std::endl;
    std::cout << "  阶段 4: Accumulate (累加)" << std::endl;

    // 双缓冲技术
    std::cout << "\n双缓冲技术:" << std::endl;
    std::cout << "  Buffer 0: 计算当前 Tile" << std::endl;
    std::cout << "  Buffer 1: 预取下一 Tile" << std::endl;
    std::cout << "  效果：计算和传输重叠" << std::endl;

    // 软件流水线
    std::cout << "\n软件流水线深度:" << std::endl;
    std::cout << "  2-stage: 简单，寄存器压力小" << std::endl;
    std::cout << "  3-stage: 平衡" << std::endl;
    std::cout << "  4-stage: 复杂，需要更多寄存器" << std::endl;

    std::cout << std::endl;
}

// ============================================================================
// 第七部分：MMA vs WMMA 性能对比
// ============================================================================

void test_mma_vs_wmma() {
    std::cout << "=== 测试 7: MMA vs WMMA 对比 ===" << std::endl;

    std::cout << "MMA (PTX/CUTE) vs WMMA (API) 对比:" << std::endl;
    std::cout << "\nWMMA API 优点:" << std::endl;
    std::cout << "  - 易于使用" << std::endl;
    std::cout << "  - 可移植性好" << std::endl;
    std::cout << "  - 编译器优化" << std::endl;
    std::cout << "缺点:" << std::endl;
    std::cout << "  - 灵活性较低" << std::endl;
    std::cout << "  - 无法精细控制" << std::endl;

    std::cout << "\nMMA (PTX/CUTE) 优点:" << std::endl;
    std::cout << "  - 完全控制" << std::endl;
    std::cout << "  - 极致优化" << std::endl;
    std::cout << "  - 支持高级特性" << std::endl;
    std::cout << "缺点:" << std::endl;
    std::cout << "  - 复杂度高" << std::endl;
    std::cout << "  - 需要手动管理" << std::endl;

    std::cout << "\n推荐使用:" << std::endl;
    std::cout << "  - 原型开发：WMMA API" << std::endl;
    std::cout << "  - 生产优化：CUTE MMA / PTX" << std::endl;

    std::cout << std::endl;
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  第八课：WMMA 和 MMA 深入详解" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    test_tensor_core_basics();
    test_wmma_api();
    test_cute_mma_atom();
    test_ptx_mma_instruction();
    test_mma_kernel();
    test_mma_pipeline();
    test_mma_vs_wmma();

    std::cout << "========================================" << std::endl;
    std::cout << "  第八课完成！" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
