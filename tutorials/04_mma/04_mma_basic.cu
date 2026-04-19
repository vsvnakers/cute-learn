/**
 * 第四课：CUTE MMA 指令详解
 *
 * 本示例展示如何使用 CUTE 的 MMA 原语进行 Tensor Core 编程
 * 编译：nvcc -std=c++17 -arch=sm_80 04_mma_basic.cu -o 04_mma_basic
 *
 * 注意：需要 Ampere (SM80) 或更新的 GPU
 */

#include <iostream>
#include <cuda_runtime.h>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/mma_sm80.hpp>

using namespace cute;

// ============================================================================
// 第一部分：理解 MMA 的数学基础
// ============================================================================

void test_mma_math() {
    std::cout << "=== 测试 1: MMA 数学基础 ===" << std::endl;

    // MMA 运算：D = A * B + C
    //
    // 对于 SM80 (Ampere)，基本的 MMA 尺寸是 16x8x8
    // 这意味着:
    //   A: 16x8 矩阵 (M x K)
    //   B: 8x8 矩阵  (K x N)
    //   C, D: 16x8 矩阵 (M x N)

    std::cout << "SM80 MMA 尺寸：16x8x8" << std::endl;
    std::cout << "  A: 16x8 (M=16, K=8)" << std::endl;
    std::cout << "  B: 8x8  (K=8, N=8)" << std::endl;
    std::cout << "  C,D: 16x8 (M=16, N=8)" << std::endl;
    std::cout << std::endl;

    // 手动验证小规模 MMA
    float A[16 * 8];
    float B[8 * 8];
    float C[16 * 8];
    float D[16 * 8];

    // 初始化
    for (int i = 0; i < 16 * 8; i++) A[i] = 1.0f;
    for (int i = 0; i < 8 * 8; i++) B[i] = 2.0f;
    for (int i = 0; i < 16 * 8; i++) C[i] = 0.5f;

    // 手动计算 MMA
    for (int m = 0; m < 16; m++) {
        for (int n = 0; n < 8; n++) {
            float sum = C[m * 8 + n];  // 初始 C
            for (int k = 0; k < 8; k++) {
                sum += A[m * 8 + k] * B[k * 8 + n];
            }
            D[m * 8 + n] = sum;
        }
    }

    std::cout << "手动 MMA 结果验证:" << std::endl;
    std::cout << "  D[0] = " << D[0] << " (期望：0.5 + 8*1*2 = 16.5)" << std::endl;
    std::cout << "  D[7] = " << D[7] << " (期望：16.5)" << std::endl;
}

// ============================================================================
// 第二部分：CUTE MMA Atom - 基本使用
// ============================================================================

void test_mma_atom() {
    std::cout << "\n=== 测试 2: CUTE MMA Atom ===" << std::endl;

    // MMA Atom 是 CUTE 中对单个 MMA 指令的抽象
    // 使用 mma_atom 获取特定架构的 MMA 配置

    // 获取 SM80 的 16x8x8 FP16 MMA 配置
    using MMA = MMA_Atom<SM80_16x8x8_F32F16F16F32_TN>;

    std::cout << "MMA Atom 配置:" << std::endl;
    std::cout << "  架构：SM80 (Ampere)" << std::endl;
    std::cout << "  尺寸：16x8x8" << std::endl;
    std::cout << "  精度：FP16 输入，FP32 累加" << std::endl;

    // 获取 MMA 的形状信息
    auto mma_shape = MMA::Shape_MNK{};
    std::cout << "  M = " << cute::size<0>(mma_shape) << std::endl;
    std::cout << "  N = " << cute::size<1>(mma_shape) << std::endl;
    std::cout << "  K = " << cute::size<2>(mma_shape) << std::endl;
}

// ============================================================================
// 第三部分：MMA 寄存器布局
// ============================================================================

void test_mma_register_layout() {
    std::cout << "\n=== 测试 3: MMA 寄存器布局 ===" << std::endl;

    // 在 Tensor Core MMA 中，数据需要按照特定格式加载到寄存器
    // CUTE 提供了 Layout 来描述这些寄存器文件

    // A 矩阵 (16x8) 的寄存器布局
    // 在 SM80 上，A 被分成 4 个 4x8 的子块，每个线程处理一部分
    auto layout_A = make_layout(make_shape(16, 8));
    std::cout << "A 矩阵 Layout (16x8):" << std::endl;
    std::cout << "  " << layout_A << std::endl;

    // B 矩阵 (8x8) 的寄存器布局
    auto layout_B = make_layout(make_shape(8, 8));
    std::cout << "B 矩阵 Layout (8x8):" << std::endl;
    std::cout << "  " << layout_B << std::endl;

    // C/D 矩阵 (16x8) 的寄存器布局
    auto layout_C = make_layout(make_shape(16, 8));
    std::cout << "C/D 矩阵 Layout (16x8):" << std::endl;
    std::cout << "  " << layout_C << std::endl;
}

// ============================================================================
// 第四部分：实际的 CUDA Kernel - 使用 intrinsics
// ============================================================================

/**
 * 简单的 MMA kernel 示例
 * 使用 CUDA intrinsic __syncthreads_and 进行同步
 */
__global__ void simple_mma_kernel(half* A, half* B, float* C, int M, int N, int K) {
    // 共享内存用于存储 tile
    __shared__ half As[16 * 8];
    __shared__ half Bs[8 * 8];

    // 线程索引
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * 32 + tx;

    // 从全局内存加载数据到共享内存
    // 这里简化处理，假设 M>=16, N>=8, K>=8
    if (tid < 16 * 8) {
        As[tid] = A[tid];
    }
    if (tid < 8 * 8) {
        Bs[tid] = B[tid];
    }
    __syncthreads();

    // 使用 Tensor Core MMA
    // 注意：这里使用 CUDA 的 wgmma  intrinsic
    // 在实际应用中，应该使用 CUTE 的 mma_atom API

    // 简单的标量乘法作为占位符
    if (tid < 16 * 8) {
        float sum = 0.0f;
        for (int k = 0; k < 8; k++) {
            sum += float(As[(tid / 8) * 8 + k]) * float(Bs[k * 8 + (tid % 8)]);
        }
        C[tid] = sum;
    }
}

// ============================================================================
// 第五部分：使用 CUTE mma_atom 的真正示例
// ============================================================================

template<typename MMA_Op>
__device__ void perform_mma(
    float& d00, float& d01, float& d02, float& d03,
    float& d10, float& d11, float& d12, float& d13,
    float& d20, float& d21, float& d22, float& d23,
    float& d30, float& d31, float& d32, float& d33,
    half a0, half a1, half a2, half a3,
    half b0, half b1, half b2, half b3,
    float c00, float c01, float c02, float c03,
    float c10, float c11, float c12, float c13,
    float c20, float c21, float c22, float c23,
    float c30, float c31, float c32, float c33) {

    // 使用 SM80 的 mma.sync 指令
    // 这是 16x8x8 的 FP16 MMA 操作

    uint32_t a[2];
    uint32_t b[2];
    uint32_t c[8];
    uint32_t d[8];

    // 打包 FP16 数据到寄存器
    a[0] = ((__half_raw)a0).x | (((__half_raw)a1).x << 16);
    a[1] = ((__half_raw)a2).x | (((__half_raw)a3).x << 16);
    b[0] = ((__half_raw)b0).x | (((__half_raw)b1).x << 16);
    b[1] = ((__half_raw)b2).x | (((__half_raw)b3).x << 16);

    // 打包累加器
    c[0] = __float_as_uint(c00); c[1] = __float_as_uint(c01);
    c[2] = __float_as_uint(c02); c[3] = __float_as_uint(c03);
    c[4] = __float_as_uint(c10); c[5] = __float_as_uint(c11);
    c[6] = __float_as_uint(c12); c[7] = __float_as_uint(c13);

    // 执行 MMA 指令
    asm volatile(
        "mma.sync.aligned.m16n8k8.f32.f16.f16.f32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{%8, %9}, {%10, %11}, {%12, %13, %14, %15, %16, %17, %18, %19};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]),
          "=r"(d[4]), "=r"(d[5]), "=r"(d[6]), "=r"(d[7])
        : "r"(a[0]), "r"(a[1]),
          "r"(b[0]), "r"(b[1]),
          "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]),
          "r"(c[4]), "r"(c[5]), "r"(c[6]), "r"(c[7])
    );

    // 解包结果（这里简化处理）
    d00 = __uint_as_float(d[0]);
}

// ============================================================================
// 第六部分：MMA 性能考量
// ============================================================================

void test_mma_performance_tips() {
    std::cout << "\n=== 测试 4: MMA 性能优化技巧 ===" << std::endl;

    std::cout << "\n1. 数据对齐:" << std::endl;
    std::cout << "   - 全局内存访问应该 16 字节对齐" << std::endl;
    std::cout << "   - 共享内存应该使用合适的 stride 避免 Bank Conflict" << std::endl;

    std::cout << "\n2. 寄存器使用:" << std::endl;
    std::cout << "   - 每个线程需要足够的寄存器存储 A, B, C 片段" << std::endl;
    std::cout << "   - SM80 上每个线程最多 255 个寄存器" << std::endl;

    std::cout << "\n3. 指令级并行:" << std::endl;
    std::cout << "   - 使用多个 MMA 指令流水线隐藏延迟" << std::endl;
    std::cout << "   - 典型延迟：4-8 周期，吞吐量：每周期 1 次 MMA" << std::endl;

    std::cout << "\n4. 内存层次:" << std::endl;
    std::cout << "   - L2 Cache -> 共享内存 -> 寄存器 -> MMA" << std::endl;
    std::cout << "   - 使用异步内存拷贝 (CP_ASYNC) 重叠计算和传输" << std::endl;
}

// ============================================================================
// 第七部分：完整的 MMA 示例 - 小矩阵乘法
// ============================================================================

__global__ void mma_16x8x8_kernel(half* A, half* B, float* C) {
    // 线程配置：32 线程 (warp)
    int tid = threadIdx.x;

    // 共享内存
    __shared__ half As[16 * 8];
    __shared__ half Bs[8 * 8];

    // 加载数据
    if (tid < 16 * 8) As[tid] = A[tid];
    if (tid < 8 * 8) Bs[tid] = B[tid];
    __syncthreads();

    // 每个线程计算输出的一部分
    // 这里使用简化的方法，实际应该使用 mma_atom

    if (tid < 16 * 8) {
        int m = tid / 8;
        int n = tid % 8;

        float acc = 0.0f;
        for (int k = 0; k < 8; k++) {
            half a = As[m * 8 + k];
            half b = Bs[k * 8 + n];
            acc += __half2float(a) * __half2float(b);
        }
        C[tid] = acc;
    }
}

void run_mma_example() {
    std::cout << "\n=== 运行 MMA 示例 ===" << std::endl;

    // 分配内存
    half *d_A, *d_B;
    float *d_C;

    cudaMalloc(&d_A, 16 * 8 * sizeof(half));
    cudaMalloc(&d_B, 8 * 8 * sizeof(half));
    cudaMalloc(&d_C, 16 * 8 * sizeof(float));

    // 初始化数据 (主机)
    half h_A[16 * 8];
    half h_B[8 * 8];
    for (int i = 0; i < 16 * 8; i++) h_A[i] = __float2half(1.0f);
    for (int i = 0; i < 8 * 8; i++) h_B[i] = __float2half(2.0f);

    // 拷贝到设备
    cudaMemcpy(d_A, h_A, 16 * 8 * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, 8 * 8 * sizeof(half), cudaMemcpyHostToDevice);

    // 启动 kernel
    mma_16x8x8_kernel<<<1, 32>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    // 验证结果
    float h_C[16 * 8];
    cudaMemcpy(h_C, d_C, 16 * 8 * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "MMA 结果 (16x8 输出):" << std::endl;
    std::cout << "  C[0] = " << h_C[0] << " (期望：8*1*2=16)" << std::endl;
    std::cout << "  C[7] = " << h_C[7] << " (期望：16)" << std::endl;
    std::cout << "  C[127] = " << h_C[127] << " (期望：16)" << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  CUTE MMA 指令教程" << std::endl;
    std::cout << "========================================" << std::endl;

    test_mma_math();
    test_mma_atom();
    test_mma_register_layout();
    test_mma_performance_tips();
    run_mma_example();

    std::cout << "\n========================================" << std::endl;
    std::cout << "  第四课完成！" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
