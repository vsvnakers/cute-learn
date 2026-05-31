/**
 * ============================================================================
 * CuTe + MMA 教程 05: MMA 基础
 * ============================================================================
 *
 * MMA (Matrix Multiply-Accumulate) 是 Tensor Core 的核心操作。
 * D = A * B + C
 *
 * CuTe 中的 MMA 层次结构：
 *   - MMA_Op (arch/mma_smXX.hpp): 底层 PTX 指令封装
 *   - MMA_Traits (atom/mma_traits_smXX.hpp): MMA 的属性描述
 *   - MMA_Atom (atom/mma_atom.hpp): 单个 MMA 指令的完整抽象
 *   - TiledMMA: 将 MMA_Atom 扩展到更大的 tile
 *
 * SM80 (Ampere) 支持的 MMA 操作 (16x8xK):
 *   - FP16:   16x8x8,  16x8x16, 16x8x32
 *   - BF16:   16x8x8,  16x8x16
 *   - TF32:   16x8x4,  16x8x8
 *   - INT8:   16x8x16, 16x8x32
 *   - FP64:   8x8x4
 *
 * 编译：make 05_mma_basic
 * 运行：./05_mma_basic
 */

#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/layout.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cute/arch/mma_sm80.hpp>

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
// 1. MMA 操作的数学本质
// ============================================================================
// MMA: D = A * B + C
// 其中:
//   A: M x K 矩阵
//   B: K x N 矩阵
//   C: M x N 矩阵 (累加器)
//   D: M x N 矩阵 (输出)
//
// SM80 的基本 MMA 形状是 16x8xK:
//   M=16, N=8, K={4,8,16,32}
//   使用 32 个线程 (一个 warp)
//   每个线程负责输出的一部分

void test_mma_math() {
    std::cout << "=== 1. MMA 数学基础 ===" << std::endl;
    std::cout << std::endl;

    std::cout << "  MMA 运算: D = A * B + C" << std::endl;
    std::cout << std::endl;

    // 手动计算一个小规模 MMA
    // 4x4x4 矩阵乘法
    const int M = 4, N = 4, K = 4;
    float A[M*K] = {1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16};
    float B[K*N] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};  // 单位矩阵
    float C[M*N] = {0};
    float D[M*N] = {0};

    // D = A * B + C
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = C[m * N + n];
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[k * N + n];
            }
            D[m * N + n] = sum;
        }
    }

    std::cout << "  A (4x4):" << std::endl;
    for (int m = 0; m < M; m++) {
        std::cout << "    ";
        for (int k = 0; k < K; k++) std::cout << std::setw(4) << A[m*K+k];
        std::cout << std::endl;
    }

    std::cout << "  B (4x4, 单位矩阵):" << std::endl;
    for (int k = 0; k < K; k++) {
        std::cout << "    ";
        for (int n = 0; n < N; n++) std::cout << std::setw(4) << B[k*N+n];
        std::cout << std::endl;
    }

    std::cout << "  D = A*B (应该等于 A):" << std::endl;
    for (int m = 0; m < M; m++) {
        std::cout << "    ";
        for (int n = 0; n < N; n++) std::cout << std::setw(4) << D[m*N+n];
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// ============================================================================
// 2. MMA_Atom 基础
// ============================================================================
// MMA_Atom 是 CuTe 中单个 MMA 指令的抽象
// 它包含:
//   - Shape_MNK: MMA 的形状 (M, N, K)
//   - ThrID: 线程到输出的映射
//   - ALayout/BLayout/CLayout: 输入/输出的 Thread-Value 布局

void test_mma_atom() {
    std::cout << "=== 2. MMA_Atom 基础 ===" << std::endl;
    std::cout << std::endl;

    // ---- FP16 MMA: 16x8x8 ----
    // SM80_16x8x8_F32F16F16F32_TN:
    //   输入 A: FP16, 16x8
    //   输入 B: FP16, 8x8
    //   累加器: FP32, 16x8
    //   输出:   FP32, 16x8
    //   TN: Transpose N, 即 A 是 Row-Major, B 是 Column-Major
    using MMA_16x8x8 = MMA_Atom<SM80_16x8x8_F32F16F16F32_TN>;

    std::cout << "  SM80_16x8x8_F32F16F16F32_TN:" << std::endl;
    std::cout << "    Shape_MNK = " << MMA_16x8x8::Shape_MNK{} << std::endl;
    std::cout << "    M = " << size<0>(MMA_16x8x8::Shape_MNK{}) << std::endl;
    std::cout << "    N = " << size<1>(MMA_16x8x8::Shape_MNK{}) << std::endl;
    std::cout << "    K = " << size<2>(MMA_16x8x8::Shape_MNK{}) << std::endl;
    std::cout << std::endl;

    // ThrID: 线程编号布局
    // 32 个线程，排列成 _32 的 1D layout
    std::cout << "    ThrID = " << MMA_16x8x8::ThrID{} << std::endl;
    std::cout << "    线程数 = " << size(MMA_16x8x8::ThrID{}) << std::endl;
    std::cout << std::endl;

    // ALayout: A 矩阵的 Thread-Value 布局
    // (T32, V4) -> (M16, K8)
    // T32 = (4,8):(32,1)  - 32 个线程的 2D 排列
    // V4  = (2,2):(16,8)  - 每个线程持有 4 个值
    std::cout << "    ALayout (Thr,Val) -> (M,K):" << std::endl;
    std::cout << "      " << MMA_16x8x8::ALayout{} << std::endl;
    std::cout << "      含义: 32 线程 x 4 值/线程 = 128 = 16*8 个元素" << std::endl;
    std::cout << std::endl;

    // BLayout: B 矩阵的 Thread-Value 布局
    // (T32, V2) -> (K8, N8)
    std::cout << "    BLayout (Thr,Val) -> (K,N):" << std::endl;
    std::cout << "      " << MMA_16x8x8::BLayout{} << std::endl;
    std::cout << "      含义: 32 线程 x 2 值/线程 = 64 = 8*8 个元素" << std::endl;
    std::cout << std::endl;

    // CLayout: C/D 矩阵的 Thread-Value 布局
    // (T32, V4) -> (M16, N8)
    std::cout << "    CLayout (Thr,Val) -> (M,N):" << std::endl;
    std::cout << "      " << MMA_16x8x8::CLayout{} << std::endl;
    std::cout << "      含义: 32 线程 x 4 值/线程 = 128 = 16*8 个元素" << std::endl;
    std::cout << std::endl;

    // 值类型
    std::cout << "    ValTypeA = " << typeid(MMA_16x8x8::ValTypeA).name() << std::endl;
    std::cout << "    ValTypeB = " << typeid(MMA_16x8x8::ValTypeB).name() << std::endl;
    std::cout << "    ValTypeC = " << typeid(MMA_16x8x8::ValTypeC).name() << std::endl;
    std::cout << "    ValTypeD = " << typeid(MMA_16x8x8::ValTypeD).name() << std::endl;
    std::cout << std::endl;
}

// ============================================================================
// 3. SM80 MMA 操作列表
// ============================================================================

void test_sm80_mma_list() {
    std::cout << "=== 3. SM80 MMA 操作列表 ===" << std::endl;
    std::cout << std::endl;

    // FP16 -> FP32
    std::cout << "  FP16 输入, FP32 累加:" << std::endl;
    std::cout << "    SM80_16x8x8_F32F16F16F32_TN    (K=8)" << std::endl;
    std::cout << "    SM80_16x8x16_F32F16F16F32_TN   (K=16)" << std::endl;
    std::cout << std::endl;

    // FP16 -> FP16
    std::cout << "  FP16 输入, FP16 累加:" << std::endl;
    std::cout << "    SM80_16x8x8_F16F16F16F16_TN    (K=8)" << std::endl;
    std::cout << "    SM80_16x8x16_F16F16F16F16_TN   (K=16)" << std::endl;
    std::cout << std::endl;

    // BF16 -> FP32
    std::cout << "  BF16 输入, FP32 累加:" << std::endl;
    std::cout << "    SM80_16x8x8_F32BF16BF16F32_TN   (K=8)" << std::endl;
    std::cout << "    SM80_16x8x16_F32BF16BF16F32_TN  (K=16)" << std::endl;
    std::cout << std::endl;

    // TF32 -> FP32
    std::cout << "  TF32 输入, FP32 累加:" << std::endl;
    std::cout << "    SM80_16x8x4_F32TF32TF32F32_TN   (K=4)" << std::endl;
    std::cout << "    SM80_16x8x8_F32TF32TF32F32_TN   (K=8)" << std::endl;
    std::cout << std::endl;

    // INT8 -> INT32
    std::cout << "  INT8 输入, INT32 累加:" << std::endl;
    std::cout << "    SM80_16x8x16_S32S8S8S32_TN      (K=16)" << std::endl;
    std::cout << "    SM80_16x8x32_S32S8S8S32_TN      (K=32)" << std::endl;
    std::cout << std::endl;

    // FP64
    std::cout << "  FP64 输入, FP64 累加:" << std::endl;
    std::cout << "    SM80_8x8x4_F64F64F64F64_TN      (M=8, K=4)" << std::endl;
    std::cout << std::endl;

    std::cout << "  命名规则: SM{arch}_{M}x{N}x{K}_{DType}{AType}{BType}{CType}_{TN}" << std::endl;
    std::cout << "    TN: A 是 Row-Major, B 是 Column-Major" << std::endl;
    std::cout << std::endl;
}

// ============================================================================
// 4. MMA 的底层 PTX 指令
// ============================================================================
// CuTe 的 MMA 最终映射到 PTX 的 mma.sync 指令
// 指令格式: mma.sync.aligned.m{M}n{N}k{K}.row.col.{d}.{a}.{b}.{c}

void test_mma_ptx() {
    std::cout << "=== 4. MMA PTX 指令 ===" << std::endl;
    std::cout << std::endl;

    std::cout << "  PTX 指令格式:" << std::endl;
    std::cout << "    mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32" << std::endl;
    std::cout << std::endl;
    std::cout << "    mma.sync     : 同步 MMA 操作" << std::endl;
    std::cout << "    aligned      : warp 内所有线程都参与" << std::endl;
    std::cout << "    m16n8k8      : MMA 形状 M=16, N=8, K=8" << std::endl;
    std::cout << "    row          : A 矩阵 Row-Major" << std::endl;
    std::cout << "    col          : B 矩阵 Column-Major" << std::endl;
    std::cout << "    f32          : 输出/累加器类型" << std::endl;
    std::cout << "    f16.f16      : A 和 B 的输入类型" << std::endl;
    std::cout << "    f32          : C 的类型" << std::endl;
    std::cout << std::endl;

    std::cout << "  CuTe 对应的底层结构 (SM80_16x8x8_F32F16F16F32_TN):" << std::endl;
    std::cout << "    DRegisters = float[4]   (每个线程 4 个 FP32 输出)" << std::endl;
    std::cout << "    ARegisters = uint32_t[2] (每个线程 2 个 uint32，打包 4 个 FP16)" << std::endl;
    std::cout << "    BRegisters = uint32_t[1] (每个线程 1 个 uint32，打包 2 个 FP16)" << std::endl;
    std::cout << "    CRegisters = float[4]   (每个线程 4 个 FP32 累加器)" << std::endl;
    std::cout << std::endl;

    std::cout << "  寄存器布局 (16x8x8 FP16 MMA):" << std::endl;
    std::cout << "    A 矩阵 (16x8):" << std::endl;
    std::cout << "      32 线程, 每线程 4 个 FP16 (打包为 2 个 uint32)" << std::endl;
    std::cout << "      线程排列: (4,8) -> (M 组, K 组)" << std::endl;
    std::cout << "    B 矩阵 (8x8):" << std::endl;
    std::cout << "      32 线程, 每线程 2 个 FP16 (打包为 1 个 uint32)" << std::endl;
    std::cout << "    C/D 矩阵 (16x8):" << std::endl;
    std::cout << "      32 线程, 每线程 4 个 FP32" << std::endl;
    std::cout << std::endl;
}

// ============================================================================
// 5. 实际 MMA Kernel: 小矩阵乘法
// ============================================================================
// 使用 MMA_Atom 执行一个 16x8 的矩阵乘法
// A: 16x8 (FP16)
// B: 8x8  (FP16)
// C: 16x8 (FP32)

__global__ void mma_small_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C)
{
    // 定义 MMA Atom
    using MMA = MMA_Atom<SM80_16x8x8_F32F16F16F32_TN>;

    // 每个线程的寄存器
    // A: 2 个 uint32 (4 个 FP16)
    uint32_t a_reg[2];
    // B: 1 个 uint32 (2 个 FP16)
    uint32_t b_reg[1];
    // C/D: 4 个 float
    float c_reg[4] = {0, 0, 0, 0};
    float d_reg[4];

    int tid = threadIdx.x;

    // ---- 加载 A 矩阵到寄存器 ----
    // SM80_16x8_Row 的布局:
    // (thr, val) -> (M, K)
    // thr = (tid/8, tid%8)  -> M 方向: tid/8 * 4, K 方向: tid%8
    // val = (v/2, v%2)      -> M 方向: v/2 * 16, K 方向: v%2 * 8
    //
    // 但这里我们简化：直接按 MMA 的寄存器布局加载
    // 每个线程需要 4 个 FP16 值，打包为 2 个 uint32

    // 简化加载：假设 A 是全 1 的矩阵
    half a_vals[4] = {__float2half(1.0f), __float2half(1.0f),
                      __float2half(1.0f), __float2half(1.0f)};
    // 打包 FP16 到 uint32
    // 每个 uint32 包含 2 个 FP16
    a_reg[0] = ((__half_raw&)a_vals[0]).x | (((__half_raw&)a_vals[1]).x << 16);
    a_reg[1] = ((__half_raw&)a_vals[2]).x | (((__half_raw&)a_vals[3]).x << 16);

    // ---- 加载 B 矩阵到寄存器 ----
    half b_vals[2] = {__float2half(1.0f), __float2half(1.0f)};
    b_reg[0] = ((__half_raw&)b_vals[0]).x | (((__half_raw&)b_vals[1]).x << 16);

    // ---- 执行 MMA ----
    // 使用底层 MMA_Op 的 fma 接口
    // MMA_Op 是 SM80_16x8x8_F32F16F16F32_TN
    // 参数: (d..., a..., b..., c...)
    SM80_16x8x8_F32F16F16F32_TN::fma(
             d_reg[0], d_reg[1], d_reg[2], d_reg[3],
             a_reg[0], a_reg[1],
             b_reg[0],
             c_reg[0], c_reg[1], c_reg[2], c_reg[3]);

    // ---- 存储结果 ----
    // 每个线程存储 4 个 float
    // CLayout 的布局: (thr, val) -> (M, N)
    // 简化存储：按线程顺序存储
    if (tid < 32) {
        C[tid * 4 + 0] = d_reg[0];
        C[tid * 4 + 1] = d_reg[1];
        C[tid * 4 + 2] = d_reg[2];
        C[tid * 4 + 3] = d_reg[3];
    }
}

void test_mma_small() {
    std::cout << "=== 5. 小规模 MMA Kernel ===" << std::endl;
    std::cout << std::endl;

    // 分配设备内存
    float* d_C;
    CUDA_CHECK(cudaMalloc(&d_C, 32 * 4 * sizeof(float)));

    // 启动 kernel (单个 warp)
    mma_small_kernel<<<1, 32>>>(nullptr, nullptr, d_C);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 拷贝回 host
    float h_C[128];
    CUDA_CHECK(cudaMemcpy(h_C, d_C, 32 * 4 * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "  MMA 结果 (每个线程 4 个值):" << std::endl;
    std::cout << "    线程 0: " << h_C[0] << " " << h_C[1] << " " << h_C[2] << " " << h_C[3] << std::endl;
    std::cout << "    (期望: 8.0 8.0 8.0 8.0, 因为 A 全 1, B 全 1, K=8)" << std::endl;
    std::cout << std::endl;

    CUDA_CHECK(cudaFree(d_C));
}

// ============================================================================
// 6. MMA 的寄存器映射详解
// ============================================================================

void test_mma_register_mapping() {
    std::cout << "=== 6. MMA 寄存器映射详解 ===" << std::endl;
    std::cout << std::endl;

    // 使用 SM80_16x8x8_F32F16F16F32_TN 为例
    using MMA = MMA_Atom<SM80_16x8x8_F32F16F16F32_TN>;

    std::cout << "  SM80_16x8x8_F32F16F16F32_TN 寄存器映射:" << std::endl;
    std::cout << std::endl;

    // A 矩阵寄存器布局
    // ALayout: (T32, V4) -> (M16, K8)
    // T32 = (4,8):(32,1)  - 线程的 2D 排列
    // V4  = (2,2):(16,8)  - 每线程 4 个值
    std::cout << "  A 矩阵 (16x8, FP16):" << std::endl;
    std::cout << "    每线程: 4 个 FP16, 打包为 2 个 uint32" << std::endl;
    std::cout << "    a_reg[0] = {A[tid_m, tid_k], A[tid_m, tid_k+8]}" << std::endl;
    std::cout << "    a_reg[1] = {A[tid_m+16, tid_k], A[tid_m+16, tid_k+8]}" << std::endl;
    std::cout << "    其中: tid_m = (tid/8)*4, tid_k = tid%8" << std::endl;
    std::cout << std::endl;

    // B 矩阵寄存器布局
    std::cout << "  B 矩阵 (8x8, FP16):" << std::endl;
    std::cout << "    每线程: 2 个 FP16, 打包为 1 个 uint32" << std::endl;
    std::cout << "    b_reg[0] = {B[tid_k, tid_n], B[tid_k, tid_n+8]}" << std::endl;
    std::cout << "    其中: tid_k = tid%8, tid_n = (tid/8)*2" << std::endl;
    std::cout << std::endl;

    // C/D 矩阵寄存器布局
    std::cout << "  C/D 矩阵 (16x8, FP32):" << std::endl;
    std::cout << "    每线程: 4 个 float" << std::endl;
    std::cout << "    c_reg[0] = C[tid_m, tid_n]" << std::endl;
    std::cout << "    c_reg[1] = C[tid_m, tid_n+8]" << std::endl;
    std::cout << "    c_reg[2] = C[tid_m+16, tid_n]" << std::endl;
    std::cout << "    c_reg[3] = C[tid_m+16, tid_n+8]" << std::endl;
    std::cout << "    其中: tid_m = (tid/8)*4, tid_n = (tid%8)*1" << std::endl;
    std::cout << std::endl;
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  CuTe + MMA 教程 05: MMA 基础" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    test_mma_math();
    test_mma_atom();
    test_sm80_mma_list();
    test_mma_ptx();
    test_mma_small();
    test_mma_register_mapping();

    std::cout << "========================================" << std::endl;
    std::cout << "  教程 05 完成！" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
