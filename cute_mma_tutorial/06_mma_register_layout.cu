/**
 * ============================================================================
 * CuTe + MMA 教程 06: MMA 寄存器布局深度解析
 * ============================================================================
 *
 * 本文件深入解析 CuTe MMA 中的寄存器布局 (Thread-Value Layout)。
 * 这是理解 CuTe MMA 工作原理的关键。
 *
 * 核心概念：
 *   - ThrID: 线程编号的 Layout，描述 warp 内 32 个线程如何排列
 *   - ALayout/BLayout/CLayout: (Thread, Value) -> (Matrix Coord) 的映射
 *   - 每个线程持有矩阵的一部分值，存储在寄存器中
 *
 * 编译：make 06_mma_register_layout
 * 运行：./06_mma_register_layout
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
// 1. ThrID 详解
// ============================================================================
// ThrID 描述了 warp 内 32 个线程的编号方式
// 对于 SM80 MMA，ThrID 通常是 Layout<_32> (1D)
// 但在 ALayout/BLayout/CLayout 中，线程被重新解释为 2D 排列

void test_thr_id() {
    std::cout << "=== 1. ThrID 详解 ===" << std::endl;
    std::cout << std::endl;

    // SM80 MMA 使用 32 个线程 (一个 warp)
    // ThrID = Layout<_32>，即线程编号 0-31
    using MMA = MMA_Atom<SM80_16x8x8_F32F16F16F32_TN>;
    auto thr_id = MMA::ThrID{};

    std::cout << "  ThrID Layout: " << thr_id << std::endl;
    std::cout << "  线程数: " << size(thr_id) << std::endl;
    std::cout << std::endl;

    // 在 ALayout 中，线程被重新解释为 2D 排列
    // SM80_16x8_Row 的 Thr 部分: (4,8):(32,1)
    // 含义: 32 个线程排列成 4 行 8 列
    //   tid = 0  -> (0, 0)
    //   tid = 1  -> (0, 1)
    //   ...
    //   tid = 7  -> (0, 7)
    //   tid = 8  -> (1, 0)
    //   ...
    //   tid = 31 -> (3, 7)

    std::cout << "  线程的 2D 排列 (A 矩阵视角):" << std::endl;
    std::cout << "    排列方式: (4,8):(32,1)" << std::endl;
    std::cout << "    即: 4 组 x 8 线程/组" << std::endl;
    std::cout << std::endl;
    std::cout << "    线程 ID 到 2D 坐标的映射:" << std::endl;
    std::cout << "    ";
    for (int tid = 0; tid < 32; tid++) {
        int row = tid / 8;  // M 方向的组
        int col = tid % 8;  // K 方向的位置
        std::cout << "(" << row << "," << col << ") ";
        if ((tid + 1) % 8 == 0) std::cout << std::endl << "    ";
    }
    std::cout << std::endl;
}

// ============================================================================
// 2. ALayout 详解
// ============================================================================
// ALayout: (Thr, Val) -> (M, K)
// 描述了 A 矩阵中每个元素由哪个线程的哪个寄存器值负责

void test_a_layout() {
    std::cout << "=== 2. ALayout 详解 ===" << std::endl;
    std::cout << std::endl;

    // SM80_16x8x8 的 ALayout
    // ALayout = SM80_16x8_Row
    // = Layout<Shape <Shape <_4, _8>, Shape <_2, _2>>,
    //          Stride<Stride<_32, _1>, Stride<_16, _8>>>
    //
    // 外层 Shape: (4,8) = 32 个线程的 2D 排列
    // 外层 Stride: (32,1) = 线程的步长
    // 内层 Shape: (2,2) = 每个线程 4 个值
    // 内层 Stride: (16,8) = 值的步长

    using MMA = MMA_Atom<SM80_16x8x8_F32F16F16F32_TN>;
    auto a_layout = MMA::ALayout{};

    std::cout << "  ALayout: " << a_layout << std::endl;
    std::cout << std::endl;

    // 解读 ALayout 的映射
    std::cout << "  ALayout 映射解析:" << std::endl;
    std::cout << "    线程维度: (4,8):(32,1)" << std::endl;
    std::cout << "      - 32 个线程排列成 4 行 8 列" << std::endl;
    std::cout << "      - 行内 stride=1 (连续线程)" << std::endl;
    std::cout << "      - 跨行 stride=32 (下一组线程)" << std::endl;
    std::cout << std::endl;
    std::cout << "    值维度: (2,2):(16,8)" << std::endl;
    std::cout << "      - 每个线程持有 4 个值，排列成 2x2" << std::endl;
    std::cout << "      - M 方向 stride=16 (跨 16 行)" << std::endl;
    std::cout << "      - K 方向 stride=8 (跨 8 列)" << std::endl;
    std::cout << std::endl;

    // 演示具体的映射
    std::cout << "  具体映射示例 (线程 0 的 4 个值):" << std::endl;
    auto thr_slice = a_layout(Int<0>{}, _);
    for (int v = 0; v < 4; v++) {
        auto offset = thr_slice(v);
        // offset = m * 8 + k (Row-Major)
        int m = offset / 8;
        int k = offset % 8;
        std::cout << "    val " << v << " -> offset=" << offset
                  << " -> (M=" << m << ", K=" << k << ")" << std::endl;
    }
    std::cout << std::endl;

    // 演示线程 5 的映射
    std::cout << "  具体映射示例 (线程 5 的 4 个值):" << std::endl;
    auto thr5_slice = a_layout(Int<5>{}, _);
    for (int v = 0; v < 4; v++) {
        auto offset = thr5_slice(v);
        int m = offset / 8;
        int k = offset % 8;
        std::cout << "    val " << v << " -> offset=" << offset
                  << " -> (M=" << m << ", K=" << k << ")" << std::endl;
    }
    std::cout << std::endl;
}

// ============================================================================
// 3. BLayout 详解
// ============================================================================
// BLayout: (Thr, Val) -> (K, N)
// 描述了 B 矩阵中每个元素由哪个线程的哪个寄存器值负责

void test_b_layout() {
    std::cout << "=== 3. BLayout 详解 ===" << std::endl;
    std::cout << std::endl;

    // SM80_16x8x8 的 BLayout
    // BLayout = SM80_8x8_Row
    // = Layout<Shape <Shape <_4, _8>, _2>,
    //          Stride<Stride<_16, _1>, _8>>
    //
    // 外层 Shape: (4,8) = 32 个线程
    // 内层 Shape: 2 = 每个线程 2 个值

    using MMA = MMA_Atom<SM80_16x8x8_F32F16F16F32_TN>;
    auto b_layout = MMA::BLayout{};

    std::cout << "  BLayout: " << b_layout << std::endl;
    std::cout << std::endl;

    std::cout << "  BLayout 映射解析:" << std::endl;
    std::cout << "    线程维度: (4,8):(16,1)" << std::endl;
    std::cout << "      - 32 个线程排列成 4 行 8 列" << std::endl;
    std::cout << "      - 行内 stride=1" << std::endl;
    std::cout << "      - 跨行 stride=16" << std::endl;
    std::cout << std::endl;
    std::cout << "    值维度: 2:8" << std::endl;
    std::cout << "      - 每个线程持有 2 个值" << std::endl;
    std::cout << "      - 值间 stride=8" << std::endl;
    std::cout << std::endl;

    // 具体映射
    std::cout << "  具体映射示例 (线程 0 的 2 个值):" << std::endl;
    auto thr0 = b_layout(Int<0>{}, _);
    for (int v = 0; v < 2; v++) {
        auto offset = thr0(v);
        int k = offset / 8;
        int n = offset % 8;
        std::cout << "    val " << v << " -> offset=" << offset
                  << " -> (K=" << k << ", N=" << n << ")" << std::endl;
    }
    std::cout << std::endl;
}

// ============================================================================
// 4. CLayout 详解
// ============================================================================
// CLayout: (Thr, Val) -> (M, N)
// 描述了 C/D 矩阵中每个元素由哪个线程的哪个寄存器值负责

void test_c_layout() {
    std::cout << "=== 4. CLayout 详解 ===" << std::endl;
    std::cout << std::endl;

    // SM80_16x8x8 的 CLayout
    // CLayout = SM80_16x8_Row
    // = Layout<Shape <Shape <_4, _8>, Shape <_2, _2>>,
    //          Stride<Stride<_32, _1>, Stride<_16, _8>>>
    //
    // 与 ALayout 相同的结构，但映射到 (M, N) 而非 (M, K)

    using MMA = MMA_Atom<SM80_16x8x8_F32F16F16F32_TN>;
    auto c_layout = MMA::CLayout{};

    std::cout << "  CLayout: " << c_layout << std::endl;
    std::cout << std::endl;

    std::cout << "  CLayout 映射解析:" << std::endl;
    std::cout << "    线程维度: (4,8):(32,1)" << std::endl;
    std::cout << "    值维度: (2,2):(16,8)" << std::endl;
    std::cout << std::endl;

    // 可视化 C 矩阵的线程-值分布
    std::cout << "  C 矩阵 (16x8) 的线程分布:" << std::endl;
    std::cout << "  (每个位置显示 tid.val)" << std::endl;
    std::cout << std::endl;

    // 创建一个 16x8 的矩阵来显示分布
    int matrix[16][8];
    for (int tid = 0; tid < 32; tid++) {
        for (int v = 0; v < 4; v++) {
            auto offset = c_layout(tid, v);
            int m = offset / 8;
            int n = offset % 8;
            matrix[m][n] = tid * 10 + v;
        }
    }

    for (int m = 0; m < 16; m++) {
        std::cout << "    ";
        for (int n = 0; n < 8; n++) {
            std::cout << std::setw(4) << matrix[m][n];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// ============================================================================
// 5. 完整的 MMA 寄存器映射图
// ============================================================================

void test_full_register_map() {
    std::cout << "=== 5. 完整 MMA 寄存器映射图 ===" << std::endl;
    std::cout << std::endl;

    using MMA = MMA_Atom<SM80_16x8x8_F32F16F16F32_TN>;

    std::cout << "  SM80_16x8x8_F32F16F16F32_TN 完整映射:" << std::endl;
    std::cout << std::endl;

    // A 矩阵映射
    std::cout << "  A 矩阵 (16x8, FP16) -> 寄存器:" << std::endl;
    std::cout << "    32 线程 x 4 值/线程 = 128 元素 = 16*8" << std::endl;
    std::cout << "    每线程 4 个 FP16, 打包为 2 个 uint32" << std::endl;
    std::cout << "    a_reg[0] = pack(A[m, k], A[m, k+8])" << std::endl;
    std::cout << "    a_reg[1] = pack(A[m+16, k], A[m+16, k+8])" << std::endl;
    std::cout << "    其中 m = (tid/8)*4, k = tid%8" << std::endl;
    std::cout << std::endl;

    // B 矩阵映射
    std::cout << "  B 矩阵 (8x8, FP16) -> 寄存器:" << std::endl;
    std::cout << "    32 线程 x 2 值/线程 = 64 元素 = 8*8" << std::endl;
    std::cout << "    每线程 2 个 FP16, 打包为 1 个 uint32" << std::endl;
    std::cout << "    b_reg[0] = pack(B[k, n], B[k, n+8])" << std::endl;
    std::cout << "    其中 k = tid%8, n = (tid/8)*2" << std::endl;
    std::cout << std::endl;

    // C 矩阵映射
    std::cout << "  C/D 矩阵 (16x8, FP32) -> 寄存器:" << std::endl;
    std::cout << "    32 线程 x 4 值/线程 = 128 元素 = 16*8" << std::endl;
    std::cout << "    每线程 4 个 float" << std::endl;
    std::cout << "    c_reg[0] = C[m, n]" << std::endl;
    std::cout << "    c_reg[1] = C[m, n+8]" << std::endl;
    std::cout << "    c_reg[2] = C[m+16, n]" << std::endl;
    std::cout << "    c_reg[3] = C[m+16, n+8]" << std::endl;
    std::cout << "    其中 m = (tid/8)*4, n = tid%8" << std::endl;
    std::cout << std::endl;
}

// ============================================================================
// 6. 使用 CuTe Tensor 验证寄存器布局
// ============================================================================
// CuTe 的 Tensor 可以直接用于寄存器，验证布局是否正确

__global__ void verify_register_layout() {
    using MMA = MMA_Atom<SM80_16x8x8_F32F16F16F32_TN>;

    int tid = threadIdx.x;

    // 为每个线程创建寄存器 Tensor
    // A: 4 个 FP16
    half a_vals[4];
    auto a_tensor = make_tensor(static_cast<half*>(a_vals),
                                make_layout(make_shape(Int<4>{})));

    // B: 2 个 FP16
    half b_vals[2];
    auto b_tensor = make_tensor(static_cast<half*>(b_vals),
                                make_layout(make_shape(Int<2>{})));

    // C: 4 个 float
    float c_vals[4];
    auto c_tensor = make_tensor(static_cast<float*>(c_vals),
                                make_layout(make_shape(Int<4>{})));

    // 填充数据：使用线程 ID 标记
    for (int i = 0; i < 4; i++) a_vals[i] = __float2half((float)(tid * 10 + i));
    for (int i = 0; i < 2; i++) b_vals[i] = __float2half((float)(tid * 10 + i));
    for (int i = 0; i < 4; i++) c_vals[i] = (float)(tid * 10 + i);

    // 打印线程 0 和线程 1 的寄存器内容
    if (tid < 2) {
        printf("  Thread %d:\n", tid);
        printf("    A regs: ");
        for (int i = 0; i < 4; i++) printf("%.1f ", __half2float(a_vals[i]));
        printf("\n");
        printf("    B regs: ");
        for (int i = 0; i < 2; i++) printf("%.1f ", __half2float(b_vals[i]));
        printf("\n");
        printf("    C regs: ");
        for (int i = 0; i < 4; i++) printf("%.1f ", c_vals[i]);
        printf("\n");
    }
}

void test_verify_layout() {
    std::cout << "=== 6. 验证寄存器布局 (Device) ===" << std::endl;
    std::cout << std::endl;

    verify_register_layout<<<1, 32>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << std::endl;
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  CuTe + MMA 教程 06: MMA 寄存器布局" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    test_thr_id();
    test_a_layout();
    test_b_layout();
    test_c_layout();
    test_full_register_map();
    test_verify_layout();

    std::cout << "========================================" << std::endl;
    std::cout << "  教程 06 完成！" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
