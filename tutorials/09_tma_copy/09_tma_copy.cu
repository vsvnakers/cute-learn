/**
 * 第九课：TMA - Tensor Memory Accelerator 拷贝
 *
 * 本课讲解 Hopper 架构引入的 TMA 拷贝机制：
 * 1. TMA 描述符
 * 2. 异步拷贝
 * 3. TMA 与 MMA 流水线
 *
 * 编译：nvcc -std=c++17 -arch=sm_90 09_tma_copy.cu -o 09_tma_copy
 * 注意：TMA 需要 Hopper (SM90) GPU
 */

#include <iostream>
#include <cuda_runtime.h>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

using namespace cute;

// ============================================================================
// TMA 基础概念讲解
// ============================================================================

void test_tma_basics() {
    std::cout << "=== TMA 基础概念 ===" << std::endl;

    std::cout << R"(
TMA (Tensor Memory Accelerator) 是 Hopper 架构引入的新特性：

传统拷贝方式:
  全局内存 -> 寄存器 -> 共享内存
  (需要线程执行加载指令)

TMA 拷贝方式:
  全局内存 -> 共享内存 (硬件加速，异步)
  (专用硬件单元，不占用线程)

TMA 优势:
  1. 异步执行 - 与计算重叠
  2. 硬件加速 - 专用数据路径
  3. 简化编程 - 无需手动向量化
  4. 支持 ND 拷贝 - 多维数据布局
    )" << std::endl;

    // TMA 支持的操作
    std::cout << "TMA 支持的操作:" << std::endl;
    std::cout << "  1. 1D/2D/3D/4D/5D 拷贝" << std::endl;
    std::cout << "  2. Box 拷贝 (矩形区域)" << std::endl;
    std::cout << "  3. Im2Col 转换" << std::endl;
    std::cout << "  4. 数据重排" << std::endl;
}

// ============================================================================
// TMA 描述符创建
// ============================================================================

void test_tma_descriptor() {
    std::cout << "\n=== TMA 描述符 ===" << std::endl;

    // TMA 描述符包含：
    // - 源地址（全局内存）
    // - 目标地址（共享内存）
    // - 形状和 stride
    // - 数据打包格式

    std::cout << "TMA 描述符组成:" << std::endl;
    std::cout << "  - global_base_ptr: 全局内存基地址" << std::endl;
    std::cout << "  - global_stride:   全局内存 stride" << std::endl;
    std::cout << "  - smem_base_ptr:   共享内存基地址" << std::endl;
    std::cout << "  - shape:           数据形状" << std::endl;
    std::cout << "  - element_type:    数据类型" << std::endl;

    // 示例：创建 2D 矩阵的 TMA 描述符
    constexpr int M = 1024;
    constexpr int N = 1024;

    std::cout << "\n示例：2D 矩阵 TMA (" << M << "x" << N << ")" << std::endl;
    std::cout << "  global_stride[0] = " << sizeof(half) << " (element size)" << std::endl;
    std::cout << "  global_stride[1] = " << M * sizeof(half) << " (row stride)" << std::endl;
}

// ============================================================================
// CUTE TMA API
// ============================================================================

void test_cute_tma_api() {
    std::cout << "\n=== CUTE TMA API ===" << std::endl;

    // CUTE 提供了高级 TMA 接口
    std::cout << "CUTE TMA 相关 API:" << std::endl;
    std::cout << "  1. make_tma_copy() - 创建 TMA 拷贝操作" << std::endl;
    std::cout << "  2. tma_load() - TMA 加载" << std::endl;
    std::cout << "  3. tma_store() - TMA 存储" << std::endl;
    std::cout << "  4. mbarrier.arrive() - 屏障同步" << std::endl;

    // TMA 拷贝模式
    std::cout << "\nTMA 拷贝模式:" << std::endl;
    std::cout << "  - TMA::Copy_Mode::Default: 默认模式" << std::endl;
    std::cout << "  - TMA::Copy_Mode::Vector:  向量化模式" << std::endl;
    std::cout << "  - TMA::Copy_Mode::Swizzle: Swizzle 模式" << std::endl;
}

// ============================================================================
// TMA 与 MMA 流水线
// ============================================================================

void test_tma_mma_pipeline() {
    std::cout << "\n=== TMA + MMA 流水线 ===" << std::endl;

    std::cout << R"(
典型 Hopper GEMM 流水线:

Stage 0:
  TMA Load A[0], B[0] -> SMEM
Stage 1:
  TMA Load A[1], B[1] -> SMEM
  SMEM[0] -> Regs (ldmatrix)
Stage 2:
  TMA Load A[2], B[2] -> SMEM
  SMEM[1] -> Regs (ldmatrix)
  Regs[0] -> MMA (mma.sync)
Stage 3:
  ...

关键技术:
  - 多缓冲 (Multi-buffering)
  - mbarrier 同步
  - wgmma 指令
    )" << std::endl;
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  第九课：TMA - Tensor Memory Accelerator" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    test_tma_basics();
    test_tma_descriptor();
    test_cute_tma_api();
    test_tma_mma_pipeline();

    std::cout << "\n========================================" << std::endl;
    std::cout << "  注意：TMA 需要 Hopper (SM90) GPU" << std::endl;
    std::cout << "  完整示例请参考 CUTLASS 3.x" << std::endl;
    std::cout << "  第九课完成！" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
