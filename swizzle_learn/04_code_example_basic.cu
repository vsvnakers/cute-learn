/**
 * CUTE Swizzle 基础示例代码
 *
 * 本文件演示 Swizzle 的基本使用方法
 * 编译：nvcc -std=c++17 -arch=sm_80 04_code_example_basic.cu -o 04_basic
 */

#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>

// CUTE 核心头文件
#include <cute/swizzle.hpp>
#include <cute/layout.hpp>
#include <cute/swizzle_layout.hpp>
#include <cute/tensor.hpp>

using namespace cute;

// ============================================================================
// 辅助函数：检查 CUDA 错误
// ============================================================================

#define CUDA_CHECK(call) do {                                                   \
    cudaError_t err = call;                                                     \
    if (err != cudaSuccess) {                                                   \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err)                  \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;        \
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
} while(0)

// ============================================================================
// 第一部分：理解 Swizzle 函数 - 详细的注释版本
// ============================================================================

void print_separator(const char* title) {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << std::string(70, '=') << std::endl << std::endl;
}

// ----------------------------------------------------------------------------
// 示例 1: 基本的 Swizzle 函数调用
// ----------------------------------------------------------------------------
void example_1_basic_swizzle_function() {
    print_separator("示例 1: 基本的 Swizzle 函数");

    // Swizzle<BBits, NumBanks, SShift>
    // 创建一个 Swizzle 函数对象
    auto swz = Swizzle<2, 4, 3>{};

    std::cout << "Swizzle<2, 4, 3> 的映射表:" << std::endl;
    std::cout << "  BBits=2  -> Bank Size = 2^2 = 4 元素" << std::endl;
    std::cout << "  NumBanks=4 -> Bank 数量 = 2^4 = 16" << std::endl;
    std::cout << "  SShift=3 -> XOR 位移量 = 3" << std::endl;
    std::cout << std::endl;

    std::cout << "  原始偏移 -> Swizzled 偏移 -> Bank 变化" << std::endl;
    for (int i = 0; i < 32; i++) {
        int swizzled = Swizzle<2, 4, 3>::apply(i);

        // 计算 Bank ID (假设每个 Bank 4 个元素)
        int original_bank = (i / 4) % 16;
        int swizzled_bank = (swizzled / 4) % 16;

        std::cout << "  " << std::setw(2) << i << " -> "
                  << std::setw(2) << swizzled << " -> Bank "
                  << original_bank << " -> Bank " << swizzled_bank << std::endl;
    }
}

// ----------------------------------------------------------------------------
// 示例 2: 不同参数的对比
// ----------------------------------------------------------------------------
void example_2_different_params() {
    print_separator("示例 2: 不同参数配置对比");

    // 配置 A: 小 Bank
    std::cout << "配置 A: Swizzle<2, 4, 3>" << std::endl;
    std::cout << "  (Bank Size=4, Num Banks=16)" << std::endl;
    for (int i = 0; i < 16; i++) {
        std::cout << std::setw(2) << Swizzle<2, 4, 3>::apply(i) << " ";
    }
    std::cout << std::endl;

    // 配置 B: 大 Bank
    std::cout << "\n配置 B: Swizzle<3, 4, 4>" << std::endl;
    std::cout << "  (Bank Size=8, Num Banks=16)" << std::endl;
    for (int i = 0; i < 16; i++) {
        std::cout << std::setw(2) << Swizzle<3, 4, 4>::apply(i) << " ";
    }
    std::cout << std::endl;

    // 配置 C: 更多 Bank
    std::cout << "\n配置 C: Swizzle<2, 5, 3>" << std::endl;
    std::cout << "  (Bank Size=4, Num Banks=32)" << std::endl;
    for (int i = 0; i < 16; i++) {
        std::cout << std::setw(2) << Swizzle<2, 5, 3>::apply(i) << " ";
    }
    std::cout << std::endl;
}

// ============================================================================
// 第二部分：Swizzle Layout - 将 Swizzle 应用于 Layout
// ============================================================================

// ----------------------------------------------------------------------------
// 示例 3: 创建带 Swizzle 的 Layout
// ----------------------------------------------------------------------------
void example_3_swizzle_layout() {
    print_separator("示例 3: Swizzle Layout 基础");

    // 创建一个 8x8 的基础 Layout (行优先)
    auto base_layout = make_layout(make_shape(8, 8));

    std::cout << "基础 Layout (8x8 行优先):" << std::endl;
    std::cout << "  Shape: " << base_layout.shape() << std::endl;
    std::cout << "  Stride: " << base_layout.stride() << std::endl;
    std::cout << std::endl;

    // 打印 Layout 矩阵
    std::cout << "  物理存储布局:" << std::endl;
    for (int i = 0; i < 8; i++) {
        std::cout << "    Row " << i << ": ";
        for (int j = 0; j < 8; j++) {
            std::cout << std::setw(3) << base_layout(i, j);
        }
        std::cout << std::endl;
    }

    // 创建 Swizzle Layout
    // 注意：CUTE 中通常对列维度应用 Swizzle
    auto swizzle_fn = Swizzle<2, 3, 3>{};

    // 使用 composition 组合 Swizzle 和基础 Layout
    // composition(f, g) 表示先应用 g，再应用 f
    auto swizzled_layout = composition(swizzle_fn, base_layout);

    std::cout << "\nSwizzled Layout (应用 Swizzle<2,3,3>):" << std::endl;
    std::cout << "  Shape: " << swizzled_layout.shape() << std::endl;
    std::cout << std::endl;

    // 打印 Swizzled Layout 矩阵
    std::cout << "  物理存储布局 (Swizzled):" << std::endl;
    for (int i = 0; i < 8; i++) {
        std::cout << "    Row " << i << ": ";
        for (int j = 0; j < 8; j++) {
            std::cout << std::setw(3) << swizzled_layout(i, j);
        }
        std::cout << std::endl;
    }
}

// ----------------------------------------------------------------------------
// 示例 4: 验证 Swizzle 的可逆性
// ----------------------------------------------------------------------------
void example_4_swizzle_inverse() {
    print_separator("示例 4: 验证 Swizzle 的可逆性");

    auto swz = Swizzle<2, 4, 3>{};

    std::cout << "验证 Swizzle 是可对逆的 (两次应用恢复原值):" << std::endl;
    std::cout << "  offset -> swizzled -> recovered" << std::endl;

    for (int i = 0; i < 32; i++) {
        int swizzled = swz.apply(i);
        int recovered = swz.apply(swizzled);  // 再次应用应该恢复

        std::cout << "  " << std::setw(2) << i << " -> "
                  << std::setw(2) << swizzled << " -> "
                  << std::setw(2) << recovered;

        if (i == recovered) {
            std::cout << " ✓" << std::endl;
        } else {
            std::cout << " ✗ (ERROR!)" << std::endl;
        }
    }
}

// ============================================================================
// 第三部分：GPU Kernel 中的 Swizzle
// ============================================================================

// ----------------------------------------------------------------------------
// 示例 5: 简单的 GPU Kernel 演示
// ----------------------------------------------------------------------------
__global__ void simple_swizzle_kernel(float* output, size_t n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < n) {
        // 应用 Swizzle 变换
        int swizzled = Swizzle<2, 4, 3>::apply(tid);
        output[tid] = static_cast<float>(swizzled);
    }
}

void example_5_gpu_kernel() {
    print_separator("示例 5: GPU Kernel 中的 Swizzle");

    const size_t n = 64;
    float* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(float)));

    // 启动 Kernel
    int threads_per_block = 32;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    simple_swizzle_kernel<<<num_blocks, threads_per_block>>>(d_output, n);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 复制结果回主机
    std::vector<float> h_output(n);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, n * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // 打印结果
    std::cout << "GPU 计算结果 (前 32 个元素):" << std::endl;
    for (int i = 0; i < 32; i++) {
        std::cout << "  [" << std::setw(2) << i << "] = "
                  << std::setw(6) << h_output[i] << std::endl;
    }

    CUDA_CHECK(cudaFree(d_output));
}

// ----------------------------------------------------------------------------
// 示例 6: 共享内存中的 Swizzle 应用
// ----------------------------------------------------------------------------
__global__ void shared_mem_swizzle_kernel() {
    __shared__ float shared_data[64];

    int tid = threadIdx.x;

    // 计算 Swizzled 地址
    int swizzled_addr = Swizzle<2, 4, 3>::apply(tid);

    // 使用 Swizzled 地址写入
    shared_data[swizzled_addr] = static_cast<float>(tid);

    __syncthreads();

    // 读取并验证
    float value = shared_data[swizzled_addr];

    // 输出到全局内存（简化版本，只输出前 32 个）
    if (tid < 32) {
        printf("Thread %2d: swizzled_addr=%2d, value=%.0f\n",
               tid, swizzled_addr, value);
    }
}

void example_6_shared_memory() {
    print_separator("示例 6: 共享内存中的 Swizzle");

    // 启动 32 个线程
    shared_mem_swizzle_kernel<<<1, 32>>>();

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// 第四部分：可视化 Swizzle 效果
// ============================================================================

void example_7_visualize_2d() {
    print_separator("示例 7: 2D 矩阵的 Swizzle 可视化");

    const int ROWS = 8;
    const int COLS = 8;

    auto base_layout = make_layout(make_shape(ROWS, COLS));
    auto swizzle_fn = Swizzle<2, 3, 3>{};
    auto swizzled_layout = composition(swizzle_fn, base_layout);

    std::cout << "原始 8x8 矩阵的线性索引:" << std::endl;
    for (int i = 0; i < ROWS; i++) {
        std::cout << "  ";
        for (int j = 0; j < COLS; j++) {
            std::cout << std::setw(3) << base_layout(i, j);
        }
        std::cout << std::endl;
    }

    std::cout << "\nSwizzled 后的线性索引:" << std::endl;
    for (int i = 0; i < ROWS; i++) {
        std::cout << "  ";
        for (int j = 0; j < COLS; j++) {
            std::cout << std::setw(3) << swizzled_layout(i, j);
        }
        std::cout << std::endl;
    }
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  CUTE Swizzle 基础示例代码" << std::endl;
    std::cout <<  "========================================" << std::endl;

    // 第一部分：基础 Swizzle 函数
    example_1_basic_swizzle_function();
    example_2_different_params();

    // 第二部分：Swizzle Layout
    example_3_swizzle_layout();
    example_4_swizzle_inverse();

    // 第三部分：GPU Kernel
    example_5_gpu_kernel();
    example_6_shared_memory();

    // 第四部分：可视化
    example_7_visualize_2d();

    std::cout << "\n========================================" << std::endl;
    std::cout << "  所有示例完成！" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
