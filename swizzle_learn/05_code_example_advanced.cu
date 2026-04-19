/**
 * CUTE Swizzle 进阶示例代码
 *
 * 本文件演示 Swizzle 的进阶使用方法
 * 编译：nvcc -std=c++17 -arch=sm_80 05_code_example_advanced.cu -o 05_advanced
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <cuda_runtime.h>

// CUTE 核心头文件
#include <cute/swizzle.hpp>
#include <cute/layout.hpp>
#include <cute/swizzle_layout.hpp>
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>

using namespace cute;

// ============================================================================
// 辅助宏
// ============================================================================

#define CUDA_CHECK(call) do {                                                   \
    cudaError_t err = call;                                                     \
    if (err != cudaSuccess) {                                                   \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err)                  \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;        \
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
} while(0)

void print_separator(const char* title) {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << std::string(70, '=') << std::endl << std::endl;
}

// ============================================================================
// 第一部分：Swizzle 组合与变换
// ============================================================================

// ----------------------------------------------------------------------------
// 示例 1: 多级 Swizzle 组合
// ----------------------------------------------------------------------------
void example_1_composed_swizzle() {
    print_separator("示例 1: 多级 Swizzle 组合");

    // 创建两个不同的 Swizzle
    auto swz1 = Swizzle<2, 4, 3>{};
    auto swz2 = Swizzle<3, 4, 4>{};

    std::cout << "原始值 -> Swizzle<2,4,3> -> Swizzle<3,4,4>" << std::endl;
    for (int i = 0; i < 16; i++) {
        int s1 = swz1.apply(i);
        int s2 = swz2.apply(s1);
        std::cout << "  " << std::setw(2) << i << " -> "
                  << std::setw(2) << s1 << " -> "
                  << std::setw(2) << s2 << std::endl;
    }
}

// ----------------------------------------------------------------------------
// 示例 2: Swizzle 与 Layout 的复杂组合
// ----------------------------------------------------------------------------
void example_2_swizzle_with_layout() {
    print_separator("示例 2: Swizzle 与复杂 Layout 组合");

    // 创建一个分块的 Layout
    // 形状：(4 个 block, 每个 block 是 4x4)
    auto blocked_layout = make_layout(
        make_shape(make_shape(4, 4), 4),  // 4x4 的块，共 4 块
        make_stride(make_stride(1, 16), 16)
    );

    std::cout << "分块 Layout:" << std::endl;
    std::cout << "  Shape: " << blocked_layout.shape() << std::endl;
    std::cout << "  Stride: " << blocked_layout.stride() << std::endl;

    // 应用 Swizzle 到每个 block 内部
    auto swizzle_fn = Swizzle<2, 2, 3>{};

    std::cout << "\n每个 Block 内的 Swizzle 映射:" << std::endl;
    for (int i = 0; i < 16; i++) {
        int swizzled = swizzle_fn.apply(i);
        std::cout << "  " << std::setw(2) << i << " -> "
                  << std::setw(2) << swizzled << std::endl;
    }
}

// ============================================================================
// 第二部分：Bank Conflict 演示与优化
// ============================================================================

// ----------------------------------------------------------------------------
// Kernel 1: 无 Swizzle - 演示 Bank Conflict
// ----------------------------------------------------------------------------
__global__ void no_swizzle_kernel(float* output, int* bank_conflicts) {
    __shared__ float shared_data[32 * 16];  // 32 行，16 列

    int tid = threadIdx.x;
    int row = tid / 32;
    int col = tid % 32;

    // 糟糕的访问模式：所有线程访问同一列
    // 这会导致严重的 Bank Conflict
    shared_data[row * 16 + 0] = static_cast<float>(tid);

    __syncthreads();

    // 读取数据
    float value = shared_data[row * 16 + 0];
    output[tid] = value;
}

// ----------------------------------------------------------------------------
// Kernel 2: 使用 Swizzle - 避免 Bank Conflict
// ----------------------------------------------------------------------------
__global__ void with_swizzle_kernel(float* output) {
    __shared__ float shared_data[32 * 16];  // 32 行，16 列

    int tid = threadIdx.x;
    int row = tid / 32;
    int col = tid % 32;

    // 使用 Swizzle 计算列地址
    int swizzled_col = Swizzle<2, 4, 3>::apply(col);

    // 好的访问模式：分散到不同 Bank
    shared_data[row * 16 + swizzled_col] = static_cast<float>(tid);

    __syncthreads();

    // 读取数据
    float value = shared_data[row * 16 + swizzled_col];
    output[tid] = value;
}

void example_3_bank_conflict_demo() {
    print_separator("示例 3: Bank Conflict 对比演示");

    const int num_threads = 32 * 16;  // 512 线程
    float* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, num_threads * sizeof(float)));

    std::cout << "测试配置：" << std::endl;
    std::cout << "  线程数：" << num_threads << std::endl;
    std::cout << "  共享内存：32x16 float 矩阵" << std::endl;
    std::cout << "  访问模式：所有线程访问第 0 列" << std::endl;

    // 测试无 Swizzle
    std::cout << "\n[无 Swizzle] 执行..." << std::endl;
    no_swizzle_kernel<<<1, num_threads>>>(d_output, nullptr);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "  完成" << std::endl;

    // 测试有 Swizzle
    std::cout << "[有 Swizzle] 执行..." << std::endl;
    with_swizzle_kernel<<<1, num_threads>>>(d_output);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "  完成" << std::endl;

    // 验证结果
    std::vector<float> h_output(num_threads);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output,
                          num_threads * sizeof(float),
                          cudaMemcpyDeviceToHost));

    std::cout << "\n结果验证 (前 32 个线程):" << std::endl;
    for (int i = 0; i < 32; i++) {
        std::cout << "  Thread " << std::setw(2) << i
                  << ": output = " << h_output[i] << std::endl;
    }

    CUDA_CHECK(cudaFree(d_output));
}

// ============================================================================
// 第三部分：Tensor 与 Swizzle 配合
// ============================================================================

// ----------------------------------------------------------------------------
// 示例 4: 创建带 Swizzle 的 Tensor Layout
// ----------------------------------------------------------------------------
void example_4_swizzle_tensor_layout() {
    print_separator("示例 4: 带 Swizzle 的 Tensor Layout");

    // 创建一个 16x16 的矩阵 Layout
    auto matrix_layout = make_layout(make_shape(16, 16));

    // 创建 Swizzle
    auto swizzle_fn = Swizzle<2, 4, 3>{};

    // 使用 composition 创建 Swizzled Layout
    auto swizzled_layout = composition(swizzle_fn, matrix_layout);

    std::cout << "原始 16x16 Layout (部分):" << std::endl;
    for (int i = 0; i < 8; i++) {
        std::cout << "  Row " << i << ": ";
        for (int j = 0; j < 8; j++) {
            std::cout << std::setw(3) << matrix_layout(i, j);
        }
        std::cout << std::endl;
    }

    std::cout << "\nSwizzled 16x16 Layout (部分):" << std::endl;
    for (int i = 0; i < 8; i++) {
        std::cout << "  Row " << i << ": ";
        for (int j = 0; j < 8; j++) {
            std::cout << std::setw(3) << swizzled_layout(i, j);
        }
        std::cout << std::endl;
    }
}

// ----------------------------------------------------------------------------
// Kernel 3: 使用 Tensor 和 Swizzle
// ----------------------------------------------------------------------------
__global__ void swizzle_tensor_kernel(float* data, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        // 创建带 Swizzle 的 Layout
        auto base_layout = make_layout(make_shape(rows, cols));
        auto swizzle_fn = Swizzle<2, 4, 3>{};
        auto layout = composition(swizzle_fn, base_layout);

        // 创建 Tensor
        auto tensor = make_tensor(data, layout);

        // 写入数据 - Swizzle 自动应用
        int idx = row * cols + col;
        tensor(row, col) = static_cast<float>(idx);
    }
}

void example_5_swizzle_tensor_kernel() {
    print_separator("示例 5: Swizzle Tensor Kernel");

    const int rows = 16;
    const int cols = 16;

    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, rows * cols * sizeof(float)));

    dim3 block(4, 4);
    dim3 grid((cols + block.x - 1) / block.x,
              (rows + block.y - 1) / block.y);

    swizzle_tensor_kernel<<<grid, block>>>(d_data, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 复制回主机
    std::vector<float> h_data(rows * cols);
    CUDA_CHECK(cudaMemcpy(h_data.data(), d_data,
                          rows * cols * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // 打印结果
    std::cout << "Swizzled Tensor 内容 (16x16):" << std::endl;
    for (int i = 0; i < rows; i++) {
        std::cout << "  Row " << std::setw(2) << i << ": ";
        for (int j = 0; j < cols; j++) {
            std::cout << std::setw(6) << h_data[i * cols + j];
        }
        std::cout << std::endl;
    }

    CUDA_CHECK(cudaFree(d_data));
}

// ============================================================================
// 第四部分：实用的 Swizzle 模式
// ============================================================================

// ----------------------------------------------------------------------------
// 示例 6: GEMM 共享内存 Layout 模式
// ----------------------------------------------------------------------------
void example_6_gemm_shared_layout() {
    print_separator("示例 6: GEMM 共享内存 Layout 模式");

    // GEMM 中常用的 16x16 tile 布局
    // 每个线程负责 8 个元素

    // 基础 Layout: 16x16 矩阵
    auto base = make_layout(make_shape(16, 16));

    // Swizzle: 避免 Bank Conflict
    auto swizzle_fn = Swizzle<2, 4, 3>{};

    // 组合后的 Layout
    auto gemm_layout = composition(swizzle_fn, base);

    std::cout << "GEMM 16x16 共享内存 Layout:" << std::endl;
    std::cout << "  总元素数：" << size(gemm_layout) << std::endl;

    // 验证同一行的访问模式
    std::cout << "\n  第 0 行各列的 Bank 分布:" << std::endl;
    for (int col = 0; col < 16; col++) {
        int offset = gemm_layout(0, col);
        int bank = (offset / 4) % 16;  // 假设每个 Bank 4 个 float
        std::cout << "    列 " << std::setw(2) << col
                  << " -> 偏移 " << std::setw(3) << offset
                  << " -> Bank " << bank << std::endl;
    }
}

// ----------------------------------------------------------------------------
// 示例 7: 不同数据类型的 Swizzle 配置
// ----------------------------------------------------------------------------
void example_7_different_dtypes() {
    print_separator("示例 7: 不同数据类型的 Swizzle 配置");

    // float32 (4 bytes)
    std::cout << "float32 (4 bytes):" << std::endl;
    std::cout << "  推荐配置：Swizzle<2, 5, 3>" << std::endl;
    std::cout << "  Bank Size: 4 元素 = 16 bytes" << std::endl;
    for (int i = 0; i < 8; i++) {
        std::cout << "    " << i << " -> " << Swizzle<2, 5, 3>::apply(i) << std::endl;
    }

    // float16/half (2 bytes)
    std::cout << "\nfloat16/half (2 bytes):" << std::endl;
    std::cout << "  推荐配置：Swizzle<3, 5, 4>" << std::endl;
    std::cout << "  Bank Size: 8 元素 = 16 bytes" << std::endl;
    for (int i = 0; i < 8; i++) {
        std::cout << "    " << i << " -> " << Swizzle<3, 5, 4>::apply(i) << std::endl;
    }

    // int8 (1 byte)
    std::cout << "\nint8 (1 byte):" << std::endl;
    std::cout << "  推荐配置：Swizzle<4, 5, 5>" << std::endl;
    std::cout << "  Bank Size: 16 元素 = 16 bytes" << std::endl;
    for (int i = 0; i < 8; i++) {
        std::cout << "    " << i << " -> " << Swizzle<4, 5, 5>::apply(i) << std::endl;
    }
}

// ============================================================================
// 第五部分：CUTE 高级 Swizzle 工具
// ============================================================================

// ----------------------------------------------------------------------------
// 示例 8: 使用 CUTE 的 swizzle_layout 工具
// ----------------------------------------------------------------------------
void example_8_cute_swizzle_tools() {
    print_separator("示例 8: CUTE 内置 Swizzle 工具");

    // 使用 CUTE 提供的 layout_swizzle_right 工具
    auto base = make_layout(make_shape(8, 8));

    std::cout << "基础 Layout:" << std::endl;
    std::cout << "  " << base << std::endl;

    // 手动创建 Swizzle Layout (推荐方式)
    auto swizzle_fn = Swizzle<2, 3, 3>{};
    auto swizzled = composition(swizzle_fn, base);

    std::cout << "\nSwizzled Layout:" << std::endl;
    std::cout << "  " << swizzled << std::endl;

    // 验证 Layout 的正确性 - 检查同一行的访问模式
    std::cout << "\n验证同一行各列的偏移:" << std::endl;
    for (int j = 0; j < 8; j++) {
        auto idx = swizzled(0, j);
        std::cout << "  (0," << j << ") -> " << idx;
        // 检查是否分散到不同 Bank
        int bank = (idx / 4) % 8;
        std::cout << " [Bank " << bank << "]" << std::endl;
    }
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  CUTE Swizzle 进阶示例代码" << std::endl;
    std::cout << "========================================" << std::endl;

    // 第一部分：Swizzle 组合
    example_1_composed_swizzle();
    example_2_swizzle_with_layout();

    // 第二部分：Bank Conflict 演示
    example_3_bank_conflict_demo();

    // 第三部分：Tensor 与 Swizzle
    example_4_swizzle_tensor_layout();
    example_5_swizzle_tensor_kernel();

    // 第四部分：实用模式
    example_6_gemm_shared_layout();
    example_7_different_dtypes();

    // 第五部分：CUTE 工具
    example_8_cute_swizzle_tools();

    std::cout << "\n========================================" << std::endl;
    std::cout << "  所有进阶示例完成！" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
