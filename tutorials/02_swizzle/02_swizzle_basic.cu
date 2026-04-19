/**
 * 第二课：CUTE Swizzle 技术
 *
 * 本示例展示 Swizzle 的工作原理和使用方法
 * 编译：nvcc -std=c++17 -arch=sm_80 02_swizzle_basic.cu -o 02_swizzle_basic
 */

#include <iostream>
#include <cuda_runtime.h>

// CUTE 核心头文件
#include <cute/swizzle.hpp>
#include <cute/layout.hpp>
#include <cute/swizzle_layout.hpp>
#include <cute/tensor.hpp>

using namespace cute;

// ============================================================================
// 第一部分：理解 Swizzle 函数
// ============================================================================

void test_basic_swizzle() {
    std::cout << "=== 测试 1: 基础 Swizzle 函数 ===" << std::endl;

    // Swizzle<2,4,2> 的含义：
    // - 第一个参数 2: log2(bank_size)，表示每个 Bank 的大小为 2^2=4 元素
    // - 第二个参数 4: log2(num_banks)，表示 Bank 数量为 2^4=16
    // - 第三个参数 2: 用于 XOR 运算的位移量
    auto swz = Swizzle<2, 4, 2>{};

    std::cout << "Swizzle<2,4,2> 映射表:" << std::endl;
    std::cout << "  原始偏移 -> Swizzled 偏移" << std::endl;
    for (int i = 0; i < 32; i++) {
        std::cout << "  " << i << " -> " << Swizzle<2, 4, 2>::apply(i) << std::endl;
    }

    std::cout << std::endl;
}

// ============================================================================
// 第二部分：Swizzle Layout - 将 Swizzle 应用于 Layout
// ============================================================================

void test_swizzle_layout() {
    std::cout << "=== 测试 2: Swizzle Layout ===" << std::endl;

    // 创建一个 8x8 的基础 Layout
    auto base_layout = make_layout(make_shape(8, 8));
    std::cout << "基础 Layout: " << base_layout << std::endl;

    // 创建 Swizzle Layout - 使用 composition
    // Swizzle<2,4,2> 应用于列维度
    auto swizzle_fn = Swizzle<2, 4, 2>{};
    auto swizzled_layout = composition(swizzle_fn, base_layout);
    std::cout << "Swizzled Layout: " << swizzled_layout << std::endl;

    // 验证同一行的访问
    std::cout << "\n  验证同一行的访问:" << std::endl;
    std::cout << "  基础 Layout 访问列 0-7:" << std::endl;
    for (int i = 0; i < 8; i++) {
        std::cout << "    (" << 0 << "," << i << ") -> " << base_layout(0, i) << std::endl;
    }

    std::cout << "  Swizzled Layout 访问列 0-7:" << std::endl;
    for (int i = 0; i < 8; i++) {
        std::cout << "    (" << 0 << "," << i << ") -> " << swizzled_layout(0, i) << std::endl;
    }

    std::cout << std::endl;
}

// ============================================================================
// 第三部分：Bank Conflict 演示
// ============================================================================

__global__ void bank_conflict_kernel(float* output, bool use_swizzle) {
    __shared__ float shared_data[32 * 32];

    int tid = threadIdx.x;
    int row = threadIdx.y;

    // 没有 Swizzle：直接访问
    // 当多个线程访问同一列时会发生 Bank Conflict
    if (!use_swizzle) {
        // 所有线程访问第 0 列 - 严重的 Bank Conflict
        shared_data[row * 32 + 0] = tid;
    } else {
        // 使用 Swizzle：对列索引进行 XOR 变换
        int col_swizzled = Swizzle<2, 5, 3>::apply(0);  // SShift >= BBits
        shared_data[row * 32 + col_swizzled] = tid;
    }

    __syncthreads();

    // 输出结果
    if (tid < 32 && row == 0) {
        output[tid] = shared_data[tid * 32 + (use_swizzle ? Swizzle<2, 5, 3>::apply(0) : 0)];
    }
}

void test_bank_conflict() {
    std::cout << "=== 测试 3: Bank Conflict 对比 ===" << std::endl;

    float* d_output;
    cudaMalloc(&d_output, 32 * sizeof(float));

    // 测试无 Swizzle
    bank_conflict_kernel<<<1, dim3(32, 32)>>>(d_output, false);
    cudaDeviceSynchronize();
    std::cout << "  无 Swizzle 执行完成" << std::endl;

    // 测试有 Swizzle
    bank_conflict_kernel<<<1, dim3(32, 32)>>>(d_output, true);
    cudaDeviceSynchronize();
    std::cout << "  有 Swizzle 执行完成" << std::endl;

    cudaFree(d_output);
    std::cout << std::endl;
}

// ============================================================================
// 第四部分：Swizzle 参数详解
// ============================================================================

void test_swizzle_params() {
    std::cout << "=== 测试 4: Swizzle 参数详解 ===" << std::endl;

    // 不同的 Swizzle 配置
    // 注意：SShift 的绝对值必须 >= BBits
    auto swz_a = Swizzle<2, 4, 3>{};  // 常用配置
    auto swz_b = Swizzle<3, 4, 4>{};  // 更大的 Bank
    auto swz_c = Swizzle<2, 5, 3>{};  // 更多 Bank

    std::cout << "Swizzle<2,4,3> (Bank=4, 数量=16):" << std::endl;
    for (int i = 0; i < 16; i++) {
        std::cout << "  " << i << " -> " << Swizzle<2, 4, 3>::apply(i) << std::endl;
    }

    std::cout << "\nSwizzle<3,4,4> (Bank=8, 数量=16):" << std::endl;
    for (int i = 0; i < 16; i++) {
        std::cout << "  " << i << " -> " << Swizzle<3, 4, 4>::apply(i) << std::endl;
    }

    std::cout << std::endl;
}

// ============================================================================
// 第五部分：实用的 Swizzle Layout 模式
// ============================================================================

void test_practical_swizzle() {
    std::cout << "=== 测试 5: 实用 Swizzle 模式 ===" << std::endl;

    // 模式 1: 共享内存矩阵 Layout（用于 GEMM）
    // 16x16 矩阵，使用 Swizzle 避免 Bank Conflict
    auto base_layout = make_layout(make_shape(16, 16));
    auto swizzle_fn = Swizzle<2, 4, 3>{};  // SShift 必须 >= BBits
    auto smem_layout = composition(swizzle_fn, base_layout);
    std::cout << "GEMM 共享内存 Layout: " << smem_layout << std::endl;

    // 模式 2: Vectorized 访问 Layout
    // 4 元素 vector 访问
    auto vec_layout = make_layout(
        make_shape(make_shape(4, 4), 8),  // 4x4 块，每块 8 元素
        make_stride(make_stride(1, 16), 4)
    );
    std::cout << "Vectorized Layout: " << vec_layout << std::endl;

    std::cout << std::endl;
}

// ============================================================================
// 第六部分：Swizzle 与 Tensor 配合使用
// ============================================================================

__global__ void swizzle_tensor_kernel(float* data) {
    // 创建带 Swizzle 的 Layout - 使用 composition
    auto base_layout = make_layout(make_shape(8, 8));
    auto swizzle_fn = Swizzle<2, 4, 3>{};  // SShift 必须 >= BBits
    auto layout = composition(swizzle_fn, base_layout);

    // 创建 Tensor
    auto tensor = make_tensor(data, layout);

    int tid = threadIdx.x;
    if (tid < 64) {
        int row = tid / 8;
        int col = tid % 8;
        // 写入数据 - 注意 Layout 会自动应用 Swizzle
        tensor(row, col) = row * 10 + col;
    }
}

void test_swizzle_tensor() {
    std::cout << "=== 测试 6: Swizzle Tensor ===" << std::endl;

    float* d_data;
    cudaMalloc(&d_data, 64 * sizeof(float));

    swizzle_tensor_kernel<<<1, 64>>>(d_data);
    cudaDeviceSynchronize();

    // 复制回主机
    float h_data[64];
    cudaMemcpy(h_data, d_data, 64 * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Swizzled Tensor 内容 (8x8):" << std::endl;
    std::cout << "注意：物理存储顺序已被 Swizzle 重排" << std::endl;
    for (int i = 0; i < 8; i++) {
        std::cout << "  Row " << i << ": ";
        for (int j = 0; j < 8; j++) {
            std::cout << h_data[i * 8 + j] << " ";
        }
        std::cout << std::endl;
    }

    cudaFree(d_data);
    std::cout << std::endl;
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  CUTE Swizzle 技术教程" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    test_basic_swizzle();
    test_swizzle_layout();
    test_bank_conflict();
    test_swizzle_params();
    test_practical_swizzle();
    test_swizzle_tensor();

    std::cout << "========================================" << std::endl;
    std::cout << "  第二课完成！" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
