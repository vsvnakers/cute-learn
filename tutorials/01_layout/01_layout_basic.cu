/**
 * 第一课：CUTE Layout 基础
 *
 * 本示例展示 CUTE Layout 的基本概念和使用方法
 * 编译：nvcc -std=c++17 -arch=sm_80 01_layout_basic.cu -o 01_layout_basic
 */

#include <iostream>
#include <cuda_runtime.h>

// CUTE 核心头文件
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/pointer.hpp>

using namespace cute;

// ============================================================================
// 第一部分：理解 Shape 和 Layout
// ============================================================================

void test_basic_layout() {
    std::cout << "=== 测试 1: 基础 Layout ===" << std::endl;

    // 创建一个 1D Layout，大小为 8
    auto layout_1d = make_layout(make_shape(8));
    std::cout << "1D Layout (8): " << layout_1d << std::endl;

    // 创建一个 2D Layout，形状为 8x8
    auto layout_2d = make_layout(make_shape(8, 8));
    std::cout << "2D Layout (8x8): " << layout_2d << std::endl;

    // 创建一个 3D Layout，形状为 4x8x16
    auto layout_3d = make_layout(make_shape(4, 8, 16));
    std::cout << "3D Layout (4x8x16): " << layout_3d << std::endl;

    std::cout << std::endl;
}

// ============================================================================
// 第二部分：理解 Stride（步长）
// ============================================================================

void test_stride_layout() {
    std::cout << "=== 测试 2: 带 Stride 的 Layout ===" << std::endl;

    // 默认 Layout 使用行优先（row-major）stride
    // 对于 shape (M, N)，stride 为 (N, 1)
    auto layout_row_major = make_layout(make_shape(4, 8));
    std::cout << "行优先 Layout (4x8): " << layout_row_major << std::endl;
    std::cout << "  stride: " << layout_row_major.stride() << std::endl;

    // 列优先（column-major）stride
    // 对于 shape (M, N)，stride 为 (1, M)
    auto layout_col_major = make_layout(make_shape(4, 8), make_stride(1, 4));
    std::cout << "列优先 Layout (4x8): " << layout_col_major << std::endl;
    std::cout << "  stride: " << layout_col_major.stride() << std::endl;

    // 验证索引计算
    std::cout << "\n  索引验证 (row=2, col=3):" << std::endl;
    std::cout << "    行优先偏移：" << layout_row_major(2, 3) << std::endl;  // 2*8+3=19
    std::cout << "    列优先偏移：" << layout_col_major(2, 3) << std::endl;  // 3*4+2=14

    std::cout << std::endl;
}

// ============================================================================
// 第三部分：Layout 的组合与变换
// ============================================================================

void test_layout_composition() {
    std::cout << "=== 测试 3: Layout 组合 ===" << std::endl;

    // 创建一个基础 Layout
    auto base_layout = make_layout(make_shape(8, 8));

    // 使用 make_ordered_layout 创建指定顺序的 Layout
    // 需要传入 order tuple
    auto ordered_01 = make_ordered_layout(make_shape(8, 8), make_stride(0, 1));
    auto ordered_10 = make_ordered_layout(make_shape(8, 8), make_stride(1, 0));

    std::cout << "Ordered (0,1): " << ordered_01 << std::endl;
    std::cout << "Ordered (1,0): " << ordered_10 << std::endl;

    std::cout << std::endl;
}

// ============================================================================
// 第四部分：Tensor - Layout 的实际应用
// ============================================================================

__global__ void tensor_kernel(float* data) {
    // 在 device 代码中使用 Layout
    auto layout = make_layout(make_shape(8, 8));

    // 创建 Tensor - 将指针与 Layout 绑定
    auto tensor = make_tensor(data, layout);

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < 64) {
        // 使用多维索引访问
        int row = tid / 8;
        int col = tid % 8;
        tensor(row, col) = row * 10 + col;  // 自动计算偏移
    }
}

void test_tensor_usage() {
    std::cout << "=== 测试 4: Tensor 使用 ===" << std::endl;

    // 分配设备内存
    float* d_data;
    cudaMalloc(&d_data, 64 * sizeof(float));

    // 启动 kernel
    tensor_kernel<<<1, 64>>>(d_data);
    cudaDeviceSynchronize();

    // 复制回主机验证
    float h_data[64];
    cudaMemcpy(h_data, d_data, 64 * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Tensor 内容 (8x8):" << std::endl;
    for (int i = 0; i < 8; i++) {
        std::cout << "  ";
        for (int j = 0; j < 8; j++) {
            std::cout << h_data[i * 8 + j] << " ";
        }
        std::cout << std::endl;
    }

    cudaFree(d_data);
    std::cout << std::endl;
}

// ============================================================================
// 第五部分：分块 Layout（Tile Layout）
// ============================================================================

void test_tiled_layout() {
    std::cout << "=== 测试 5: 分块 Layout ===" << std::endl;

    // 创建一个 16x16 的 Layout
    auto big_layout = make_layout(make_shape(16, 16));
    std::cout << "大 Layout (16x16): " << big_layout << std::endl;

    // 使用简单索引来演示访问
    std::cout << "\n  Layout 访问示例:" << std::endl;
    std::cout << "    (0, 0) -> " << big_layout(0, 0) << std::endl;
    std::cout << "    (5, 0) -> " << big_layout(5, 0) << std::endl;
    std::cout << "    (0, 3) -> " << big_layout(0, 3) << std::endl;

    std::cout << std::endl;
}

// ============================================================================
// 第六部分：实用技巧 - 查看 Layout 信息
// ============================================================================

void test_layout_info() {
    std::cout << "=== 测试 6: Layout 信息查询 ===" << std::endl;

    auto layout = make_layout(make_shape(4, 8, 16));

    std::cout << "Shape: " << layout.shape() << std::endl;
    std::cout << "Stride: " << layout.stride() << std::endl;
    std::cout << "Rank (维度数): " << decltype(layout)::rank << std::endl;
    std::cout << "Size (总大小): " << size(layout) << std::endl;

    // 访问单个元素
    std::cout << "\n  单个维度信息:" << std::endl;
    std::cout << "    dim 0 shape: " << layout.shape<0>() << std::endl;
    std::cout << "    dim 1 shape: " << layout.shape<1>() << std::endl;
    std::cout << "    dim 2 shape: " << layout.shape<2>() << std::endl;

    std::cout << std::endl;
}

// ============================================================================
// 主函数 - 运行所有测试
// ============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  CUTE Layout 基础教程" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    test_basic_layout();
    test_stride_layout();
    test_layout_composition();
    test_tensor_usage();
    test_tiled_layout();
    test_layout_info();

    std::cout << "========================================" << std::endl;
    std::cout << "  第一课完成！" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
