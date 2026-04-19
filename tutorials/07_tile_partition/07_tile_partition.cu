/**
 * 第七课：Tile Partitioning - 分块与划分详解
 *
 * 本课深入讲解 CUTE 中的分块策略：
 * 1. Thread Level Tiling - 线程级分块
 * 2. Warp Level Tiling - Warp 级分块
 * 3. Block Level Tiling - Block 级分块
 * 4. Swizzled Partitioning - Swizzle 分块
 *
 * 编译：nvcc -std=c++17 -arch=sm_80 07_tile_partition.cu -o 07_tile_partition
 */

#include <iostream>
#include <cuda_runtime.h>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>

using namespace cute;

// ============================================================================
// 第一部分：理解 CUTE 中的分块概念
// ============================================================================

void test_basic_tiling() {
    std::cout << "=== 测试 1: 基础分块概念 ===" << std::endl;

    // 问题：如何高效处理 1024x1024 矩阵？
    // 答案：分块处理

    // 1. 定义全局矩阵大小
    constexpr int M = 1024;
    constexpr int N = 1024;

    // 2. 定义 Tile 大小
    constexpr int TILE_M = 64;
    constexpr int TILE_N = 64;

    // 3. 计算 Tile 数量
    constexpr int TILES_M = (M + TILE_M - 1) / TILE_M;  // 16
    constexpr int TILES_N = (N + TILE_N - 1) / TILE_N;  // 16

    std::cout << "全局矩阵：" << M << "x" << N << std::endl;
    std::cout << "Tile 大小：" << TILE_M << "x" << TILE_N << std::endl;
    std::cout << "Tile 数量：" << TILES_M << "x" << TILES_N << " = " << TILES_M * TILES_N << std::endl;

    // 4. 创建 Tile Layout
    // Layout 可以看作是从 (tile_m, tile_n, m, n) 到线性偏移的映射
    auto global_layout = make_layout(make_shape(M, N));
    auto tile_layout = make_layout(
        make_shape(TILE_M, TILE_N),   // Tile 内形状
        make_shape(TILES_M, TILES_N)  // Tile 间形状
    );

    std::cout << "\n全局 Layout: " << global_layout << std::endl;

    std::cout << std::endl;
}

// ============================================================================
// 第二部分：Thread Level Tiling - 线程级分块
// ============================================================================

void test_thread_tiling() {
    std::cout << "=== 测试 2: 线程级分块 ===" << std::endl;

    // 场景：一个 Block 内有 256 个线程，处理 64x64 的 Tile
    constexpr int THREADS = 256;
    constexpr int TILE_M = 64;
    constexpr int TILE_N = 64;

    // 每个线程处理的元素数量
    constexpr int ELEMENTS_PER_THREAD = (TILE_M * TILE_N) / THREADS;  // 16

    std::cout << "Block 线程数：" << THREADS << std::endl;
    std::cout << "Tile 大小：" << TILE_M << "x" << TILE_N << " = " << TILE_M * TILE_N << std::endl;
    std::cout << "每线程处理：" << ELEMENTS_PER_THREAD << " 元素" << std::endl;

    // 方法 1：1D 分块 - 每个线程处理连续的元素
    auto thread_layout_1d = make_layout(make_shape(THREADS));
    std::cout << "\n1D 线程布局：" << thread_layout_1d << std::endl;

    // 方法 2：2D 分块 - 更利于内存访问
    // 假设线程组织为 16x16
    auto thread_layout_2d = make_layout(make_shape(16, 16));
    std::cout << "2D 线程布局 (16x16): " << thread_layout_2d << std::endl;

    // 方法 3：带 Swizzle 的分块 - 避免 Bank Conflict
    // 使用 8x32 线程组织，列维度添加 Swizzle
    auto thread_layout_swizzled = make_layout(
        make_shape(8, 32),
        make_stride(32, 1)  // 列优先
    );
    std::cout << "Swizzled 线程布局 (8x32): " << thread_layout_swizzled << std::endl;

    std::cout << std::endl;
}

// ============================================================================
// 第三部分：Warp Level Tiling - Warp 级分块
// ============================================================================

void test_warp_tiling() {
    std::cout << "=== 测试 3: Warp 级分块 ===" << std::endl;

    // 在 Ampere GPU 上，一个 Warp 有 32 个线程
    constexpr int WARP_SIZE = 32;
    constexpr int WARPS_PER_BLOCK = 8;  // 256 线程 / 32 = 8 Warps

    // Tensor Core MMA 的基本尺寸 (SM80)
    // 16x8x8 FP16 MMA
    constexpr int MMA_M = 16;
    constexpr int MMA_N = 8;
    constexpr int MMA_K = 8;

    std::cout << "Warp 大小：" << WARP_SIZE << " 线程" << std::endl;
    std::cout << "每 Block Warp 数：" << WARPS_PER_BLOCK << std::endl;
    std::cout << "MMA 尺寸：" << MMA_M << "x" << MMA_N << "x" << MMA_K << std::endl;

    // Warp 级分块策略：
    // 将 64x64 的 Tile 分配给 8 个 Warp
    // 每个 Warp 负责 32x32 的区域

    // Warp Layout: 8 Warps 组织为 2x4
    auto warp_layout = make_layout(
        make_shape(2, 4),  // 2 Warps in M, 4 Warps in N
        make_stride(4, 1)
    );

    // 每个 Warp 处理的区域
    constexpr int WARP_TILE_M = 64 / 2;  // 32
    constexpr int WARP_TILE_N = 64 / 4;  // 16

    std::cout << "\nWarp 布局：" << warp_layout << std::endl;
    std::cout << "每 Warp 处理：" << WARP_TILE_M << "x" << WARP_TILE_N << std::endl;

    // 在每个 Warp 内，使用 MMA 指令
    // 32x16 的区域需要多个 MMA 操作

    std::cout << std::endl;
}

// ============================================================================
// 第四部分：CUTE 分块 API 演示
// ============================================================================

void test_cute_partition_api() {
    std::cout << "=== 测试 4: CUTE 分块 API ===" << std::endl;

    // CUTE 提供了多种分块相关的 API

    // 1. 创建一个 128x128 的 Layout
    auto layout = make_layout(make_shape(128, 128));
    std::cout << "原始 Layout: " << layout << std::endl;

    // 2. 使用 slice 获取子区域
    // local_tile 需要 4 个参数：layout, tile_shape, cluster_shape, coord
    // 简化演示：直接计算子区域
    auto tile_00 = make_layout(make_shape(32, 32));
    std::cout << "Tile (0,0) 32x32: " << tile_00 << std::endl;

    auto tile_01 = make_layout(make_shape(32, 32));
    std::cout << "Tile (0,1) 32x32: " << tile_01 << std::endl;

    // 3. 分块迭代
    // 将 128x128 分块为 4x4 个 32x32 的 Tile
    constexpr int TILE_SIZE = 32;
    constexpr int NUM_TILES_M = 128 / TILE_SIZE;  // 4
    constexpr int NUM_TILES_N = 128 / TILE_SIZE;  // 4

    std::cout << "\n分块迭代 (4x4 Tiles):" << std::endl;
    for (int i = 0; i < NUM_TILES_M; i++) {
        for (int j = 0; j < NUM_TILES_N; j++) {
            auto tile = make_layout(make_shape(TILE_SIZE, TILE_SIZE));
            std::cout << "  Tile(" << i << "," << j << "): size=" << size(tile) << std::endl;
        }
    }

    std::cout << std::endl;
}

// ============================================================================
// 第五部分：实际的 Tile Kernel
// ============================================================================

template<int BLOCK_M, int BLOCK_N, int THREADS_PER_BLOCK>
__global__ void tile_copy_kernel(
    const float* input, float* output,
    int width, int height) {

    // 计算 Block 索引
    int tile_x = blockIdx.x;
    int tile_y = blockIdx.y;

    // 计算线程在 Block 内的索引
    int tid = threadIdx.x;
    int thread_x = tid % BLOCK_N;
    int thread_y = tid / BLOCK_N;

    // 计算全局坐标
    int x = tile_x * BLOCK_N + thread_x;
    int y = tile_y * BLOCK_M + thread_y;

    // 边界检查
    if (x < width && y < height) {
        int idx = y * width + x;
        output[idx] = input[idx];
    }
}

void test_tile_kernel() {
    std::cout << "=== 测试 5: Tile Kernel 演示 ===" << std::endl;

    // 参数
    constexpr int WIDTH = 256;
    constexpr int HEIGHT = 256;
    constexpr int BLOCK_M = 32;
    constexpr int BLOCK_N = 32;

    // 分配内存
    size_t size = WIDTH * HEIGHT * sizeof(float);
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    // 初始化
    float* h_data = new float[WIDTH * HEIGHT];
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        h_data[i] = float(i);
    }
    cudaMemcpy(d_input, h_data, size, cudaMemcpyHostToDevice);

    // 配置 Kernel
    dim3 block(BLOCK_M * BLOCK_N);  // 1024 线程
    dim3 grid((WIDTH + BLOCK_N - 1) / BLOCK_N,
              (HEIGHT + BLOCK_M - 1) / BLOCK_M);

    // 启动 Kernel
    tile_copy_kernel<BLOCK_M, BLOCK_N, 1024>
        <<<grid, block>>>(d_input, d_output, WIDTH, HEIGHT);

    // 验证
    float* h_output = new float[WIDTH * HEIGHT];
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    bool passed = true;
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        if (h_output[i] != h_data[i]) {
            passed = false;
            break;
        }
    }

    std::cout << "Tile Kernel 验证：" << (passed ? "PASS" : "FAIL") << std::endl;
    std::cout << "  Block 配置：" << BLOCK_M << "x" << BLOCK_N << std::endl;
    std::cout << "  Grid 配置：" << grid.x << "x" << grid.y << " Blocks" << std::endl;

    delete[] h_data;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    std::cout << std::endl;
}

// ============================================================================
// 第六部分：多级分块策略
// ============================================================================

void test_multi_level_tiling() {
    std::cout << "=== 测试 6: 多级分块策略 ===" << std::endl;

    // 完整的多级分块层次：
    // Global (M x N) -> Block Level -> Warp Level -> Thread Level -> MMA Level

    // 示例：GEMM 中的分块层次
    std::cout << "GEMM 分块层次 (1024x1024 矩阵):" << std::endl;

    // 1. Block Level: 128x128 per Block
    std::cout << "  Block Level: 128x128 (需要 8x8=64 Blocks)" << std::endl;

    // 2. Warp Level: 32x32 per Warp
    std::cout << "  Warp Level:  32x32 (每个 Block 4x4=16 Warps)" << std::endl;

    // 3. Thread Level: 8x8 per Thread
    std::cout << "  Thread Level: 8x8 (每个 Warp 2x2=4 Threads)" << std::endl;

    // 4. MMA Level: 使用 Tensor Core
    std::cout << "  MMA Level:   16x8x8 (Tensor Core 指令)" << std::endl;

    // 数据流：
    // Global Memory -> Shared Memory (128x128)
    //              -> Register File (32x32 per Warp)
    //              -> MMA Accumulators (16x8 per Thread)

    std::cout << "\n数据流:" << std::endl;
    std::cout << "  全局内存 (GB) -> 共享内存 (KB) -> 寄存器 (B) -> MMA" << std::endl;
}

// ============================================================================
// 第七部分：Swizzled Partitioning - 避免 Bank Conflict
// ============================================================================

void test_swizzled_partition() {
    std::cout << "=== 测试 7: Swizzled 分块 ===" << std::endl;

    // 在共享内存中，正确的分块可以避免 Bank Conflict
    // 方法 1: Padding
    // 方法 2: Swizzle
    // 方法 3: XOR Hash

    // Padding 示例
    constexpr int COLS_PADDED = 128 + 1;  // +1 padding
    std::cout << "Padding 策略：128 -> " << COLS_PADDED << " (避免 32-way conflict)" << std::endl;

    // Swizzle 示例
    // 使用位操作重新排列地址
    auto swizzle_func = [] (int offset) -> int {
        return offset ^ ((offset >> 2) & 0x7);
    };

    std::cout << "Swizzle 策略：offset ^ ((offset >> 2) & 0x7)" << std::endl;
    std::cout << "  0 -> " << (0 ^ ((0 >> 2) & 0x7)) << std::endl;
    std::cout << "  8 -> " << (8 ^ ((8 >> 2) & 0x7)) << std::endl;
    std::cout << "  16 -> " << (16 ^ ((16 >> 2) & 0x7)) << std::endl;

    // CUTE Swizzle Layout
    using Swizzle = cute::Swizzle<2, 5, 3>;  // B=2, M=5, S=3
    auto base_layout = make_layout(make_shape(32, 32));
    auto swizzled_layout = composition(Swizzle{}, base_layout);

    std::cout << "\nCUTE Swizzle Layout:" << std::endl;
    std::cout << "  基础：" << base_layout << std::endl;
    std::cout << "  Swizzled: " << swizzled_layout << std::endl;

    std::cout << std::endl;
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  第七课：Tile Partitioning 详解" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    test_basic_tiling();
    test_thread_tiling();
    test_warp_tiling();
    test_cute_partition_api();
    test_tile_kernel();
    test_multi_level_tiling();
    test_swizzled_partition();

    std::cout << "========================================" << std::endl;
    std::cout << "  第七课完成！" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
