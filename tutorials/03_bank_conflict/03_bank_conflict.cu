/**
 * 第三课：Bank Conflict 详解与解决方案
 *
 * 本示例演示 Bank Conflict 的发生和解决方法
 * 编译：nvcc -std=c++17 -arch=sm_80 03_bank_conflict.cu -o 03_bank_conflict
 *
 * 运行建议：使用 nvprof 或 nsys 分析性能差异
 *   nvprof ./03_bank_conflict
 */

#include <iostream>
#include <cuda_runtime.h>
#include <cute/layout.hpp>
#include <cute/swizzle.hpp>
#include <cute/tensor.hpp>

using namespace cute;

// ============================================================================
// 第一部分：Bank Conflict 演示 - 列访问模式
// ============================================================================

// 坏例子：列访问导致 Bank Conflict
__global__ void bad_column_access(float* output, int size) {
    __shared__ float smem[32 * 32];

    int row = threadIdx.y;
    int col = threadIdx.x;
    int tid = threadIdx.y * 32 + threadIdx.x;

    // 初始化共享内存
    smem[row * 32 + col] = tid;
    __syncthreads();

    // 坏模式：所有线程访问同一列
    // 线程 (0,0), (1,0), (2,0), ... (31,0) 都访问 Bank 0
    // 造成 32-way Bank Conflict!
    float val = smem[row * 32 + 0];

    output[tid] = val;
}

// 好例子：行访问无 Bank Conflict
__global__ void good_row_access(float* output, int size) {
    __shared__ float smem[32 * 32];

    int row = threadIdx.y;
    int col = threadIdx.x;
    int tid = threadIdx.y * 32 + threadIdx.x;

    // 初始化共享内存
    smem[row * 32 + col] = tid;
    __syncthreads();

    // 好模式：每行访问不同列
    // 线程 (0,0), (0,1), (0,2), ... (0,31) 访问 Bank 0-31
    // 无 Bank Conflict!
    float val = smem[0 * 32 + col];

    output[tid] = val;
}

// 最好例子：使用 Swizzle 解决 Bank Conflict
__global__ void swizzled_access(float* output, int size) {
    __shared__ float smem[32 * 32];

    int row = threadIdx.y;
    int col = threadIdx.x;
    int tid = threadIdx.y * 32 + threadIdx.x;

    // 使用 Swizzle - SShift 必须 >= BBits
    // Swizzle<BBits, MBase, SShift>

    // Swizzled 写入
    int col_swizzled = Swizzle<2, 5, 3>::apply(col);
    smem[row * 32 + col_swizzled] = tid;
    __syncthreads();

    // Swizzled 读取
    int read_col_swizzled = Swizzle<2, 5, 3>::apply(0);
    float val = smem[row * 32 + read_col_swizzled];

    output[tid] = val;
}

// ============================================================================
// 第二部分：Padding 技术
// ============================================================================

// 使用 Padding 避免 Bank Conflict
__global__ void padded_access(float* output, int size) {
    // 添加 padding：32 列变成 33 列
    __shared__ float smem[32 * 33];

    int row = threadIdx.y;
    int col = threadIdx.x;
    int tid = threadIdx.y * 32 + threadIdx.x;

    // 写入
    smem[row * 33 + col] = tid;
    __syncthreads();

    // 列访问：由于 padding，现在分散到不同 Bank
    // row 0: Bank 0
    // row 1: Bank 1 (因为 33 % 32 = 1)
    // row 2: Bank 2
    // ...
    float val = smem[row * 33 + 0];

    output[tid] = val;
}

// ============================================================================
// 第三部分：性能测试框架
// ============================================================================

#define WARMUP_ITER 10
#define TEST_ITER 100

template<typename Kernel, typename... Args>
float measure_kernel(Kernel kernel, dim3 grid, dim3 block, Args... args) {
    // Warmup
    for (int i = 0; i < WARMUP_ITER; i++) {
        kernel<<<grid, block>>>(args...);
    }
    cudaDeviceSynchronize();

    // 测量
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < TEST_ITER; i++) {
        kernel<<<grid, block>>>(args...);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return elapsed_time / TEST_ITER;  // 平均每次 kernel 时间
}

void run_performance_test() {
    std::cout << "=== 性能测试 ===" << std::endl;
    std::cout << "测试配置：32x32 线程块，" << TEST_ITER << " 次迭代" << std::endl;
    std::cout << std::endl;

    int size = 32 * 32;
    float *d_output;
    cudaMalloc(&d_output, size * sizeof(float));

    dim3 grid(1);
    dim3 block(32, 32);

    // 测试各种访问模式
    float t_bad = measure_kernel(bad_column_access, grid, block, d_output, size);
    std::cout << "坏模式 (列访问): " << t_bad << " ms" << std::endl;

    float t_good = measure_kernel(good_row_access, grid, block, d_output, size);
    std::cout << "好模式 (行访问): " << t_good << " ms" << std::endl;

    float t_swizzle = measure_kernel(swizzled_access, grid, block, d_output, size);
    std::cout << "Swizzle 模式： " << t_swizzle << " ms" << std::endl;

    float t_padding = measure_kernel(padded_access, grid, block, d_output, size);
    std::cout << "Padding 模式： " << t_padding << " ms" << std::endl;

    std::cout << std::endl;
    std::cout << "性能对比:" << std::endl;
    std::cout << "  好模式 vs 坏模式：" << t_bad / t_good << "x 更快" << std::endl;
    std::cout << "  Swizzle vs 坏模式：" << t_bad / t_swizzle << "x 更快" << std::endl;
    std::cout << "  Padding vs 坏模式：" << t_bad / t_padding << "x 更快" << std::endl;

    cudaFree(d_output);
}

// ============================================================================
// 第四部分：使用 CUTE Layout 避免 Bank Conflict
// ============================================================================

void test_cute_layout_bank_conflict() {
    std::cout << "\n=== CUTE Layout Bank Conflict 分析 ===" << std::endl;

    // 问题 Layout：直接映射导致 Bank Conflict
    auto bad_layout = make_layout(make_shape(32, 32));
    std::cout << "基础 Layout (32x32):" << std::endl;
    std::cout << "  " << bad_layout << std::endl;
    std::cout << "  stride: " << bad_layout.stride() << std::endl;

    // 分析第一列的 Bank 分布
    std::cout << "  第一列访问的 Bank:" << std::endl;
    for (int row = 0; row < 8; row++) {
        int offset = bad_layout(row, 0);
        int bank = (offset * 4) % 128 / 4;  // 4 字节 per word
        std::cout << "    row " << row << ": offset=" << offset << ", Bank=" << bank << std::endl;
    }

    // 解决方案：使用 composition 创建 Swizzle Layout
    auto swizzle_fn = Swizzle<2, 5, 3>{};  // SShift >= BBits
    auto good_layout = composition(swizzle_fn, make_layout(make_shape(32, 32)));
    std::cout << "\nSwizzled Layout (32x32):" << std::endl;
    std::cout << "  " << good_layout << std::endl;

    // 分析 Swizzled 后的 Bank 分布
    std::cout << "  第一列访问的 Bank (Swizzled):" << std::endl;
    for (int row = 0; row < 8; row++) {
        int offset = good_layout(row, 0);
        int bank = offset % 32;
        std::cout << "    row " << row << ": offset=" << offset << ", Bank=" << bank << std::endl;
    }
}

// ============================================================================
// 第五部分：矩阵转置中的 Bank Conflict
// ============================================================================

// 坏例子：直接转置
__global__ void bad_transpose(float* out, float* in, int width) {
    __shared__ float smem[32 * 32];

    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.y * 32 + threadIdx.x;

    // 读取：行访问模式（好）
    int in_idx = row * width + col;
    smem[tid] = in[in_idx];
    __syncthreads();

    // 写入：转置后变成列访问（坏！）
    int out_row = col;
    int out_col = row;
    int out_idx = out_row * width + out_col;
    out[out_idx] = smem[tid];
}

// 好例子：使用 Swizzle 转置
__global__ void good_transpose(float* out, float* in, int width) {
    __shared__ float smem[32 * 32];

    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    // Swizzle<BBits, MBase, SShift> - SShift 必须 >= BBits
    constexpr int BBits = 2;
    constexpr int MBase = 5;
    constexpr int SShift = 3;

    // 读取
    int tid = threadIdx.y * 32 + threadIdx.x;
    int in_idx = row * width + col;
    smem[tid] = in[in_idx];
    __syncthreads();

    // 写入：使用 Swizzle
    int out_row = col;
    int out_col = row;
    int out_idx = out_row * width + out_col;

    // 关键：使用 Swizzle 分散写入地址
    int swizzled_tid = Swizzle<BBits, MBase, SShift>::apply(out_col);
    out[out_idx] = smem[swizzled_tid];
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Bank Conflict 详解与解决方案" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    // 运行性能测试
    run_performance_test();

    // CUTE Layout 分析
    test_cute_layout_bank_conflict();

    std::cout << "\n========================================" << std::endl;
    std::cout << "  第三课完成！" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
