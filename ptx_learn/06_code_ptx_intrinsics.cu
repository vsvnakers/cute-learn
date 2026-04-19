// PTX 内联汇编实战示例

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <cstring>

// ============================================================================
// 示例 1: 使用 PTX 实现快速数学库
// ============================================================================

/**
 * 快速平方根 (使用近似指令)
 * 精度：约 1% 误差
 * 速度：比标准 sqrt 快 2-3 倍
 */
__device__ __forceinline__ float fast_sqrt(float x) {
    float result;
    asm volatile("sqrt.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
}

/**
 * 快速倒数 (使用近似指令)
 * 精度：约 0.01% 误差
 * 速度：比除法快 3-4 倍
 */
__device__ __forceinline__ float fast_rcp(float x) {
    float result;
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
}

/**
 * 快速正弦 (使用近似指令)
 */
__device__ __forceinline__ float fast_sin(float x) {
    float result;
    asm volatile("sin.approx.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
}

/**
 * 快速余弦 (使用近似指令)
 */
__device__ __forceinline__ float fast_cos(float x) {
    float result;
    asm volatile("cos.approx.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
}

/**
 * 快速指数 (使用近似指令)
 */
__device__ __forceinline__ float fast_exp(float x) {
    float result;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
}

/**
 * 快速对数 (使用近似指令)
 */
__device__ __forceinline__ float fast_log2(float x) {
    float result;
    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
}

// ============================================================================
// 示例 2: 使用 PTX 实现 SIMD 优化
// ============================================================================

/**
 * 使用 PTX 实现 4 路 SIMD 向量加法
 * 一次处理 4 个元素
 */
__global__ void ptx_simd_add(float* d_out, const float* d_a, const float* d_b, int n) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (idx + 3 < n) {
        float a0, a1, a2, a3;
        float b0, b1, b2, b3;
        float c0, c1, c2, c3;

        // 并行加载 4 个元素
        asm volatile(
            "ld.global.f32 %0, [%4];\n\t"
            "ld.global.f32 %1, [%5];\n\t"
            "ld.global.f32 %2, [%6];\n\t"
            "ld.global.f32 %3, [%7];\n\t"
            : "=f"(a0), "=f"(a1), "=f"(a2), "=f"(a3)
            : "l"(d_a + idx), "l"(d_a + idx + 1), "l"(d_a + idx + 2), "l"(d_a + idx + 3)
        );

        asm volatile(
            "ld.global.f32 %0, [%4];\n\t"
            "ld.global.f32 %1, [%5];\n\t"
            "ld.global.f32 %2, [%6];\n\t"
            "ld.global.f32 %3, [%7];\n\t"
            : "=f"(b0), "=f"(b1), "=f"(b2), "=f"(b3)
            : "l"(d_b + idx), "l"(d_b + idx + 1), "l"(d_b + idx + 2), "l"(d_b + idx + 3)
        );

        // 并行加法
        asm volatile("add.f32 %0, %1, %2;" : "=f"(c0) : "f"(a0), "f"(b0));
        asm volatile("add.f32 %0, %1, %2;" : "=f"(c1) : "f"(a1), "f"(b1));
        asm volatile("add.f32 %0, %1, %2;" : "=f"(c2) : "f"(a2), "f"(b2));
        asm volatile("add.f32 %0, %1, %2;" : "=f"(c3) : "f"(a3), "f"(b3));

        // 并行存储
        asm volatile(
            "st.global.f32 [%0], %1;\n\t"
            "st.global.f32 [%2], %3;\n\t"
            "st.global.f32 [%4], %5;\n\t"
            "st.global.f32 [%6], %7;\n\t"
            : : "l"(d_out + idx), "f"(c0),
                "l"(d_out + idx + 1), "f"(c1),
                "l"(d_out + idx + 2), "f"(c2),
                "l"(d_out + idx + 3), "f"(c3)
        );
    }
}

// ============================================================================
// 示例 3: 使用 PTX FMA 实现点积优化
// ============================================================================

/**
 * 使用 FMA 实现 4 元素点积
 * fma.rn.f32: result = a * b + c
 */
__device__ float ptx_dot4(const float* a, const float* b) {
    float a0, a1, a2, a3;
    float b0, b1, b2, b3;
    float result;

    // 加载 a 向量
    asm volatile(
        "ld.global.f32 %0, [%4];\n\t"
        "ld.global.f32 %1, [%5];\n\t"
        "ld.global.f32 %2, [%6];\n\t"
        "ld.global.f32 %3, [%7];\n\t"
        : "=f"(a0), "=f"(a1), "=f"(a2), "=f"(a3)
        : "l"(a), "l"(a+1), "l"(a+2), "l"(a+3)
    );

    // 加载 b 向量
    asm volatile(
        "ld.global.f32 %0, [%4];\n\t"
        "ld.global.f32 %1, [%5];\n\t"
        "ld.global.f32 %2, [%6];\n\t"
        "ld.global.f32 %3, [%7];\n\t"
        : "=f"(b0), "=f"(b1), "=f"(b2), "=f"(b3)
        : "l"(b), "l"(b+1), "l"(b+2), "l"(b+3)
    );

    // 使用 FMA 链式计算点积
    asm volatile("fma.rn.f32 %0, %1, %2, %3;"
                 : "=f"(result)
                 : "f"(a0), "f"(b0), "f"(a1 * b1));

    asm volatile("fma.rn.f32 %0, %1, %2, %3;"
                 : "=f"(result)
                 : "f"(a2), "f"(b2), "f"(result));

    asm volatile("fma.rn.f32 %0, %1, %2, %3;"
                 : "=f"(result)
                 : "f"(a3), "f"(b3), "f"(result));

    return result;
}

// ============================================================================
// 示例 4: 使用 PTX 实现位操作优化
// ============================================================================

/**
 * 使用 PTX 实现快速位计数
 */
__device__ __forceinline__ unsigned int ptx_popc(unsigned int x) {
    unsigned int result;
    asm volatile("popc.b32 %0, %1;" : "=r"(result) : "r"(x));
    return result;
}

/**
 * 使用 PTX 实现快速前导零计数
 */
__device__ __forceinline__ unsigned int ptx_clz(unsigned int x) {
    unsigned int result;
    asm volatile("clz.b32 %0, %1;" : "=r"(result) : "r"(x));
    return result;
}

/**
 * 使用 PTX 实现快速位查找
 * 注意：ffs 指令在某些架构上不可用，使用 clz 替代
 */
__device__ __forceinline__ unsigned int ptx_ffs(unsigned int x) {
    // ffs 返回最低位 1 的位置 (从 1 开始)
    // 使用 __ffs 内置函数
    return __ffs((int)x);
}

/**
 * 使用 PTX 实现位反转
 */
__device__ __forceinline__ unsigned int ptx_brev(unsigned int x) {
    unsigned int result;
    asm volatile("brev.b32 %0, %1;" : "=r"(result) : "r"(x));
    return result;
}

__global__ void ptx_bitwise_demo() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < 8) {
        unsigned int x = (idx + 1) * 0x0F0F0F0F;

        unsigned int popc_result = ptx_popc(x);
        unsigned int clz_result = ptx_clz(x);
        unsigned int ffs_result = ptx_ffs(x);
        unsigned int brev_result = ptx_brev(x);

        printf("线程 %d: x=0x%08x\n", idx, x);
        printf("  popc(x)=%u (1 的个数)\n", popc_result);
        printf("  clz(x)=%u (前导零个数)\n", clz_result);
        printf("  ffs(x)=%u (第一个 1 的位置)\n", ffs_result);
        printf("  brev(x)=0x%08x (位反转)\n", brev_result);
    }
}

// ============================================================================
// 示例 5: 使用 PTX 实现矩阵转置优化
// ============================================================================

/**
 * 使用共享内存和 PTX 实现高效的矩阵转置
 * Tile 大小：32x32
 */
constexpr int TILE_SIZE = 32;

__global__ void ptx_matrix_transpose(
    float* d_out, const float* d_in,
    int width, int height
) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];  // +1 避免 bank conflict

    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    // 读取输入
    if (x < width && y < height) {
        int in_idx = y * width + x;
        float val;
        asm volatile("ld.global.f32 %0, [%1];"
                     : "=f"(val) : "l"(d_in + in_idx));
        tile[threadIdx.y][threadIdx.x] = val;
    }

    __syncthreads();

    // 写入转置结果
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;

    if (x < height && y < width) {
        int out_idx = y * height + x;
        float val = tile[threadIdx.x][threadIdx.y];
        asm volatile("st.global.f32 [%0], %1;"
                     : : "l"(d_out + out_idx), "f"(val));
    }
}

// ============================================================================
// 示例 6: 使用 PTX 实现原子操作锁
// ============================================================================

/**
 * 使用 PTX CAS 实现自旋锁
 */
__device__ void ptx_spin_lock(unsigned int* lock) {
    unsigned int expected = 0;
    unsigned int desired = 1;

    // 自旋直到成功获取锁
    while (true) {
        unsigned int old;
        asm volatile("atom.global.cas.b32 %0, [%1], %2, %3;"
                     : "=r"(old)
                     : "l"(lock), "r"(expected), "r"(desired)
                     : "memory");

        if (old == expected) {
            break;  // 成功获取锁
        }

        expected = 0;  // 重置期望值
        // 可选：添加退避策略
    }
}

__device__ void ptx_spin_unlock(unsigned int* lock) {
    // 释放锁：将锁设置为 0
    asm volatile("st.global.relaxed.sys.u32 [%0], 0;"
                 : : "l"(lock) : "memory");
}

__global__ void ptx_lock_demo(unsigned int* d_lock, unsigned int* d_counter) {
    int tid = threadIdx.x;

    // 获取锁
    ptx_spin_lock(d_lock);

    // 临界区
    unsigned int old = *d_counter;
    *d_counter = old + 1;

    if (tid == 0) {
        printf("线程 0 获取锁，counter=%u\n", *d_counter);
    }

    // 释放锁
    ptx_spin_unlock(d_lock);
}

// ============================================================================
// 示例 7: 使用 PTX 实现高精度计时
// ============================================================================

/**
 * 使用 PTX 实现高精度计时器
 */
__device__ __forceinline__ unsigned long long ptx_clock64() {
    unsigned long long clock;
    asm volatile("mov.u64 %0, %clock64;" : "=l"(clock));
    return clock;
}

/**
 * 使用 PTX 实现 SM 时钟 (更精确)
 */
__device__ __forceinline__ unsigned int ptx_smem_clock() {
    unsigned int clock;
    asm volatile("mov.u32 %0, %clock;" : "=r"(clock));
    return clock;
}

__global__ void ptx_timing_demo() {
    int tid = threadIdx.x;

    if (tid == 0) {
        unsigned long long start = ptx_clock64();

        // 执行计算
        volatile float sum = 0.0f;
        for (int i = 0; i < 10000; i++) {
            sum += i * 0.5f;
        }

        unsigned long long end = ptx_clock64();

        printf("\n高精度计时演示:\n");
        printf("  开始时钟：%llu\n", start);
        printf("  结束时钟：%llu\n", end);
        printf("  消耗周期：%llu\n", end - start);
        printf("  迭代次数：10000\n");
        printf("  每迭代周期：%.2f\n", (double)(end - start) / 10000);
    }
}

// ============================================================================
// 示例 8: 使用 PTX 实现 warp 级原语
// ============================================================================

/**
 * warp 级投票 (ballot)
 * 对应的 PTX 指令：mov.u32 %r, %ballot;
 */
__device__ __forceinline__ unsigned int ptx_ballot(bool pred) {
    unsigned int result;
    asm volatile("mov.u32 %0, %ballot;" : "=r"(result) : "r"(pred ? 1 : 0));
    return result;
}

/**
 * warp 级广播
 * 对应的 PTX 指令：shfl.sync.bfly
 */
__device__ __forceinline__ float ptx_shfl_sync(float val, int src_lane) {
    float result;
    asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 31, -1;"
                 : "=f"(result)
                 : "f"(val), "r"(src_lane));
    return result;
}

/**
 * warp 级求和
 */
__device__ __forceinline__ float ptx_warp_sum(float val) {
    float result = val;

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other;
        asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 31, -1;"
                     : "=f"(other)
                     : "f"(result), "r"(offset));
        result += other;
    }

    return result;
}

__global__ void ptx_warp_primitive_demo() {
    int tid = threadIdx.x;
    int lane_id = tid % 32;

    float val = lane_id * 0.5f;

    // warp 级求和
    float warp_sum = ptx_warp_sum(val);

    if (lane_id == 0) {
        printf("Warp %d: sum(0..31)*0.5 = %.2f\n", tid / 32, warp_sum);
    }
}

// ============================================================================
// 验证函数
// ============================================================================

__global__ void verify_fast_math_kernel() {
    for (int i = 0; i < 8; i++) {
        float x = (i + 1) * 0.1f;

        float fast_s = fast_sqrt(x);
        float std_s = sqrtf(x);

        float fast_r = fast_rcp(x);
        float std_r = 1.0f / x;

        float fast_sn = fast_sin(x);
        float std_sn = sinf(x);

        printf("x=%.2f: sqrt(快=%.6f,标=%.6f), rcp(快=%.6f,标=%.6f), sin(快=%.6f,标=%.6f)\n",
               x, fast_s, std_s, fast_r, std_r, fast_sn, std_sn);
    }
}

void verify_fast_math() {
    printf("\n----------------------------------------\n");
    printf("  验证快速数学库\n");
    printf("----------------------------------------\n");

    verify_fast_math_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}

void verify_bitwise_ops() {
    printf("\n----------------------------------------\n");
    printf("  验证位操作\n");
    printf("----------------------------------------\n");

    dim3 block(32);
    dim3 grid(1);
    ptx_bitwise_demo<<<grid, block>>>();
    cudaDeviceSynchronize();
}

void verify_matrix_transpose() {
    printf("\n----------------------------------------\n");
    printf("  验证矩阵转置\n");
    printf("----------------------------------------\n");

    int width = 64, height = 64;
    std::vector<float> h_in(width * height);
    std::vector<float> h_out(width * height);

    // 初始化输入矩阵
    for (int i = 0; i < width * height; i++) {
        h_in[i] = i * 0.5f;
    }

    // 分配设备内存
    float *d_in, *d_out;
    cudaMalloc(&d_in, width * height * sizeof(float));
    cudaMalloc(&d_out, width * height * sizeof(float));

    cudaMemcpy(d_in, h_in.data(), width * height * sizeof(float), cudaMemcpyHostToDevice);

    // 启动 kernel
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((width + TILE_SIZE - 1) / TILE_SIZE,
              (height + TILE_SIZE - 1) / TILE_SIZE);

    ptx_matrix_transpose<<<grid, block>>>(d_out, d_in, width, height);
    cudaDeviceSynchronize();

    // 拷贝结果回主机
    cudaMemcpy(h_out.data(), d_out, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    // 验证结果
    bool passed = true;
    for (int i = 0; i < width && passed; i++) {
        for (int j = 0; j < height && passed; j++) {
            float expected = h_in[j * width + i];
            float actual = h_out[i * height + j];
            if (fabsf(expected - actual) > 1e-5f) {
                passed = false;
            }
        }
    }

    printf("矩阵转置验证：%s\n", passed ? "通过 ✓" : "失败 ✗");
    printf("验证点：[0][0]=%.2f, [0][1]=%.2f, [1][0]=%.2f\n",
           h_out[0], h_out[1], h_out[height]);

    cudaFree(d_in);
    cudaFree(d_out);
}

void verify_warp_primitives() {
    printf("\n----------------------------------------\n");
    printf("  验证 Warp 级原语\n");
    printf("----------------------------------------\n");

    dim3 block(64);  // 2 个 warp
    dim3 grid(1);
    ptx_warp_primitive_demo<<<grid, block>>>();
    cudaDeviceSynchronize();
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("========================================\n");
    printf("  PTX 内联汇编实战示例\n");
    printf("========================================\n");

    // 检查 GPU
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("\nGPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("SM 数量：%d\n", prop.multiProcessorCount);

    // 示例 1: 快速数学库
    printf("\n----------------------------------------\n");
    printf("  示例 1: PTX 快速数学库\n");
    printf("----------------------------------------\n");

    verify_fast_math();

    // 示例 2: SIMD 优化
    printf("\n----------------------------------------\n");
    printf("  示例 2: PTX SIMD 向量加法\n");
    printf("----------------------------------------\n");

    int n = 256;
    std::vector<float> h_a(n, 1.0f), h_b(n, 2.0f), h_out(n);
    float *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));

    cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block2(32);
    dim3 grid2((n + 32 * 4 - 1) / (32 * 4));
    ptx_simd_add<<<grid2, block2>>>(d_out, d_a, d_b, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

    printf("SIMD 加法验证：前 8 个元素\n");
    for (int i = 0; i < 8; i++) {
        printf("  [%d] = %.2f (预期：3.00)\n", i, h_out[i]);
    }

    // 示例 3: 位操作
    printf("\n----------------------------------------\n");
    printf("  示例 3: PTX 位操作优化\n");
    printf("----------------------------------------\n");

    verify_bitwise_ops();

    // 示例 4: 矩阵转置
    printf("\n----------------------------------------\n");
    printf("  示例 4: PTX 矩阵转置\n");
    printf("----------------------------------------\n");

    verify_matrix_transpose();

    // 示例 5: Warp 级原语
    printf("\n----------------------------------------\n");
    printf("  示例 5: PTX Warp 级原语\n");
    printf("----------------------------------------\n");

    verify_warp_primitives();

    // 示例 6: 高精度计时
    printf("\n----------------------------------------\n");
    printf("  示例 6: PTX 高精度计时\n");
    printf("----------------------------------------\n");

    dim3 block6(32);
    dim3 grid6(1);
    ptx_timing_demo<<<grid6, block6>>>();
    cudaDeviceSynchronize();

    // 总结
    printf("\n========================================\n");
    printf("  PTX 内联汇编实战示例完成!\n");
    printf("========================================\n");
    printf("\n知识点总结:\n");
    printf("1. 快速数学：sqrt.approx, rcp.approx, sin.approx\n");
    printf("2. SIMD 优化：并行加载/存储 4 个元素\n");
    printf("3. FMA 优化：fma.rn.f32 链式计算\n");
    printf("4. 位操作：popc, clz, ffs, brev\n");
    printf("5. 矩阵转置：共享内存 + PTX 加载存储\n");
    printf("6. 自旋锁：atom.cas.u32 实现\n");
    printf("7. 高精度计时：%clock64 寄存器\n");
    printf("8. Warp 原语：shfl.sync, ballot\n");

    // 清理
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    return 0;
}
