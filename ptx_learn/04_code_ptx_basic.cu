// PTX 基础代码示例

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

// ============================================================================
// 示例 1: 使用内联 PTX 读取特殊寄存器
// ============================================================================

/**
 * 读取线程索引和块索引
 * 对应的 PTX 指令：mov.u32 %r, %tid.x;
 */
__device__ int get_thread_idx() {
    int tid;
    asm volatile("mov.u32 %0, %tid.x;" : "=r"(tid));
    return tid;
}

__device__ int get_block_idx() {
    int bid;
    asm volatile("mov.u32 %0, %ctaid.x;" : "=r"(bid));
    return bid;
}

__device__ int get_block_dim() {
    int bdim;
    asm volatile("mov.u32 %0, %ntid.x;" : "=r"(bdim));
    return bdim;
}

__global__ void ptx_register_demo() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_ptx = get_thread_idx();
    int bid_ptx = get_block_idx();
    int bdim_ptx = get_block_dim();

    if (idx < 8) {
        printf("线程 %d: PTX tid=%d, bid=%d, bdim=%d\n",
               idx, tid_ptx, bid_ptx, bdim_ptx);
    }
}

// ============================================================================
// 示例 2: 使用内联 PTX 读取时钟
// ============================================================================

/**
 * 读取 GPU 时钟周期
 * 对应的 PTX 指令：mov.u64 %rd, %clock64;
 */
__device__ unsigned long long get_clock64() {
    unsigned long long clock;
    asm volatile("mov.u64 %0, %clock64;" : "=l"(clock));
    return clock;
}

__global__ void ptx_clock_demo() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned long long start_clock = get_clock64();

    // 做一些工作
    volatile float sum = 0.0f;
    for (int i = 0; i < 100; i++) {
        sum += i * 0.5f;
    }

    unsigned long long end_clock = get_clock64();

    if (idx == 0) {
        printf("\n时钟周期演示:\n");
        printf("  开始时钟：%llu\n", start_clock);
        printf("  结束时钟：%llu\n", end_clock);
        printf("  消耗周期：%llu\n", end_clock - start_clock);
    }
}

// ============================================================================
// 示例 3: 使用内联 PTX 进行基本算术
// ============================================================================

/**
 * 使用 PTX 实现向量加法
 * 对应的 PTX 指令：add.f32 %f, %f1, %f2;
 */
__device__ float ptx_add(float a, float b) {
    float result;
    asm volatile("add.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
    return result;
}

/**
 * 使用 PTX 实现向量乘法
 * 对应的 PTX 指令：mul.f32 %f, %f1, %f2;
 */
__device__ float ptx_mul(float a, float b) {
    float result;
    asm volatile("mul.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
    return result;
}

/**
 * 使用 PTX 实现乘加操作 (FMA)
 * 对应的 PTX 指令：fma.rn.f32 %f, %f1, %f2, %f3;
 */
__device__ float ptx_fma(float a, float b, float c) {
    float result;
    asm volatile("fma.rn.f32 %0, %1, %2, %3;"
                 : "=f"(result) : "f"(a), "f"(b), "f"(c));
    return result;
}

__global__ void ptx_arithmetic_demo(float* d_result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float a = idx * 0.5f;
        float b = idx * 0.25f;
        float c = 1.0f;

        // 使用 PTX 内联算术
        float add_result = ptx_add(a, b);
        float mul_result = ptx_mul(a, b);
        float fma_result = ptx_fma(a, b, c);

        d_result[idx] = add_result + mul_result + fma_result;

        if (idx < 4) {
            printf("线程 %d: a=%.2f, b=%.2f\n", idx, a, b);
            printf("  add=%.4f, mul=%.4f, fma=%.4f\n",
                   add_result, mul_result, fma_result);
        }
    }
}

// ============================================================================
// 示例 4: 使用 PTX 进行类型转换
// ============================================================================

/**
 * float 到 int 的转换
 * 对应的 PTX 指令：cvt.u32.f32 %r, %f;
 */
__device__ int float_to_uint(float f) {
    int result;
    asm volatile("cvt.rzi.u32.f32 %0, %1;" : "=r"(result) : "f"(f));
    return result;
}

/**
 * int 到 float 的转换
 * 对应的 PTX 指令：cvt.f32.u32 %f, %r;
 */
__device__ float int_to_float(int i) {
    float result;
    asm volatile("cvt.rn.f32.s32 %0, %1;" : "=f"(result) : "r"(i));
    return result;
}

/**
 * 位模式重解释 (float <-> int)
 * 对应的 PTX 指令：mov.b32 %r, %f;
 */
__device__ int float_as_int(float f) {
    int result;
    asm volatile("mov.b32 %0, %1;" : "=r"(result) : "f"(f));
    return result;
}

__device__ float int_as_float(int i) {
    float result;
    asm volatile("mov.b32 %0, %1;" : "=f"(result) : "r"(i));
    return result;
}

__global__ void ptx_convert_demo() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < 8) {
        float f = idx * 1.5f + 0.5f;

        int u = float_to_uint(f);
        int f2i = float_to_uint(f);
        float i2f = int_as_float(idx);
        int f_bits = float_as_int(f);

        printf("线程 %d: float=%.2f -> uint=%d, float_bits=0x%08x\n",
               idx, f, u, f_bits);
    }
}

// ============================================================================
// 示例 5: 使用 PTX 进行比较操作
// ============================================================================

/**
 * 浮点比较
 * 使用 CUDA 内置比较
 */
__device__ __forceinline__ bool ptx_less_than(float a, float b) {
    return a < b;
}

/**
 * 等于比较
 * 使用 CUDA 内置比较
 */
__device__ __forceinline__ bool ptx_equal(float a, float b) {
    return a == b;
}

__global__ void ptx_compare_demo() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < 8) {
        float a = idx * 0.5f;
        float b = 2.0f;

        bool lt = ptx_less_than(a, b);
        bool eq = ptx_equal(a, b);

        printf("线程 %d: %.2f < %.2f ? %s, %.2f == %.2f ? %s\n",
               idx, a, b, lt ? "true" : "false",
               a, b, eq ? "true" : "false");
    }
}

// ============================================================================
// 示例 6: 使用 PTX 进行位操作
// ============================================================================

/**
 * 按位与
 * 对应的 PTX 指令：and.b32 %r, %r1, %r2;
 */
__device__ unsigned int ptx_and(unsigned int a, unsigned int b) {
    unsigned int result;
    asm volatile("and.b32 %0, %1, %2;" : "=r"(result) : "r"(a), "r"(b));
    return result;
}

/**
 * 按位或
 * 对应的 PTX 指令：or.b32 %r, %r1, %r2;
 */
__device__ unsigned int ptx_or(unsigned int a, unsigned int b) {
    unsigned int result;
    asm volatile("or.b32 %0, %1, %2;" : "=r"(result) : "r"(a), "r"(b));
    return result;
}

/**
 * 按位异或
 * 对应的 PTX 指令：xor.b32 %r, %r1, %r2;
 */
__device__ unsigned int ptx_xor(unsigned int a, unsigned int b) {
    unsigned int result;
    asm volatile("xor.b32 %0, %1, %2;" : "=r"(result) : "r"(a), "r"(b));
    return result;
}

/**
 * 左移
 * 对应的 PTX 指令：shl.b32 %r, %r1, %r2;
 */
__device__ unsigned int ptx_shl(unsigned int a, int shift) {
    unsigned int result;
    asm volatile("shl.b32 %0, %1, %2;" : "=r"(result) : "r"(a), "r"(shift));
    return result;
}

/**
 * 右移
 * 对应的 PTX 指令：shr.b32 %r, %r1, %r2;
 */
__device__ unsigned int ptx_shr(unsigned int a, int shift) {
    unsigned int result;
    asm volatile("shr.b32 %0, %1, %2;" : "=r"(result) : "r"(a), "r"(shift));
    return result;
}

/**
 * 计算 1 的个数
 * 对应的 PTX 指令：popc.b32 %r, %r1;
 */
__device__ unsigned int ptx_popc(unsigned int a) {
    unsigned int result;
    asm volatile("popc.b32 %0, %1;" : "=r"(result) : "r"(a));
    return result;
}

__global__ void ptx_bitwise_demo() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < 4) {
        unsigned int a = (idx + 1) * 0x11;
        unsigned int b = 0x0F;

        unsigned int and_result = ptx_and(a, b);
        unsigned int or_result = ptx_or(a, b);
        unsigned int xor_result = ptx_xor(a, b);
        unsigned int shl_result = ptx_shl(a, 2);
        unsigned int shr_result = ptx_shr(a, 2);
        unsigned int popc_result = ptx_popc(a);

        printf("线程 %d: a=0x%08x, b=0x%08x\n", idx, a, b);
        printf("  and=0x%02x, or=0x%02x, xor=0x%02x\n",
               and_result, or_result, xor_result);
        printf("  shl=0x%08x, shr=0x%08x, popc=%u\n",
               shl_result, shr_result, popc_result);
    }
}

// ============================================================================
// 示例 7: 使用 PTX 实现快速数学函数
// ============================================================================

/**
 * 近似平方根
 * 对应的 PTX 指令：sqrt.approx.ftz.f32 %f, %f1;
 */
__device__ float ptx_sqrt(float x) {
    float result;
    asm volatile("sqrt.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
}

/**
 * 近似倒数
 * 对应的 PTX 指令：rcp.approx.ftz.f32 %f, %f1;
 */
__device__ float ptx_rcp(float x) {
    float result;
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
}

/**
 * 近似正弦
 * 对应的 PTX 指令：sin.approx.f32 %f, %f1;
 */
__device__ float ptx_sin(float x) {
    float result;
    asm volatile("sin.approx.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
}

__global__ void ptx_math_demo() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < 8) {
        float x = (idx + 1) * 0.5f;

        float sqrt_result = ptx_sqrt(x);
        float rcp_result = ptx_rcp(x);
        float sin_result = ptx_sin(x);

        // 标准库对比
        float sqrt_std = sqrtf(x);
        float rcp_std = 1.0f / x;
        float sin_std = sinf(x);

        printf("线程 %d: x=%.2f\n", idx, x);
        printf("  PTX sqrt=%.6f, std sqrt=%.6f, diff=%.6e\n",
               sqrt_result, sqrt_std, fabsf(sqrt_result - sqrt_std));
        printf("  PTX rcp=%.6f, std rcp=%.6f, diff=%.6e\n",
               rcp_result, rcp_std, fabsf(rcp_result - rcp_std));
        printf("  PTX sin=%.6f, std sin=%.6f, diff=%.6e\n",
               sin_result, sin_std, fabsf(sin_result - sin_std));
    }
}

// ============================================================================
// CPU 验证函数
// ============================================================================

void verify_arithmetic() {
    printf("\n========================================\n");
    printf("  验证 PTX 算术运算\n");
    printf("========================================\n");

    float* d_result;
    int n = 16;
    cudaMalloc(&d_result, n * sizeof(float));

    dim3 block(16);
    dim3 grid(1);
    ptx_arithmetic_demo<<<grid, block>>>(d_result, n);
    cudaDeviceSynchronize();

    cudaFree(d_result);
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("========================================\n");
    printf("  PTX 基础示例代码\n");
    printf("========================================\n");

    // 检查 GPU
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("\nGPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);

    // 示例 1: 寄存器演示
    printf("\n----------------------------------------\n");
    printf("  示例 1: PTX 特殊寄存器演示\n");
    printf("----------------------------------------\n");

    dim3 block1(32);
    dim3 grid1(2);
    ptx_register_demo<<<grid1, block1>>>();
    cudaDeviceSynchronize();

    // 示例 2: 时钟演示
    printf("\n----------------------------------------\n");
    printf("  示例 2: PTX 时钟周期演示\n");
    printf("----------------------------------------\n");

    dim3 block2(1);
    dim3 grid2(1);
    ptx_clock_demo<<<grid2, block2>>>();
    cudaDeviceSynchronize();

    // 示例 3: 算术运算
    printf("\n----------------------------------------\n");
    printf("  示例 3: PTX 算术运算演示\n");
    printf("----------------------------------------\n");

    verify_arithmetic();

    // 示例 4: 类型转换
    printf("\n----------------------------------------\n");
    printf("  示例 4: PTX 类型转换演示\n");
    printf("----------------------------------------\n");

    dim3 block4(16);
    dim3 grid4(1);
    ptx_convert_demo<<<grid4, block4>>>();
    cudaDeviceSynchronize();

    // 示例 5: 比较操作
    printf("\n----------------------------------------\n");
    printf("  示例 5: PTX 比较操作演示\n");
    printf("----------------------------------------\n");

    dim3 block5(16);
    dim3 grid5(1);
    ptx_compare_demo<<<grid5, block5>>>();
    cudaDeviceSynchronize();

    // 示例 6: 位操作
    printf("\n----------------------------------------\n");
    printf("  示例 6: PTX 位操作演示\n");
    printf("----------------------------------------\n");

    dim3 block6(16);
    dim3 grid6(1);
    ptx_bitwise_demo<<<grid6, block6>>>();
    cudaDeviceSynchronize();

    // 示例 7: 数学函数
    printf("\n----------------------------------------\n");
    printf("  示例 7: PTX 数学函数演示\n");
    printf("----------------------------------------\n");

    dim3 block7(16);
    dim3 grid7(1);
    ptx_math_demo<<<grid7, block7>>>();
    cudaDeviceSynchronize();

    // 总结
    printf("\n========================================\n");
    printf("  PTX 基础示例完成!\n");
    printf("========================================\n");
    printf("\n知识点总结:\n");
    printf("1. 特殊寄存器：%%%%tid, %%%%ctaid, %%%%ntid, %%%%clock64\n");
    printf("2. 算术指令：add.f32, mul.f32, fma.rn.f32\n");
    printf("3. 类型转换：cvt.u32.f32, cvt.f32.u32, mov.b32\n");
    printf("4. 比较指令：setp.lt.f32, setp.eq.f32\n");
    printf("5. 位操作：and.b32, or.b32, shl.b32, popc.u32\n");
    printf("6. 数学函数：sqrt.approx, rcp.approx, sin.approx\n");

    return 0;
}
