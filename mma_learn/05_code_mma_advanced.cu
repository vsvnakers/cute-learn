// MMA 进阶代码示例
// 演示 Warp 级 MMA、多精度支持和性能基准测试

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

// ============================================================================
// 示例 1: Warp 级 MMA 概念
// ============================================================================

constexpr int WARP_M = 16;
constexpr int WARP_N = 16;
constexpr int WARP_K = 8;

/**
 * Warp-level MMA 概念演示
 * 每个 warp 负责一个矩阵乘法块
 */
__global__ void warp_mma_concept_demo() {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    if (lane_id == 0 && warp_id == 0) {
        printf("Warp 级 MMA 概念演示:\n");
        printf("  WARP_M=%d, WARP_N=%d, WARP_K=%d\n", WARP_M, WARP_N, WARP_K);
        printf("  每个 warp 负责：%d×%d 输出块\n", WARP_M, WARP_N);
        printf("  每线程负责：2×2 输出 (使用 4 个 FP32 寄存器)\n");
        printf("  32 线程共同完成 16×16=256 个输出元素\n");
    }
}

// ============================================================================
// 示例 2: 使用 PTX 读取特殊寄存器
// ============================================================================

__global__ void mma_special_registers_demo() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx == 0) {
        printf("\n特殊寄存器演示:\n");

        // 读取 SM ID
        unsigned int smid;
        asm volatile("mov.u32 %0, %smid;" : "=r"(smid));
        printf("  SM ID: %u\n", smid);

        // 读取 Warp ID
        unsigned int warp_id;
        asm volatile("mov.u32 %0, %warpid;" : "=r"(warp_id));
        printf("  Warp ID: %u\n", warp_id);

        // 读取 Lane ID
        unsigned int lane_id;
        asm volatile("mov.u32 %0, %laneid;" : "=r"(lane_id));
        printf("  Lane ID: %u\n", lane_id);
    }
}

// ============================================================================
// 示例 3: 多精度 MMA 支持
// ============================================================================

__global__ void multi_precision_mma_demo(int compute_major) {
    int tid = threadIdx.x;

    if (tid == 0) {
        printf("\n多精度 MMA 支持:\n");

        // FP16
        half h_a = __float2half(2.0f);
        half h_b = __float2half(3.0f);
        half h_c = __hmul(h_a, h_b);
        printf("  FP16: %.2f × %.2f = %.2f\n",
               __half2float(h_a), __half2float(h_b), __half2float(h_c));

        // BF16 (如果支持)
        if (compute_major >= 8) {
        __nv_bfloat16 bf_a = __float2bfloat16(2.0f);
        __nv_bfloat16 bf_b = __float2bfloat16(3.0f);
        __nv_bfloat16 bf_c = __hmul(bf_a, bf_b);
        printf("  BF16: %.2f × %.2f = %.2f\n",
               __bfloat162float(bf_a), __bfloat162float(bf_b),
               __bfloat162float(bf_c));
        } else {
        printf("  BF16: 不支持 (需要 SM80+)\n");
        }

        // FP32
        float f_a = 2.0f;
        float f_b = 3.0f;
        float f_c = f_a * f_b;
        printf("  FP32: %.2f × %.2f = %.2f\n", f_a, f_b, f_c);

        // FP64 (如果支持)
        if (compute_major >= 6) {
            double d_a = 2.0;
            double d_b = 3.0;
            double d_c = d_a * d_b;
            printf("  FP64: %.2f × %.2f = %.2f\n", d_a, d_b, d_c);
        }
    }
}

// ============================================================================
// 示例 4: MMA 指令格式演示
// ============================================================================

__global__ void mma_instruction_format_demo() {
    int tid = threadIdx.x;

    if (tid == 0) {
        printf("\nMMA 指令格式演示:\n");
        printf("  完整格式:\n");
        printf("    mma.sync.aligned.m<M>n<N>k<K>.<Dtype>.<Atype>.<Btype>.<Ctype>\n");
        printf("\n");
        printf("  常见配置:\n");
        printf("    FP16 GEMM: m16n8k8.row.col.f32.f16.f16.f32\n");
        printf("    BF16 GEMM: m16n8k16.row.col.f32.bf16.bf16.f32\n");
        printf("    TF32 GEMM: m16n8k8.row.col.f32.tf32.tf32.f32\n");
        printf("    FP8 GEMM:  m64n8k32.row.col.f32.f8.f8.f32 (SM90+)\n");
        printf("    FP64 GEMM: m8n8k4.row.col.f64.f64.f64.f64\n");
    }
}

// ============================================================================
// 示例 5: 性能基准测试
// ============================================================================

template<typename Func>
float benchmark_kernel(Func kernel_func, int iterations = 100) {
    // 预热
    kernel_func();
    cudaDeviceSynchronize();

    // 计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        kernel_func();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    elapsed_time /= iterations;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return elapsed_time;
}

__global__ void dummy_kernel() {
    // 空 kernel 用于基准测试
    volatile float sum = 0.0f;
    for (int i = 0; i < 100; i++) {
        sum += i * 0.5f;
    }
}

void run_performance_benchmark() {
    printf("\n性能基准测试:\n");

    // 测量空 kernel 的时间
    float time = benchmark_kernel([]() {
        dim3 block(256);
        dim3 grid(32);
        dummy_kernel<<<grid, block>>>();
    }, 100);

    printf("  Kernel 启动延迟：%.3f us\n", time * 1000);
    printf("  (实际 MMA 性能需要完整的 GEMM 实现)\n");
}

// ============================================================================
// 示例 6: 简单 GEMM 性能测试
// ============================================================================

__global__ void simple_gemm_kernel(float* d_C, const float* d_A, const float* d_B,
                                   int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += d_A[row * K + k] * d_B[k * N + col];
        }
        d_C[row * N + col] = sum;
    }
}

void benchmark_gemm(int M, int N, int K) {
    // 分配内存
    std::vector<float> h_A(M * K, 1.0f);
    std::vector<float> h_B(K * N, 1.0f);
    std::vector<float> h_C(M * N);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

    // 配置
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    // 预热
    simple_gemm_kernel<<<grid, block>>>(d_C, d_A, d_B, M, N, K);
    cudaDeviceSynchronize();

    // 计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int iterations = 10;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        simple_gemm_kernel<<<grid, block>>>(d_C, d_A, d_B, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    elapsed_time /= iterations;

    // 计算 GFLOPS
    float gflops = (2.0f * M * N * K) / (elapsed_time * 1e-3f) / 1e9f;

    printf("GEMM 性能 (M=%d, N=%d, K=%d):\n", M, N, K);
    printf("  时间：%.3f ms\n", elapsed_time);
    printf("  GFLOPS: %.2f\n", gflops);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("========================================\n");
    printf("  MMA 进阶示例代码\n");
    printf("========================================\n");

    // 检查 GPU
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("\nGPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);

    if (prop.major < 7) {
        printf("错误：MMA 需要 SM70+ 架构\n");
        return 1;
    }

    // 示例 1: Warp 级 MMA
    printf("\n----------------------------------------\n");
    printf("  示例 1: Warp 级 MMA 演示\n");
    printf("----------------------------------------\n");

    dim3 block1(32);
    warp_mma_concept_demo<<<1, block1>>>();
    cudaDeviceSynchronize();

    // 示例 2: 特殊寄存器
    printf("\n----------------------------------------\n");
    printf("  示例 2: 特殊寄存器演示\n");
    printf("----------------------------------------\n");

    dim3 block2(1);
    mma_special_registers_demo<<<1, block2>>>();
    cudaDeviceSynchronize();

    // 示例 3: 多精度
    printf("\n----------------------------------------\n");
    printf("  示例 3: 多精度 MMA 演示\n");
    printf("----------------------------------------\n");

    dim3 block3(1);
    multi_precision_mma_demo<<<1, block3>>>(prop.major);
    cudaDeviceSynchronize();

    // 示例 4: 指令格式
    printf("\n----------------------------------------\n");
    printf("  示例 4: MMA 指令格式演示\n");
    printf("----------------------------------------\n");

    dim3 block4(1);
    mma_instruction_format_demo<<<1, block4>>>();
    cudaDeviceSynchronize();

    // 示例 5: 性能基准
    printf("\n----------------------------------------\n");
    printf("  示例 5: 性能基准测试\n");
    printf("----------------------------------------\n");

    run_performance_benchmark();
    benchmark_gemm(512, 512, 512);
    benchmark_gemm(1024, 1024, 1024);

    // 总结
    printf("\n========================================\n");
    printf("  MMA 进阶示例完成!\n");
    printf("========================================\n");
    printf("\n关键知识点:\n");
    printf("1. Warp 级 MMA：每个 warp 负责一个 tile\n");
    printf("2. 特殊寄存器：%smid, %warpid, %laneid\n");
    printf("3. 多精度支持：FP16/BF16/FP32/FP64\n");
    printf("4. MMA 指令格式：m<M>n<N>k<K>.<type>\n");

    return 0;
}
