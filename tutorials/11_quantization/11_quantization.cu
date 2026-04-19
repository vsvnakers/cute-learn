/**
 * 第十一课：低精度量化 - INT8/INT4/FP8
 *
 * 本课讲解使用 CUTE 实现低精度计算：
 * 1. INT8 GEMM
 * 2. INT4 打包
 * 3. FP8 (Hopper)
 *
 * 编译：nvcc -std=c++17 -arch=sm_80 11_quantization.cu -o 11_quantization
 */

#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>

using namespace cute;

// ============================================================================
// INT8 量化基础
// ============================================================================

void test_int8_basics() {
    std::cout << "=== INT8 量化基础 ===" << std::endl;

    // 量化原理
    std::cout << "量化原理:" << std::endl;
    std::cout << "  FP32 -> INT8:  scale = max(|FP32|) / 127" << std::endl;
    std::cout << "  INT8 -> FP32:  FP32 = INT8 * scale" << std::endl;

    // INT8 MMA 支持
    std::cout << "\nSM80 INT8 MMA:" << std::endl;
    std::cout << "  mma.sync.aligned.m8n8k16.s32.s8.s8.s32" << std::endl;
    std::cout << "  - A: 8x16 INT8" << std::endl;
    std::cout << "  - B: 16x8 INT8" << std::endl;
    std::cout << "  - C/D: 8x8 INT32" << std::endl;

    // 性能优势
    std::cout << "\nINT8 优势:" << std::endl;
    std::cout << "  - 2x 带宽效率 (vs FP16)" << std::endl;
    std::cout << "  - 4x 带宽效率 (vs FP32)" << std::endl;
    std::cout << "  - 适合推理场景" << std::endl;
}

// ============================================================================
// INT4 打包技术
// ============================================================================

void test_int4_packing() {
    std::cout << "\n=== INT4 打包 ===" << std::endl;

    // INT4 打包原理
    std::cout << "INT4 打包:" << std::endl;
    std::cout << "  - 2 个 INT4 打包到 1 个字节" << std::endl;
    std::cout << "  - 需要位操作解包" << std::endl;
    std::cout << "  - 4x 带宽效率 (vs FP32)" << std::endl;

    // 解包示例
    auto unpack_int4 = [] (uint8_t packed) -> std::pair<int8_t, int8_t> {
        int8_t low = (packed & 0x0F) - 8;   // 低 4 位
        int8_t high = (packed >> 4) - 8;    // 高 4 位
        return {low, high};
    };

    uint8_t test = 0xAB;
    auto [low, high] = unpack_int4(test);
    std::cout << "\n解包示例：0x" << std::hex << (int)test << std::dec
              << " -> low=" << (int)low << ", high=" << (int)high << std::endl;
}

// ============================================================================
// FP8 (Hopper)
// ============================================================================

void test_fp8_hopper() {
    std::cout << "\n=== FP8 (Hopper) ===" << std::endl;

    std::cout << "FP8 格式 (Hopper SM90):" << std::endl;
    std::cout << "  - E4M3: 4 位指数，3 位尾数 (最大 448)" << std::endl;
    std::cout << "  - E5M2: 5 位指数，2 位尾数 (类似 FP16)" << std::endl;

    std::cout << "\nFP8 优势:" << std::endl;
    std::cout << "  - 2x 带宽 (vs FP16)" << std::endl;
    std::cout << "  - 支持动态范围" << std::endl;
    std::cout << "  - 适合 LLM 训练" << std::endl;
}

// ============================================================================
// INT8 GEMM 示例
// ============================================================================

template<int BLOCK_SIZE>
__global__ void int8_gemm_kernel(
    const int8_t* A, const int8_t* B, int32_t* C,
    int M, int N, int K) {

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (row < M && col < N) {
        int32_t acc = 0;
        for (int k = 0; k < K; k++) {
            acc += (int32_t)A[row * K + k] * (int32_t)B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}

void test_int8_gemm() {
    std::cout << "\n=== INT8 GEMM 演示 ===" << std::endl;

    constexpr int M = 64, N = 64, K = 64;
    constexpr int BLOCK = 32;

    // 分配内存
    size_t size_AB = M * K * sizeof(int8_t);
    size_t size_C = M * N * sizeof(int32_t);

    int8_t *d_A, *d_B;
    int32_t *d_C;
    cudaMalloc(&d_A, size_AB);
    cudaMalloc(&d_B, size_AB);
    cudaMalloc(&d_C, size_C);

    // 初始化测试数据
    int8_t* h_A = new int8_t[M * K];
    int8_t* h_B = new int8_t[K * N];
    for (int i = 0; i < M * K; i++) h_A[i] = 2;
    for (int i = 0; i < K * N; i++) h_B[i] = 3;

    cudaMemcpy(d_A, h_A, size_AB, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_AB, cudaMemcpyHostToDevice);

    // 启动 Kernel
    dim3 block(BLOCK, BLOCK);
    dim3 grid((N + BLOCK - 1) / BLOCK, (M + BLOCK - 1) / BLOCK);
    int8_gemm_kernel<BLOCK><<<grid, block>>>(d_A, d_B, d_C, M, N, K);

    // 验证
    int32_t* h_C = new int32_t[M * N];
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    std::cout << "INT8 GEMM 结果:" << std::endl;
    std::cout << "  C[0] = " << h_C[0] << " (期望：" << K * 2 * 3 << ")" << std::endl;
    std::cout << "  验证：" << (h_C[0] == K * 6 ? "PASS" : "FAIL") << std::endl;

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// ============================================================================
// 量化参数传递
// ============================================================================

void test_quant_scales() {
    std::cout << "\n=== 量化参数 ===" << std::endl;

    // 量化需要传递 scale 和 zero point
    std::cout << "量化参数:" << std::endl;
    std::cout << "  scale:     FP32 = INT8 * scale" << std::endl;
    std::cout << "  zero_point: 偏移校正" << std::endl;

    // 每层量化 vs 每通道量化
    std::cout << "\n量化粒度:" << std::endl;
    std::cout << "  per-tensor: 整个张量一个 scale" << std::endl;
    std::cout << "  per-channel: 每通道一个 scale (更精确)" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  第十一课：低精度量化 INT8/INT4/FP8" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    test_int8_basics();
    test_int4_packing();
    test_fp8_hopper();
    test_int8_gemm();
    test_quant_scales();

    std::cout << "\n========================================" << std::endl;
    std::cout << "  第十一课完成！" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
